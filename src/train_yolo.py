"""COCO-pretrained, fold-isolated YOLO26 training with embedded provenance.

Ultralytics mAP selects best.pt as a diagnostic, not competition PQ. Evaluate
saved snapshots using src.infer_yolo and the official PQ implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

from .prepare_yolo import atomic_json, digest, prepare_dataset, split_identity


def checkpoint_provenance(model):
    provenance = getattr(model.model, "filament_provenance", None)
    if not isinstance(provenance, dict):
        raise ValueError("Checkpoint has no embedded official-data split provenance")
    return provenance


def validate_provenance(provenance, manifest, fold):
    expected = split_identity(manifest, fold)
    if provenance.get("fold") != fold or provenance.get("data_signature") != expected["data_signature"]:
        raise ValueError("Checkpoint fold/annotation/training membership does not match this validation split")
    if provenance.get("training_images") != expected["training_images"]:
        raise ValueError("Checkpoint training images do not match the declared split")
    if set(provenance["training_images"]) & set(expected["validation_images"]):
        raise ValueError("Checkpoint training images overlap validation images")
    return {"fold": fold, "data_signature": expected["data_signature"], "verification": "embedded_checkpoint"}


def install_memory_format(trainer, channels_last=True):
    """Install layout handling before model setup, compilation, optimizer and EMA.

    Ultralytics 8.4.211 converts model weights to NHWC automatically on CUDA,
    but its preprocessing leaves image batches NCHW. Preserve every standard
    preprocessing operation and convert only the returned four-dimensional
    image tensor. Explicitly handling setup also makes the fallback reversible.
    """
    import torch
    enabled = bool(channels_last and trainer.device.type == "cuda")
    memory_format = torch.channels_last if enabled else torch.contiguous_format
    trainer.args.channels_last = enabled
    trainer.filament_channels_last = enabled
    original_setup = trainer.setup_model
    original_preprocess = trainer.preprocess_batch

    def setup_with_layout():
        checkpoint = original_setup()
        # setup_model runs before BaseTrainer's device transfer and compile.
        trainer.model.to(memory_format=memory_format)
        return checkpoint

    def preprocess_with_layout(batch):
        batch = original_preprocess(batch)
        batch["img"] = batch["img"].contiguous(memory_format=memory_format)
        trainer.filament_last_batch_layout = {
            "channels_last": batch["img"].is_contiguous(memory_format=torch.channels_last),
            "contiguous": batch["img"].is_contiguous(),
            "stride": list(batch["img"].stride()),
            "shape": list(batch["img"].shape),
        }
        return batch

    trainer.setup_model = setup_with_layout
    trainer.preprocess_batch = preprocess_with_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    parser.add_argument("--data-dir")
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model", choices=["yolo26s-seg.pt", "yolo26m-seg.pt"], default="yolo26s-seg.pt")
    parser.add_argument("--weights-dir", default="weights/pretrained")
    parser.add_argument("--init", help="Fine-tune an own checkpoint; its training images must be allowed in this split")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--nominal-batch", type=int, help="Nominal optimizer batch for gradient accumulation; defaults to --batch")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--mask-ratio", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                        help="Use NHWC model/EMA weights and input batches on CUDA (default: enabled)")
    parser.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16",
                        help="Mixed precision: bf16 for GB10, fp16 for T4, or off for float32")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data and CPU-load initialization weights; no GPU use")
    parser.add_argument("--benchmark-steps", type=int, default=0, help="Stop after 1-2 train batches, without validation/save")
    args = parser.parse_args()
    if args.benchmark_steps not in (0, 1, 2):
        raise ValueError("Benchmark is limited to two training batches")
    if args.resume and args.init:
        raise ValueError("Use either --resume or --init")
    if args.nominal_batch is not None and args.nominal_batch <= 0:
        raise ValueError("--nominal-batch must be positive")
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    report = prepare_dataset(args.manifest, args.data_dir, args.fold)
    manifest = json.loads(Path(args.manifest).read_text())
    run = Path(args.run).resolve()
    weights = Path(args.weights_dir).resolve() / args.model
    weights.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        weights = run / "weights" / "last.pt"
        model = YOLO(str(weights))
        provenance = checkpoint_provenance(model)
        validate_provenance(provenance, manifest, args.fold)
    else:
        if not args.prepare_only and (run / "provenance.json").exists():
            raise ValueError("Run already exists; use --resume or choose a new --run")
        source_provenance = None
        if args.init:
            weights = Path(args.init).resolve(strict=True)
            model = YOLO(str(weights))
            source_provenance = checkpoint_provenance(model)
            allowed = set(report["training_images"])
            source_images = source_provenance.get("training_images")
            if (source_provenance.get("benchmark") or not source_images or
                not set(source_images) <= allowed or
                source_provenance.get("annotation_sha256") != report["annotation_sha256"]):
                raise ValueError("Initialization checkpoint is not verified or has trained on excluded images")
            if model.task != "segment" or len(model.names) != 1 or model.model.end2end:
                raise ValueError("Expected an own one-class one-to-many segmentation checkpoint")
        else:
            model = YOLO(str(weights))
            if model.task != "segment" or len(model.names) != 80:
                raise ValueError("Fresh fold training must initialize generic 80-class COCO segmentation weights")
            if getattr(model.model, "filament_provenance", None):
                raise ValueError("Generic initialization unexpectedly contains task-specific training provenance")
        # Train the pretrained one-to-many branch, with conventional NMS at IoU .5.
        # Updating YAML is essential: Ultralytics reconstructs the train model.
        model.model.yaml["end2end"] = False
        model.model.end2end = False
        provenance = {"schema": 1, "architecture": source_provenance["architecture"] if source_provenance else args.model,
            "fold": args.fold,
            "data_signature": report["data_signature"], "manifest_sha256": report["manifest_sha256"],
            "annotation_sha256": report["annotation_sha256"],
            "training_images": report["training_images"], "validation_images": report["validation_images"],
            "labels_sha256": report["labels_sha256"], "initial_weights_sha256": digest(weights),
            "initial_weights_source": str(weights) if source_provenance else
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/" + args.model,
            "initialization": "own official-only checkpoint" if source_provenance else "generic COCO only",
            "parent_provenance": source_provenance, "ultralytics": ultralytics.__version__,
            "end2end": False, "mask_ratio": args.mask_ratio, "overlap_mask": False,
            "imgsz": args.imgsz, "seed": args.seed, "channels_last": args.channels_last,
            "amp": args.amp, "nominal_batch": args.nominal_batch or args.batch,
            "benchmark": bool(args.benchmark_steps)}
    if args.prepare_only:
        print(json.dumps({"status": "prepared_cpu_only", "weights": str(weights),
            "parameters": sum(p.numel() for p in model.model.parameters()),
            "train_images": report["train_unique_images"], "val_images": report["val_unique_images"],
            "train_copies": report["train_annotator_records"], "data_yaml": report["data_yaml"],
            "minimum_conversion_iou": report["minimum_conversion_iou"],
            "launch": shlex.join([sys.executable, "-u", "-m", "src.train_yolo",
                                  *[arg for arg in sys.argv[1:] if arg != "--prepare-only"]])}, indent=2))
        return
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / "provenance.json", provenance)
    atomic_json(run / "launch_args.json", vars(args))

    def attach_provenance(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        # EMA is constructed from the already-converted model after compile;
        # its unwrapped copy can safely be normalized without changing graphs.
        memory_format = torch.channels_last if trainer.filament_channels_last else torch.contiguous_format
        trainer.ema.ema.to(memory_format=memory_format)
        training_model = unwrap_model(trainer.model)
        provenance["channels_last"] = trainer.filament_channels_last
        provenance["amp"] = trainer.args.amp
        provenance["nominal_batch"] = trainer.args.nbs
        atomic_json(run / "provenance.json", provenance)
        training_model.filament_provenance = provenance
        trainer.ema.ema.filament_provenance = provenance
        if training_model.end2end:
            raise ValueError("Training unexpectedly uses the end-to-end branch")
        for name, module in (("model", training_model), ("ema", trainer.ema.ema)):
            tensors = [p for p in module.parameters() if p.ndim == 4]
            if not all(p.is_contiguous(memory_format=memory_format) for p in tensors):
                raise ValueError(f"{name} weights do not use the requested memory format")

    model.add_callback("on_pretrain_routine_start", lambda trainer: install_memory_format(trainer, args.channels_last))
    model.add_callback("on_pretrain_routine_end", attach_provenance)
    if args.benchmark_steps:
        class BenchmarkComplete(Exception):
            pass
        benchmark = {"batches": 0, "started": None, "batch_seconds": []}

        def start_batch(trainer):
            torch.cuda.synchronize() if trainer.device.type == "cuda" else None
            benchmark["started"] = time.perf_counter()

        def end_batch(trainer):
            torch.cuda.synchronize() if trainer.device.type == "cuda" else None
            benchmark["batches"] += 1
            benchmark["batch_seconds"].append(time.perf_counter() - benchmark["started"])
            if benchmark["batches"] >= args.benchmark_steps:
                atomic_json(run / "benchmark.json", {**benchmark, "started": None,
                    "batch": args.batch, "imgsz": args.imgsz,
                    "channels_last": trainer.filament_channels_last,
                    "input_layout": trainer.filament_last_batch_layout,
                    "loss": trainer.loss.detach().float().cpu().tolist(),
                    "loss_items": {name: value.detach().float().cpu().tolist()
                                   for name, value in trainer.loss_items.items()},
                    "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if trainer.device.type == "cuda" else 0})
                raise BenchmarkComplete

        model.add_callback("on_train_batch_start", start_batch)
        model.add_callback("on_train_batch_end", end_batch)
    try:
        model.train(data=report["data_yaml"], project=str(run.parent), name=run.name, exist_ok=True,
            resume=args.resume, imgsz=args.imgsz, batch=args.batch, epochs=args.epochs,
            device=args.device, workers=args.workers, seed=args.seed, deterministic=False,
            channels_last=args.channels_last,
            optimizer="AdamW", lr0=args.lr, lrf=0.05, weight_decay=0.0005,
            cos_lr=True, warmup_epochs=3.0, patience=0,
            amp=False if args.amp == "off" else args.amp, nbs=args.nominal_batch or args.batch,
            mask_ratio=args.mask_ratio, overlap_mask=False, single_cls=True,
            degrees=180.0, translate=0.05, scale=0.2, shear=0.0, perspective=0.0,
            flipud=0.5, fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, cutmix=0.0, erasing=0.0,
            save=not args.benchmark_steps, save_period=args.save_period,
            val=args.fold >= 0 and not args.benchmark_steps, plots=False, cache=False,
            max_det=100, iou=0.5, verbose=True)
    except Exception as exc:
        if args.benchmark_steps and isinstance(exc, BenchmarkComplete):
            print((run / "benchmark.json").read_text(), flush=True)
            return
        raise


if __name__ == "__main__":
    main()
