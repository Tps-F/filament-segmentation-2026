"""Native 2048-mask YOLO inference, immutable caches, and fold verification.

Predictions depend only on JPEG pixels, checkpoint weights, and CLI settings.
Manifest membership is used to select images and verify the held-out fold;
annotation polygons/masks are never passed to prediction or postprocessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pycocotools import mask as mask_utils

from .inference import write_submission
from .prepare_yolo import atomic_json, digest
from .train_yolo import checkpoint_provenance, validate_provenance


def filter_detections(detections, confidence, min_area=5, disjoint=True, mask_nms_iou=None):
    """Filter by confidence/area, optionally suppress masks or clip overlaps.

    Mask NMS compares the original masks in confidence order. The CLI uses
    ``disjoint=False`` with mask NMS so retained masks keep every native pixel.
    With no mask-NMS threshold, the historical overlap handling is unchanged.
    """
    if mask_nms_iou is not None and not 0 < mask_nms_iou < 1:
        raise ValueError("mask_nms_iou must be strictly between 0 and 1")
    result = []
    nms_masks = []
    occupied = None
    for detection in sorted(detections, key=lambda d: -d["score"]):
        if detection["score"] < confidence:
            continue
        rle = detection["rle"]
        original_rle = rle
        if mask_nms_iou is not None:
            if int(mask_utils.area(rle)) < min_area:
                continue
            if nms_masks and np.any(mask_utils.iou([rle], nms_masks, [0] * len(nms_masks)) > mask_nms_iou):
                continue
        if disjoint:
            mask = mask_utils.decode(rle).astype(bool)
            if occupied is None:
                occupied = np.zeros_like(mask)
            mask &= ~occupied
            if int(mask.sum()) < min_area:
                continue
            occupied |= mask
            rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
        elif int(mask_utils.area(rle)) < min_area:
            continue
        result.append(rle)
        if mask_nms_iou is not None:
            nms_masks.append(original_rle)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True, help="Directory for raw per-image RLEs and threshold CSVs")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--raw-confidence", type=float, default=0.03)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.35,0.50")
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--mask-nms-iou", type=float, default=None,
        help="Suppress lower-confidence masks above this mask IoU and preserve all pixels of retained masks; replaces overlap clipping")
    parser.add_argument("--device", default="0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.mask_nms_iou is not None and not 0 < args.mask_nms_iou < 1:
        parser.error("--mask-nms-iou must be strictly between 0 and 1")
    disjoint = not (args.allow_overlap or args.mask_nms_iou is not None)
    postprocessing_policy = ("mask_nms_preserve_pixels" if args.mask_nms_iou is not None else
                            "preserve_pixels" if args.allow_overlap else "confidence_ordered_disjoint_clipping")
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    thresholds = sorted(set(float(s) for s in args.thresholds.split(",")))
    if not thresholds or min(thresholds) < args.raw_confidence or max(thresholds) > 1:
        raise ValueError("Output thresholds must be between raw confidence and 1")
    if args.split == "val" and args.fold < 0:
        raise ValueError("Validation requires a held-out fold")
    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest["test" if args.split == "test" else "train"]
    if args.split == "val":
        entries = [r for r in entries if r["fold"] == args.fold]
    elif args.split == "train":
        entries = [r for r in entries if r["fold"] != args.fold]
    entries = sorted(entries, key=lambda r: r["image_id"])
    if args.limit:
        entries = entries[:args.limit]
    checkpoint = Path(args.checkpoint)
    checkpoint_hash = digest(checkpoint)
    model = YOLO(str(checkpoint))
    if digest(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading; use a completed epoch snapshot")
    provenance = checkpoint_provenance(model)
    validation = validate_provenance(provenance, manifest, args.fold) if args.split == "val" else None
    if provenance.get("benchmark"):
        raise ValueError("Benchmark checkpoint is not a trained experiment")
    if model.task != "segment":
        raise ValueError("Expected segmentation checkpoint")
    if model.model.end2end:
        raise ValueError("Expected one-to-many checkpoint for documented NMS IoU .5")
    output = Path(args.output)
    cache = output / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    identity = {"schema": 1, "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": digest(args.manifest), "split": args.split, "fold": args.fold,
        "imgsz": args.imgsz, "nms_iou": args.nms_iou, "raw_confidence": args.raw_confidence,
        "retina_masks": True, "max_det": args.max_det, "end2end": False,
        "ultralytics": ultralytics.__version__, "validation": validation}
    identity_path = output / "inference_config.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Existing raw cache has another configuration; use a new output directory")
    if not identity_path.exists() and any(cache.glob("*.json")):
        raise ValueError("Raw predictions exist without a provenance record")
    atomic_json(identity_path, identity)
    image_folder = Path(manifest["data_root"]) / args.split.replace("val", "train")
    image_folder = image_folder / ("test_images" if args.split == "test" else "train_images")
    started = time.perf_counter()
    counts = {}
    for index, entry in enumerate(entries, 1):
        image = image_folder / entry["file_name"]
        saved = cache / (entry["image_id"] + ".json")
        if saved.exists():
            detections = json.loads(saved.read_text())
        else:
            result = model.predict(str(image), imgsz=args.imgsz, conf=args.raw_confidence,
                iou=args.nms_iou, max_det=args.max_det, retina_masks=True,
                device=args.device, verbose=False)[0]
            if tuple(result.orig_shape) != (2048, 2048):
                raise ValueError(f"Unexpected native image shape {result.orig_shape}")
            detections = []
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                for mask, score, box in zip(masks, scores, boxes):
                    if mask.shape != (2048, 2048):
                        raise ValueError("retina_masks failed to return the exact native resolution")
                    binary = np.asfortranarray(mask > 0.5, dtype=np.uint8)
                    if not binary.any():
                        continue
                    rle = mask_utils.encode(binary)
                    detections.append({"score": float(score), "box": box.tolist(),
                        "area": int(mask_utils.area(rle)),
                        "rle": {"size": [2048, 2048], "counts": rle["counts"].decode("ascii")}})
            detections.sort(key=lambda d: -d["score"])
            atomic_json(saved, detections)
        counts[entry["image_id"]] = len(detections)
        print(json.dumps({"done": index, "total": len(entries), "image": entry["image_id"],
            "instances": len(detections), "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    artifacts = {}
    ids = [r["image_id"] for r in entries]
    for threshold in thresholds:
        def predictions():
            for image_id in ids:
                detections = json.loads((cache / (image_id + ".json")).read_text())
                yield image_id, filter_detections(detections, threshold, args.min_area,
                                                 disjoint, args.mask_nms_iou)
        csv_path = output / f"submission_conf{threshold:.3f}.csv"
        info = write_submission(predictions(), csv_path, ids)
        artifacts[str(threshold)] = {**info, "sha256": digest(csv_path)}
    report = {**identity, "checkpoint": str(checkpoint.resolve()), "source_provenance": provenance,
        "images": len(entries), "image_ids": ids, "raw_instances": sum(counts.values()),
        "seconds": time.perf_counter() - started, "min_area": args.min_area,
        "disjoint": disjoint, "mask_nms_iou": args.mask_nms_iou,
        "postprocessing_policy": postprocessing_policy, "outputs": artifacts,
        "validation_note": "Held-out official image/date groups" if validation else "Not a held-out validation result"}
    atomic_json(output / "run_report.json", report)
    print(json.dumps({"images": len(entries), "raw_instances": sum(counts.values()), "outputs": artifacts}, indent=2))


if __name__ == "__main__":
    main()
