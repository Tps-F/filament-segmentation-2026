"""Overlap-tile inference, probability caching, and validated CSV export."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .metrics import read_submission
from .postprocess import PostprocessConfig, labels_to_rles, probabilities_to_labels


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    return sorted(set(list(range(0, length - tile + 1, stride)) + [length - tile]))


@torch.inference_mode()
def predict_image(model, image, tile_size=768, stride=512, batch_size=4, tta=1,
                  device="cuda", amp=True, scale=1.0):
    """Return CxHxW probabilities. TTA supports identity, flips, and all D4."""
    if tta not in (1, 2, 4, 8):
        raise ValueError("tta must be 1, 2, 4, or 8")
    preprocessing = getattr(model, "input_preprocessing", None)
    if preprocessing is not None:
        from .contrast_cache import apply_preprocessing
        image = apply_preprocessing(image, preprocessing)
    # The declared transform is applied before scale changes and tile extraction.
    original_shape = image.shape[:2]
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    height, width = image.shape[:2]
    pad_h, pad_w = max(0, tile_size - height), max(0, tile_size - width)
    image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    ph, pw = image.shape[:2]
    positions = [(y, x) for y in tile_positions(ph, tile_size, stride)
                 for x in tile_positions(pw, tile_size, stride)]
    # Raised cosine with nonzero outer weights also covers the image edges.
    window_1d = np.maximum(np.hanning(tile_size), 0.05).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    accumulated = None
    weights = np.zeros((ph, pw), dtype=np.float32)
    transforms = [(False, ())]
    if tta >= 2:
        transforms.append((False, (-1,)))
    if tta >= 4:
        transforms.extend([(False, (-2,)), (False, (-2, -1))])
    if tta == 8:
        transforms += [(True, dims) for _, dims in transforms[:4]]
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start + batch_size]
        batch = np.stack([image[y:y + tile_size, x:x + tile_size].transpose(2, 0, 1)
                          for y, x in batch_positions])
        tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32).div_(255)
        tensor = tensor.contiguous(memory_format=torch.channels_last)
        result = None
        for transpose, dims in transforms:
            augmented = tensor.transpose(-1, -2) if transpose else tensor
            if dims:
                augmented = augmented.flip(dims)
            amp_dtype = torch.bfloat16 if not str(device).startswith("cuda") or torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype,
                                enabled=amp and str(device).startswith("cuda")):
                logits = model(augmented)
                if isinstance(logits, dict):
                    logits = logits["out"]
                if logits.shape[-2:] != augmented.shape[-2:]:
                    logits = F.interpolate(logits.float(), augmented.shape[-2:], mode="bilinear", align_corners=False)
                probability = logits.float().sigmoid()
            if dims:
                probability = probability.flip(dims)
            if transpose:
                probability = probability.transpose(-1, -2)
            result = probability if result is None else result + probability
        result = (result / len(transforms)).cpu().numpy()
        if accumulated is None:
            accumulated = np.zeros((result.shape[1], ph, pw), dtype=np.float32)
        for probability, (y, x) in zip(result, batch_positions):
            accumulated[:, y:y + tile_size, x:x + tile_size] += probability * window
            weights[y:y + tile_size, x:x + tile_size] += window
    result = (accumulated / weights[None])[:, :height, :width]
    if result.shape[-2:] != original_shape:
        result = np.stack([cv2.resize(channel, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
                           for channel in result])
    return result


def write_submission(predictions, output_path, expected_image_ids=None):
    """predictions is an iterable of (image stem, list of COCO RLE dicts)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count, images, empty_images = 0, set(), []
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filament_id", "segmentation_rle"])
        for stem, rles in predictions:
            if stem in images:
                raise ValueError(f"Repeated image {stem}")
            images.add(stem)
            if not rles:
                empty_images.append(stem)
            for i, rle in enumerate(rles, start=1):
                counts = rle["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("ascii")
                writer.writerow([f"{stem}_{i}", counts])
                count += 1
    if expected_image_ids is not None and set(expected_image_ids) != images:
        raise ValueError(f"Image set differs: missing {set(expected_image_ids) - images}, extra {images - set(expected_image_ids)}")
    decoded = read_submission(output_path)
    assert sum(map(len, decoded.values())) == count
    return {"path": str(output_path), "filaments": count, "images_processed": len(images), "empty_images": empty_images}


def validation_data_signature(manifest, fold):
    """Match src.train's recorded annotation and train/validation membership."""
    split_record = {
        "annotation_sha256": manifest.get("annotation_sha256"),
        "manifest_version": manifest.get("version"),
        "fold_grouping": manifest.get("fold_grouping"),
        "validation_fold": fold,
        "training_images": sorted(e["image_id"] for e in manifest["train"] if e["fold"] != fold),
        "validation_images": sorted(e["image_id"] for e in manifest["train"] if e["fold"] == fold),
    }
    return hashlib.sha256(json.dumps(split_record, sort_keys=True).encode()).hexdigest()


def validate_validation_checkpoint(checkpoint_config, manifest, fold):
    if fold < 0 or checkpoint_config.get("fold") != fold:
        raise ValueError("Validation inference requires a checkpoint trained with the same held-out fold")
    expected = validation_data_signature(manifest, fold)
    recorded = checkpoint_config.get("data_signature")
    if recorded is not None and recorded != expected:
        raise ValueError("Checkpoint annotation/train/validation provenance differs from this manifest")
    if recorded is None:
        warnings.warn("Legacy checkpoint has matching held-out fold but no data_signature; "
                      "historical annotation and split membership cannot be verified", stacklevel=2)
    return {"fold": fold, "data_signature": recorded,
            "verification": "data_signature" if recorded is not None else "legacy_fold_only"}


def atomic_save_probabilities(path, probability):
    """Publish a complete NPZ so interrupted inference never leaves a partial cache."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, probability=probability.astype(np.float16))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    parser.add_argument("--split", choices=["val", "train", "all", "test"], default="test")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prob-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tta", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--close-radius", type=int, default=0)
    parser.add_argument("--grouping-threshold", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-hole-area", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from .model import load_checkpoint
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("input_preprocessing", {}).get("name", "raw") != "raw":
        raise ValueError("Inference needs original raw images; checkpoint preprocessing is applied automatically")
    entries = manifest["test" if args.split == "test" else "train"]
    if args.split == "val":
        entries = [e for e in entries if e["fold"] == args.fold]
    elif args.split == "train":
        entries = [e for e in entries if e["fold"] != args.fold]
    entries = sorted(entries, key=lambda e: e["image_id"])
    if args.limit:
        entries = entries[:args.limit]
    models, validation_provenance = [], []
    for checkpoint in args.checkpoint:
        loaded = load_checkpoint(checkpoint, device=args.device, use_ema=True)
        if args.split == "val":
            if not isinstance(loaded, tuple):
                raise ValueError("Validation inference requires checkpoint training metadata")
            validation_provenance.append(validate_validation_checkpoint(loaded[1].get("config", {}), manifest, args.fold))
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        model.eval().to(memory_format=torch.channels_last)
        models.append(model)
    config = PostprocessConfig(threshold=args.threshold, min_area=args.min_area,
                              close_radius=args.close_radius, grouping_threshold=args.grouping_threshold,
                              min_confidence=args.min_confidence, max_hole_area=args.max_hole_area)
    prob_dir = Path(args.prob_dir)
    prob_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = []
    for checkpoint in args.checkpoint:
        with open(checkpoint, "rb") as f:
            fingerprints.append(hashlib.file_digest(f, "sha256").hexdigest())
    run_config = {**vars(args), "checkpoint_fingerprints": fingerprints, "postprocess": config.to_dict(),
                  "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  "validation_provenance": validation_provenance,
                  "input_preprocessing": [getattr(model, "input_preprocessing", None) for model in models]}
    metadata_path = prob_dir / "inference_config.json"
    identity_keys = ["checkpoint", "checkpoint_fingerprints", "tile_size", "stride", "tta", "scale", "split", "fold"]
    if any(value is not None and value.get("name") != "raw" for value in run_config["input_preprocessing"]):
        identity_keys.append("input_preprocessing")
    if metadata_path.exists() and not args.overwrite:
        old = json.loads(metadata_path.read_text())
        if "manifest_sha256" in old:
            identity_keys.append("manifest_sha256")
        if any(old.get(k) != run_config.get(k) for k in identity_keys):
            raise ValueError("Probability cache belongs to a different inference config; use a new directory or --overwrite")
    metadata_path.write_text(json.dumps(run_config, indent=2))
    predictions, started = [], time.time()
    for idx, entry in enumerate(entries, start=1):
        stem = entry["image_id"]
        path = prob_dir / f"{stem}.npz"
        if path.exists() and not args.overwrite:
            with np.load(path) as z:
                probability = z["probability"].astype(np.float32)
        else:
            image_path = Path(entry["image_path"])
            if not image_path.is_absolute():
                image_path = Path(args.manifest).parent / image_path
            image = np.load(image_path)
            probability = None
            for model in models:
                output = predict_image(model, image, args.tile_size, args.stride, args.batch_size,
                                       args.tta, args.device, scale=args.scale)
                probability = output if probability is None else probability + output
            probability /= len(models)
            atomic_save_probabilities(path, probability)
        labels = probabilities_to_labels(probability, config)
        rles = labels_to_rles(labels)
        predictions.append((stem, rles))
        print(json.dumps({"image": idx, "total": len(entries), "id": stem, "filaments": len(rles),
                          "elapsed_s": round(time.time() - started, 1)}), flush=True)
    report = write_submission(predictions, args.output, [e["image_id"] for e in entries])
    report["elapsed_s"] = time.time() - started
    Path(str(args.output) + ".json").write_text(json.dumps({**run_config, **report}, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
