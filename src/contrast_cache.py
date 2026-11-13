"""Derived grayscale training caches and explicit inference transform identity."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .contrast_inputs import ContrastConfig, contrast_gray


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=8)
def golden_hash(clip=2., grid=8, blend=.5):
    yy, xx = np.indices((2048, 2048), dtype=np.int32)
    image = ((xx // 8 + yy // 16 + (xx * 13 + yy * 7) % 17) % 256).astype(np.uint8)
    out = contrast_gray(image, ContrastConfig(clahe_clip=clip, clahe_grid=grid, blend_weight=blend))
    return hashlib.sha256(out.tobytes()).hexdigest()


def preprocessing_spec(mode):
    if mode == "raw":
        return {"name": "raw", "version": 1}
    if mode != "gray_clahe_blend":
        raise ValueError(f"Unknown input mode {mode}")
    return {"name": mode, "version": 1, "clip_limit": 2., "grid": 8,
            "blend_weight": .5, "golden_sha256": golden_hash()}


def validate_spec(spec):
    if spec is None:
        return preprocessing_spec("raw")
    if not isinstance(spec, dict) or spec != preprocessing_spec(spec.get("name")):
        raise ValueError("Unsupported preprocessing parameters/version or mismatched OpenCV golden fingerprint")
    return spec


def apply_preprocessing(image, spec):
    spec = validate_spec(spec)
    if spec["name"] == "raw":
        return image
    return contrast_gray(image, ContrastConfig(clahe_clip=spec["clip_limit"],
        clahe_grid=spec["grid"], blend_weight=spec["blend_weight"]))


def prepare_cache(source_manifest, output_dir):
    """Make a new train/validation cache; no test-image cache is produced."""
    source_manifest, output_dir = Path(source_manifest), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Derived cache directory must be new")
    manifest = json.loads(source_manifest.read_text())
    if manifest.get("input_preprocessing", {}).get("name", "raw") != "raw":
        raise ValueError("Source cache must contain original raw grayscale")
    output_dir.mkdir(parents=True)
    (output_dir / "images").mkdir()
    result = deepcopy(manifest)
    result["test"] = []
    spec = preprocessing_spec("gray_clahe_blend")
    result["input_preprocessing"] = spec
    result["cache_scope"] = "training_and_validation_images_only"
    source_root = source_manifest.parent
    annotations = Path(result["annotation_path"])
    result["annotation_path"] = str((source_root / annotations).resolve() if not annotations.is_absolute() else annotations)
    receipts = []
    for entry in result["train"]:
        source = Path(entry["image_path"])
        source = source_root / source if not source.is_absolute() else source
        labels = Path(entry["label_path"])
        labels = source_root / labels if not labels.is_absolute() else labels
        image = np.load(source, allow_pickle=False)
        transformed = apply_preprocessing(image, spec)
        target = output_dir / "images" / (entry["image_id"] + ".npy")
        np.save(target, transformed)
        entry["image_path"] = str(target.relative_to(output_dir))
        entry["label_path"] = str(labels.resolve())
        receipts.append({"image_id": entry["image_id"], "fold": entry["fold"],
                         "source_sha256": digest(source), "derived_sha256": digest(target)})
    receipt = {"preprocessing": spec, "opencv": cv2.__version__, "images": len(receipts),
               "source_manifest_sha256": digest(source_manifest), "image_hashes": receipts,
               "algorithm_source_sha256": digest(Path(__file__).with_name("contrast_inputs.py")),
               "transform_order": "full native image before crop, rescale, augmentation and tiling"}
    result["derived_cache_receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    (output_dir / "manifest.json").write_text(json.dumps(result))
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt
