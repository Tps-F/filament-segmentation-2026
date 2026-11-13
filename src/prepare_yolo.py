"""Loss-audited official COCO -> annotator-aware YOLO segmentation dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from pycocotools import mask as mask_utils


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def split_identity(manifest, fold):
    if fold != -1 and fold not in range(manifest["n_folds"]):
        raise ValueError(f"Invalid fold {fold}")
    train = sorted(r["image_id"] for r in manifest["train"] if r["fold"] != fold)
    val = sorted(r["image_id"] for r in manifest["train"] if r["fold"] == fold)
    record = {
        "annotation_sha256": manifest["annotation_sha256"],
        "manifest_version": manifest["version"],
        "fold_grouping": manifest["fold_grouping"],
        "validation_fold": fold,
        "training_images": train,
        "validation_images": val,
    }
    record["data_signature"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record


def polygon_label(annotation, width, height):
    """Keep one instance per line, including disjoint components via thin bridges.

    The supplied official dataset has one polygon in every annotation. The
    uncommon multi-polygon path preserves components using Ultralytics' bridge
    convention; prepare_dataset audits its raster change against exact COCO.
    """
    segmentation = annotation["segmentation"]
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError(f"Expected nonempty COCO polygons: {annotation['id']}")
    polygons = []
    for coordinates in segmentation:
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid polygon: {annotation['id']}")
        polygons.append(points)
    if len(polygons) > 1:
        from ultralytics.data.converter import merge_multi_segment
        points = np.concatenate(merge_multi_segment([p.ravel().tolist() for p in polygons]))
    else:
        points = polygons[0]
    normalized = np.clip(points / [width, height], 0, 1)
    # 17 significant digits round-trip float64. Since native dimensions are
    # powers of two, normalization then recovery preserves source coordinates.
    line = "0 " + " ".join(f"{value:.17g}" for value in normalized.ravel())
    # Audit the serialized values, rather than the unrounded source coordinates.
    recovered = np.asarray([float(v) for v in line.split()[1:]]).reshape(-1, 2) * [width, height]
    original = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    converted = mask_utils.merge(mask_utils.frPyObjects([recovered.ravel().tolist()], height, width))
    inter = float(mask_utils.area(mask_utils.merge([original, converted], intersect=True)))
    union = float(mask_utils.area(mask_utils.merge([original, converted])))
    before, after = int(mask_utils.area(original)), int(mask_utils.area(converted))
    if after == 0:
        raise ValueError(f"Conversion erased instance {annotation['id']}")
    return line, {"annotation_id": annotation["id"], "components": len(polygons),
                  "source_area": before, "converted_area": after,
                  "lost_pixels": int(before - inter), "added_pixels": int(after - inter),
                  "iou": inter / union if union else 1.0}


def prepare_dataset(manifest_path="data/cache/manifest.json", output=None, fold=0):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    identity = split_identity(manifest, fold)
    output = Path(output or f"data/yolo_fold{fold}").resolve()
    annotation_path = Path(manifest["annotation_path"])
    if digest(annotation_path) != manifest["annotation_sha256"]:
        raise ValueError("Official annotations have changed since the cache was prepared")
    metadata = {"schema": 1, "manifest_sha256": digest(manifest_path), **identity,
                "conversion": "class0; per-annotator symlink copies; float64 round-trip polygons; exact COCO audit"}
    existing = output / "dataset_report.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        if any(old.get(key) != value for key, value in metadata.items()):
            raise ValueError("Existing YOLO dataset has different provenance; choose a new --output")
    coco = json.loads(annotation_path.read_text())
    images = {str(i["id"]): i for i in coco["images"]}
    annotations = {str(a["id"]): a for a in coco["annotations"]}
    train_groups = {r["date_group"] for r in manifest["train"] if r["fold"] != fold}
    val_groups = {r["date_group"] for r in manifest["train"] if r["fold"] == fold}
    if train_groups & val_groups:
        raise ValueError("Date groups overlap between training and validation")
    source_images = Path(manifest["data_root"]) / "train" / "train_images"
    copies, counts, audit, label_hash = [], Counter(), [], hashlib.sha256()
    wanted = {(split, kind): set() for split in ("train", "val") for kind in ("images", "labels")}
    for record in sorted(manifest["train"], key=lambda r: r["image_id"]):
        split = "val" if record["fold"] == fold else "train"
        source = (source_images / record["file_name"]).resolve(strict=True)
        for annotator in record["annotators"]:
            image_id = str(annotator["image_id"])
            if Path(image_id).name != image_id:
                raise ValueError(f"Unsafe image ID {image_id!r}")
            original_image = images[image_id]
            if Path(original_image["file_name"]).stem != record["image_id"]:
                raise ValueError("Annotator/image correspondence differs from official COCO")
            lines = []
            for annotation_id in annotator["annotation_ids"]:
                annotation = annotations[str(annotation_id)]
                if str(annotation["image_id"]) != image_id:
                    raise ValueError("Annotation belongs to a different annotator/image")
                line, quality = polygon_label(annotation, original_image["width"], original_image["height"])
                lines.append(line)
                audit.append(quality)
            image_path = output / "images" / split / (image_id + source.suffix)
            label_path = output / "labels" / split / (image_id + ".txt")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if image_path.is_symlink():
                if image_path.resolve() != source:
                    raise ValueError(f"Existing symlink has another source: {image_path}")
            elif image_path.exists():
                raise ValueError(f"Refusing to replace non-symlink image {image_path}")
            else:
                image_path.symlink_to(source)
            label_text = "\n".join(lines) + ("\n" if lines else "")
            label_path.write_text(label_text)
            label_hash.update((split + "/" + image_id + "\n" + label_text).encode())
            wanted[split, "images"].add(image_path.name)
            wanted[split, "labels"].add(label_path.name)
            counts[split + "_annotator_records"] += 1
            counts[split + "_instances"] += len(lines)
            copies.append({"copy_id": image_id, "image_id": record["image_id"],
                           "fold": record["fold"], "date_group": record["date_group"], "split": split})
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            directory = output / kind / split
            actual = {str(p.relative_to(directory)) for p in directory.rglob("*")
                      if p.is_file() or p.is_symlink()}
            if actual != wanted[split, kind]:
                raise ValueError(f"Unexpected files in {kind}/{split}; use a clean output directory")
    # JSON is a valid YAML subset, retaining predictable absolute dataset paths.
    (output / "data.yaml").write_text(json.dumps({"path": str(output), "train": "images/train",
        "val": "images/val" if fold >= 0 else "images/train", "names": {0: "filament"}}, indent=2))
    changed = [a for a in audit if a["lost_pixels"] or a["added_pixels"]]
    report = {**metadata, **dict(counts), "train_unique_images": len(identity["training_images"]),
              "val_unique_images": len(identity["validation_images"]),
              "train_date_groups": len(train_groups), "val_date_groups": len(val_groups),
              "labels_sha256": label_hash.hexdigest(), "instances": len(audit),
              "multi_polygon_instances": sum(a["components"] > 1 for a in audit),
              "minimum_conversion_iou": min(a["iou"] for a in audit),
              "lost_pixels": sum(a["lost_pixels"] for a in audit),
              "added_pixels": sum(a["added_pixels"] for a in audit),
              "changed_instances": changed, "records": copies,
              "data_yaml": str(output / "data.yaml")}
    atomic_json(existing, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = prepare_dataset(args.manifest, args.output, args.fold)
    print(json.dumps({k: v for k, v in report.items() if k not in
                     {"records", "training_images", "validation_images", "changed_instances"}}, indent=2))


if __name__ == "__main__":
    main()
