"""Exact organizer PQ semantics using COCO RLE intersection calculations.

The organizer evaluates each annotator/image pair separately, counts every
IoU > .5 pair, and aggregates TP IoUs, FP and FN globally. This deliberately
does not replace the published matching rule with Hungarian matching.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

DEFAULT_ANNOTATIONS = Path("data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")


@dataclass
class PQStats:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __iadd__(self, other):
        for key in ("iou_sum", "tp", "fp", "fn"):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        return self

    @property
    def pq(self):
        den = self.tp + 0.5 * (self.fp + self.fn)
        return self.iou_sum / den if den else 0.0

    def to_dict(self):
        return {**asdict(self), "pq": self.pq,
                "sq": self.iou_sum / max(1, self.tp),
                "rq": self.tp / max(1, self.tp + 0.5 * (self.fp + self.fn))}


def encode_mask(mask):
    return mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))


def json_rle(rle):
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]}


def annotation_rle(annotation, height=2048, width=2048):
    seg = annotation["segmentation"]
    if isinstance(seg, list):
        return mask_utils.merge(mask_utils.frPyObjects(seg, height, width))
    if isinstance(seg["counts"], list):
        return mask_utils.frPyObjects(seg, height, width)
    return seg


def load_ground_truth(path=DEFAULT_ANNOTATIONS):
    """Return {image_stem: [(annotator_image_id, [rles...]), ...]}."""
    with open(path) as f:
        data = json.load(f)
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)
    grouped = defaultdict(list)
    for image in data["images"]:
        rles = [annotation_rle(a, image["height"], image["width"])
                for a in annotations[image["id"]]]
        grouped[Path(image["file_name"]).stem].append((image["id"], rles))
    return dict(grouped)


def score_rles(gt_rles, pred_rles):
    if not gt_rles:
        return PQStats(fp=len(pred_rles))
    if not pred_rles:
        return PQStats(fn=len(gt_rles))
    # mask_utils.iou returns (detections, ground truths).
    iou = np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles)), dtype=np.float64).T
    hit = iou > 0.5
    return PQStats(float(iou[hit].sum()), int(hit.sum()),
                   int((~hit.any(axis=0)).sum()), int((~hit.any(axis=1)).sum()))


def score_image(ground_truth_records, pred_rles):
    total = PQStats()
    for _, gt in ground_truth_records:
        total += score_rles(gt, pred_rles)
    return total


def read_submission(path):
    predictions = defaultdict(list)
    seen = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        for row in reader:
            fid = row["filament_id"]
            if fid in seen:
                raise ValueError(f"Duplicate filament_id: {fid}")
            seen.add(fid)
            stem, suffix = fid.rsplit("_", 1)
            if not suffix:
                raise ValueError(f"Empty instance suffix: {fid}")
            rle = {"size": [2048, 2048], "counts": row["segmentation_rle"].encode("ascii")}
            area = int(mask_utils.area(rle))
            if not 0 < area < 2048 * 2048:
                raise ValueError(f"Invalid area {area} for {fid}")
            predictions[stem].append(rle)
    return dict(predictions)


def evaluate_submission(path, ground_truth=None, image_ids=None):
    ground_truth = ground_truth or load_ground_truth()
    predictions = read_submission(path)
    ids = sorted(image_ids if image_ids is not None else ground_truth)
    unknown = set(predictions) - set(ground_truth)
    if unknown:
        raise ValueError(f"Unknown prediction images: {sorted(unknown)[:10]}")
    total, per_image = PQStats(), {}
    for stem in ids:
        stat = score_image(ground_truth[stem], predictions.get(stem, []))
        total += stat
        per_image[stem] = stat.to_dict()
    return {"total": total.to_dict(), "images": per_image}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--output")
    parser.add_argument("--fold", type=int, help="Evaluate only this held-out manifest fold")
    parser.add_argument("--manifest", default="data/cache/manifest.json")
    args = parser.parse_args()
    image_ids = None
    if args.fold is not None:
        manifest = json.loads(Path(args.manifest).read_text())
        image_ids = [record["image_id"] for record in manifest["train"] if record["fold"] == args.fold]
        if not image_ids:
            raise ValueError(f"No images selected for fold {args.fold}")
    report = evaluate_submission(args.submission, load_ground_truth(args.annotations), image_ids=image_ids)
    print(json.dumps(report["total"], indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
