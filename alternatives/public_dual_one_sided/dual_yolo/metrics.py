"""Exact inference helper extraction; provenance in source_provenance.json."""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils


def json_rle(rle):
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]}


def encode_mask(mask):
    return mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))


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

