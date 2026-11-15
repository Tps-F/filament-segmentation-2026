"""Exact inference helper extraction; provenance in source_provenance.json."""
from __future__ import annotations
import tempfile
from pycocotools import mask as mask_utils
from .metrics import read_submission


def validate_predictions(predictions, expected):
    if not predictions:
        raise ValueError("No predicted filaments")
    if set(predictions) - set(expected):
        raise ValueError("Predictions contain unknown test images")
    overlaps = {}
    for stem, masks in predictions.items():
        if not masks or any(list(mask["size"]) != [2048, 2048] for mask in masks):
            raise ValueError(f"Invalid native mask shape/list for {stem}")
        areas = [int(mask_utils.area(mask)) for mask in masks]
        if any(area <= 0 or area >= 2048 * 2048 for area in areas):
            raise ValueError(f"Invalid native mask area for {stem}")
        excess = sum(areas) - int(mask_utils.area(mask_utils.merge(masks)))
        if excess:
            overlaps[stem] = excess
    if overlaps:
        raise ValueError(f"Kaggle forbids every overlapping pixel; excess coverage: {overlaps}")
    return {"pixel_disjoint": True, "overlapping_images": 0,
            "excess_overlap_pixels": 0, "images_checked": len(expected)}


def validate_payload(payload, expected):
    """Parse the exact immutable byte snapshot that will be archived/uploaded."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as stream:
        stream.write(payload)
        stream.flush()
        predictions = read_submission(stream.name)
        validation = validate_predictions(predictions, expected)
    return predictions, validation

