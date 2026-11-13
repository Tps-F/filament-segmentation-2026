"""Fixed confidence-ordered pixel ownership for already-selected instances.

Stateless CPU-only helper. Mirrors historical src.infer_yolo.filter_detections
with confidence=0, disjoint=True and mask_nms_iou=None, retaining source IDs.
No quality score, annotation, new morphology, NMS or model is consulted.
"""
from __future__ import annotations
import numpy as np
from pycocotools import mask as mu


def require(ok, message):
    if not ok:
        raise ValueError(message)


def json_rle(rle):
    return {'size': list(rle['size']), 'counts': rle['counts'].decode('ascii')
            if isinstance(rle['counts'], bytes) else rle['counts']}


def confidence_disjoint_with_ids(selected, min_area=5):
    """Mirror filter_detections exactly, retaining source identity and audit data.

    selected contains only rle, score and source_index; no quality or GT values
    affect ownership. Stable sorting intentionally preserves selected-order ties.
    """
    require(len({d['source_index'] for d in selected}) == len(selected), 'Duplicate selected source identity')
    occupied = None
    kept, audit = [], []
    for d in sorted(selected, key=lambda d: -d['score']):
        require(np.isfinite(d['score']) and 0 <= d['score'] <= 1, 'Invalid detector confidence')
        original = mu.decode(d['rle']).astype(bool)
        if occupied is None:
            occupied = np.zeros_like(original)
        require(original.shape == occupied.shape, 'Inconsistent mask shapes')
        remaining = original & ~occupied
        area = int(remaining.sum())
        item = {'source_index': int(d['source_index']), 'confidence': float(d['score']),
                'original_area': int(original.sum()), 'remaining_area': area,
                'removed_overlap_pixels': int(original.sum()) - area, 'retained': area >= min_area}
        audit.append(item)
        if area < min_area:
            continue
        occupied |= remaining
        kept.append({'rle': json_rle(mu.encode(np.asfortranarray(remaining, dtype=np.uint8))),
                     'score': float(d['score']), 'source_index': int(d['source_index'])})
    return kept, audit
