"""Independent implementation of HDJoJo's fixed dual-YOLO mask export algorithm.

Method attribution: https://www.kaggle.com/code/hdjojo/solar-filament-segmentation
Public notebook version 2; source SHA256
103b33f5f331b81c9e4cad1d3acbf09a137947d27f44ee2785f446106b47ce04.
This module does not import a model or load predictions, images, or weights.
"""
from __future__ import annotations
import cv2
import numpy as np

NATIVE_SIZE = 2048


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _input_contract(masks, scores):
    _require(isinstance(masks, np.ndarray) and masks.ndim == 3
             and masks.dtype == np.uint8 and min(masks.shape[1:]) > 0,
             'Literal source requires N,H,W uint8 masks')
    _require(isinstance(scores, np.ndarray) and scores.ndim == 1
             and scores.dtype == np.float32 and len(scores) == len(masks),
             'Literal source requires paired float32 confidences')
    _require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(),
             'Invalid confidence; no numerical rescue')
    _require(masks.max(initial=0) <= 1, 'Masks must already be binary; no threshold rescue')


def _intersection_over_union(first, second):
    denominator = np.logical_or(first, second).sum()
    if denominator == 0:
        return 0.0
    return np.logical_and(first, second).sum() / denominator


def _nms_indices(masks, scores):
    pending = np.argsort(scores)[::-1].tolist()
    retained = []
    while pending:
        head, *tail = pending
        retained.append(head)
        pending = [index for index in tail
                   if _intersection_over_union(masks[head], masks[index]) <= 0.5]
    return retained


def export_instances(masks26, scores26, masks8, scores8):
    """Return (instances, JSON-safe trace) with source ordering and gapped ranks.

    Each absent model contributes zero rows independently. The both-present
    algorithm is unchanged from the published source. Production callers must separately require the
    actual model masks to be native 2048 square. Smaller generated masks are useful
    for CPU parity checks of the fixed nearest-neighbor export operation.
    """
    trace = {'whole_image_skipped': masks26 is None and masks8 is None,
             'nms_source_indices': [], 'ownership_source_indices': [],
             'kept_source_indices': [], 'dropped_after_resize_minarea': []}
    if trace['whole_image_skipped']:
        return [], trace
    if masks26 is None:
        _input_contract(masks8, scores8)
        masks26 = np.empty((0, *masks8.shape[1:]), dtype=np.uint8)
        scores26 = np.empty(0, dtype=np.float32)
    if masks8 is None:
        _input_contract(masks26, scores26)
        masks8 = np.empty((0, *masks26.shape[1:]), dtype=np.uint8)
        scores8 = np.empty(0, dtype=np.float32)
    _input_contract(masks26, scores26)
    _input_contract(masks8, scores8)
    _require(masks26.shape[1:] == masks8.shape[1:], 'Model mask shapes differ')
    masks = np.concatenate((masks26, masks8), axis=0)
    scores = np.concatenate((scores26, scores8), axis=0)
    retained = _nms_indices(masks, scores)
    trace['nms_source_indices'] = retained
    # The second default NumPy sort is intentional, including its tie order.
    order = [retained[index] for index in np.argsort(scores[retained])[::-1]]
    trace['ownership_source_indices'] = order
    occupied = np.zeros(masks.shape[1:], dtype=np.uint8)
    instances = []
    for rank, index in enumerate(order, start=1):
        clean = masks[index] & ~occupied
        occupied |= masks[index]
        native = cv2.resize(clean, (NATIVE_SIZE, NATIVE_SIZE), interpolation=cv2.INTER_NEAREST)
        if native.sum() < 5:
            trace['dropped_after_resize_minarea'].append({'rank': rank, 'source_index': index})
            continue
        instances.append({'rank': rank, 'mask': native, 'source_index': index,
                          'score': float(scores[index])})
        trace['kept_source_indices'].append(index)
    return instances, trace
