"""Convert segmentation probabilities to distinct filament instances."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class PostprocessConfig:
    threshold: float = 0.45
    min_area: int = 120
    min_confidence: float = 0.0
    close_radius: int = 0
    grouping_threshold: float = 0.0
    max_hole_area: int = 0

    def to_dict(self):
        return asdict(self)


def fill_small_holes(binary, max_area):
    if max_area <= 0:
        return binary
    holes = ndi.binary_fill_holes(binary) & ~binary
    labels, _ = ndi.label(holes)
    areas = np.bincount(labels.ravel())
    allowed = (areas <= max_area)
    allowed[0] = False
    return binary | allowed[labels]


@dataclass
class Components:
    """Unfiltered component IDs and statistics, including background at ID zero."""
    labels: np.ndarray
    areas: np.ndarray
    confidence_sums: np.ndarray

    def selection(self, config):
        keep = ((self.areas > 0) & (self.areas >= config.min_area)
                & (self.confidence_sums >= config.min_confidence * self.areas))
        keep[0] = False
        return keep

    def filtered_labels(self, config):
        keep = self.selection(config)
        mapping = np.zeros(len(keep), dtype=np.int32)
        mapping[keep] = np.arange(1, keep.sum() + 1)
        return mapping[self.labels]


def component_key(config):
    """Parameters that change component geometry, excluding selection cutoffs."""
    return (config.threshold, config.close_radius, config.grouping_threshold,
            config.max_hole_area)


def probabilities_to_components(probabilities, config=PostprocessConfig()):
    """Keep actual foreground pixels; optional grouping only affects identity.

    Grouping-mask closing connects gaps without adding artificial bridge pixels
    to the returned foreground segmentation. All outputs are disjoint.
    """
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 2:
        foreground, grouping = probabilities, None
    elif probabilities.shape[0] <= 4:
        foreground = probabilities[0]
        grouping = probabilities[1] if len(probabilities) > 1 else None
    else:
        raise ValueError(f"Expected CxHxW probabilities, got {probabilities.shape}")
    binary = fill_small_holes(foreground >= config.threshold, config.max_hole_area)
    groups = binary.astype(np.uint8)
    if grouping is not None and config.grouping_threshold > 0:
        groups |= (grouping >= config.grouping_threshold).astype(np.uint8)
    if config.close_radius > 0:
        radius = config.close_radius
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        groups = cv2.morphologyEx(groups, cv2.MORPH_CLOSE, kernel)
    count, labels = cv2.connectedComponents(groups, connectivity=8)
    labels[~binary] = 0
    areas = np.bincount(labels.ravel(), minlength=count)
    confidence_sums = np.bincount(labels.ravel(), weights=foreground.ravel(), minlength=count)
    return Components(labels, areas, confidence_sums)


def probabilities_to_labels(probabilities, config=PostprocessConfig()):
    """Convert probabilities to contiguous, nonempty instance labels."""
    return probabilities_to_components(probabilities, config).filtered_labels(config)


def labels_to_rles(labels):
    from .metrics import encode_mask
    # Crop bounding boxes before encoding to avoid repeated full-frame equality
    # scans. Each RLE still covers the original image canvas as required.
    rles = []
    for instance_id, bounds in enumerate(ndi.find_objects(labels), start=1):
        if bounds is None:
            continue
        binary = np.zeros(labels.shape, dtype=np.uint8, order="F")
        binary[bounds] = labels[bounds] == instance_id
        rles.append(encode_mask(binary))
    return rles
