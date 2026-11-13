"""Isolated, image-only contrast representations for a controlled experiment.

No current trainer or inference path imports this module. Apply transforms to
the full native image before spatial crops, resizing, or tiled inference. CLAHE
computed independently on training crops and inference tiles would differ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ContrastConfig:
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    background_sigma: float = 32.0
    background_downsample: int = 4
    residual_gain: float = 4.0
    blend_weight: float = .5

    def __post_init__(self):
        values = [self.clahe_clip, self.background_sigma, self.residual_gain]
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Contrast scales must be finite and positive")
        if not isinstance(self.clahe_grid, int) or not 1 <= self.clahe_grid <= 2048:
            raise ValueError("CLAHE grid must be a positive native-image tile count")
        if not isinstance(self.background_downsample, int) or not 1 <= self.background_downsample <= 2048:
            raise ValueError("Background downsampling must be a positive integer")
        if not np.isfinite(self.blend_weight) or not 0 <= self.blend_weight <= 1:
            raise ValueError("Blend weight must be in [0,1]")

    def to_dict(self):
        return asdict(self)


def contrast_channels(image: np.ndarray, config: ContrastConfig = ContrastConfig()):
    """Return float32 CHW [raw, CLAHE, signed local residual], each in [0,1].

    The raw channel preserves all input information. No disk mask, segmentation
    labels, data-set statistics, or trained parameters are used. The local
    residual remains dark for dark structures; it is centered at .5.
    """
    if image.shape != (2048, 2048) or image.dtype != np.uint8:
        raise ValueError("Expected a complete 2048x2048 uint8 image before cropping")
    raw = image.astype(np.float32) / 255
    clahe = cv2.createCLAHE(config.clahe_clip, (config.clahe_grid, config.clahe_grid))
    enhanced = clahe.apply(np.ascontiguousarray(image)).astype(np.float32) / 255
    scale = config.background_downsample
    small = cv2.resize(raw, (max(1, image.shape[1] // scale), max(1, image.shape[0] // scale)),
                       interpolation=cv2.INTER_AREA)
    background = cv2.GaussianBlur(small, (0, 0), config.background_sigma / scale,
                                  borderType=cv2.BORDER_REFLECT_101)
    background = cv2.resize(background, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    residual = np.clip(.5 + config.residual_gain * (raw - background), 0, 1)
    return np.ascontiguousarray(np.stack([raw, enhanced, residual]), dtype=np.float32)


def contrast_gray(image: np.ndarray, config: ContrastConfig = ContrastConfig()):
    """Lower-risk uint8 raw/CLAHE blend, compatible with existing gray caches."""
    if image.shape != (2048, 2048) or image.dtype != np.uint8:
        raise ValueError("Expected a complete 2048x2048 uint8 image before cropping")
    clahe = cv2.createCLAHE(config.clahe_clip, (config.clahe_grid, config.clahe_grid))
    enhanced = clahe.apply(np.ascontiguousarray(image))
    return np.rint((1 - config.blend_weight) * image.astype(np.float32) +
                   config.blend_weight * enhanced.astype(np.float32)).astype(np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    image = np.load(args.image) if args.image.suffix == ".npy" else cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot decode image")
    if args.output.exists():
        raise FileExistsError("Use a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = ContrastConfig()
    np.savez_compressed(args.output, channels=contrast_channels(image, config),
                        gray_blend=contrast_gray(image, config))
    args.output.with_suffix(".json").write_text(json.dumps({"input": str(args.image),
        "config": config.to_dict(), "scope": "Image-only experiment; no model validation score."}, indent=2))
