"""
Lane confidence estimation utilities.

This module computes a simple, deterministic visibility proxy
from a binary lane segmentation mask.

The metric is intentionally lightweight and interpretable,
designed for downstream fusion and decision logic rather than
probabilistic calibration.
"""

import numpy as np


def lane_confidence(mask: np.ndarray) -> float:
    """
    Compute lane visibility confidence from a binary segmentation mask.

    Definition:
        lane_confidence = (# lane pixels) / (total image pixels)

    Assumptions:
    - mask is a 2D binary array (0 = background, 1 = lane)
    - higher values indicate clearer, more visible lanes

    Args:
        mask (np.ndarray): Binary lane mask of shape (H, W)

    Returns:
        float: Lane visibility confidence in [0.0, 1.0]
    """

    if mask is None or mask.size == 0:
        return 0.0

    # Enforce binary interpretation
    mask_bin = (mask > 0).astype(np.uint8)

    total_pixels = mask_bin.size
    lane_pixels = int(mask_bin.sum())

    confidence = lane_pixels / total_pixels

    return round(float(confidence), 4)
