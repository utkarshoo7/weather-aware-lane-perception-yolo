import numpy as np


def compute_lane_metrics(mask: np.ndarray):
    """
    Compute simple, rule-based lane visibility metrics.

    NOTE:
    - UNet is dropped
    - mask can be empty or zeros
    - function must ALWAYS exist for imports
    """

    # Ensure binary mask
    mask = (mask > 0).astype(np.uint8)

    h, w = mask.shape
    total_pixels = h * w

    lane_pixels = mask.sum()
    lane_pixel_ratio = lane_pixels / total_pixels if total_pixels > 0 else 0.0

    bottom_half = mask[h // 2 :, :]
    bottom_half_density = (
        bottom_half.sum() / bottom_half.size if bottom_half.size > 0 else 0.0
    )

    vertical_hits = (mask.sum(axis=0) > 0).sum()
    vertical_continuity = vertical_hits / w if w > 0 else 0.0

    # Rule-based visibility
    if lane_pixel_ratio > 0.01 and vertical_continuity > 0.30:
        final_visibility = "high"
    elif lane_pixel_ratio > 0.003:
        final_visibility = "medium"
    else:
        final_visibility = "low"

    return {
        "lane_pixel_ratio": float(lane_pixel_ratio),
        "bottom_half_density": float(bottom_half_density),
        "vertical_continuity": float(vertical_continuity),
        "final_visibility": final_visibility,
    }


def compute_lane_reliability(metrics: dict):
    """
    HARD GATE — no ML, no tuning
    """

    if (
        metrics["lane_pixel_ratio"] > 0.01
        and metrics["bottom_half_density"] > 0.05
        and metrics["vertical_continuity"] > 0.30
    ):
        return "reliable"

    return "unreliable"
