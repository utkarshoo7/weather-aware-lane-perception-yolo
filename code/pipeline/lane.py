"""
Lane-based visibility estimation utilities.

This module computes quantitative lane visibility metrics
and fuses them into a single lane_visibility_score in [0, 1].
"""

import numpy as np


def compute_lane_visibility(metrics: dict) -> float:
    """
    Compute a single lane visibility score from UNet-derived metrics.

    Expected keys in metrics:
    - lane_pixel_ratio
    - bottom_half_density
    - vertical_continuity

    Returns:
        lane_visibility_score (float in [0, 1])
    """

    lane_pixel_ratio = metrics.get("lane_pixel_ratio", 0.0)
    bottom_half_density = metrics.get("bottom_half_density", 0.0)
    vertical_continuity = metrics.get("vertical_continuity", 0.0)

    # Weighted fusion (intentionally simple & explainable)
    score = (
        0.4 * vertical_continuity +
        0.4 * bottom_half_density +
        0.2 * lane_pixel_ratio
    )

    return float(np.clip(score, 0.0, 1.0))


def fuse_weather_and_lane(
    weather_label: str,
    weather_confidence: float,
    lane_visibility_score: float,
) -> dict:
    """
    Fuse calibrated weather confidence with lane structure.

    Returns:
        {
            "effective_visibility": str,
            "effective_visibility_score": float
        }
    """

    # Weather-based penalty
    if weather_label in {"rainy", "snowy", "foggy"}:
        penalty = 0.2
    elif weather_label in {"overcast", "night"}:
        penalty = 0.1
    else:
        penalty = 0.0

    effective_score = (
        0.6 * weather_confidence +
        0.4 * lane_visibility_score
    ) - penalty

    effective_score = float(np.clip(effective_score, 0.0, 1.0))

    if effective_score >= 0.65:
        visibility = "high"
    elif effective_score >= 0.4:
        visibility = "moderate"
    else:
        visibility = "low"

    return {
        "effective_visibility": visibility,
        "effective_visibility_score": round(effective_score, 3),
    }
