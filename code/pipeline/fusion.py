"""
Fusion Module
-------------
Confidence-aware fusion of weather, lane, and object signals.

This module does NOT perform learning.
It applies deterministic safety rules to reconcile
conflicting perception outputs.
"""

from typing import Dict, List

# -------------------------
# TUNABLE SAFETY CONSTANTS
# -------------------------
MIN_WEATHER_CONF = 0.50        # Below this, perception is unreliable
MIN_LANE_RATIO = 0.01          # Minimum visible lane pixels (empirical)
DEGRADE_ORDER = ["high", "medium", "low"]


def _downgrade_visibility(level: str) -> str:
    """Reduce visibility by one step."""
    if level not in DEGRADE_ORDER:
        return "low"
    idx = DEGRADE_ORDER.index(level)
    return DEGRADE_ORDER[min(idx + 1, len(DEGRADE_ORDER) - 1)]


def fuse_results(
    weather: Dict,
    lane_metrics: Dict,
    detections: List[Dict]
) -> Dict:
    """
    Fuse weather, lane, and object perception into a
    final lane visibility assessment.

    Args:
        weather: {
            "label": str,
            "confidence": float
        }
        lane_metrics: {
            "final_visibility": str,
            "lane_pixel_ratio": float
        }
        detections: list of detected objects

    Returns:
        dict:
            {
                "lane_visibility": str,
                "weather_confidence": float,
                "lane_ratio": float,
                "object_count": int
            }
    """

    weather_label = weather["label"]
    weather_conf = float(weather["confidence"])

    lane_visibility = lane_metrics["final_visibility"]
    lane_ratio = float(lane_metrics["lane_pixel_ratio"])

    # ==================================================
    # RULE 1: LOW WEATHER CONFIDENCE DEGRADES LANES
    # ==================================================
    if weather_conf < MIN_WEATHER_CONF:
        lane_visibility = _downgrade_visibility(lane_visibility)

    # ==================================================
    # RULE 2: ADVERSE WEATHER + WEAK LANES
    # ==================================================
    if weather_label in {"rainy", "snowy"} and lane_ratio < MIN_LANE_RATIO:
        lane_visibility = "low"

    # ==================================================
    # FINAL FUSED OUTPUT
    # ==================================================
    return {
        "lane_visibility": lane_visibility,
        "weather_confidence": round(weather_conf, 3),
        "lane_ratio": round(lane_ratio, 4),
        "object_count": len(detections),
    }
