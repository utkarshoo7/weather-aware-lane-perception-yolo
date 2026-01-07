"""
Lane Detection Wrapper (UFLD)
-----------------------------
Provides lane visibility estimation.
Falls back safely on CPU-only systems.
"""

from pathlib import Path
from typing import Tuple, List
import cv2
import warnings

# ==================================================
# CONFIG
# ==================================================

UFLD_WEIGHTS = Path("external/UFLD/weights/culane_res18.pth")

if not UFLD_WEIGHTS.exists():
    raise FileNotFoundError(f"Missing UFLD weights: {UFLD_WEIGHTS}")

# ==================================================
# OPTIONAL UFLD INITIALIZATION
# ==================================================

_lane_detector = None
_UFLD_AVAILABLE = True

try:
    from external.code.lane.ufld_detector import UFLDLaneDetector
    _lane_detector = UFLDLaneDetector(weight_path=str(UFLD_WEIGHTS))
except Exception as e:
    # CUDA or dependency failure → safe fallback
    _UFLD_AVAILABLE = False
    warnings.warn(
        f"UFLD lane detector disabled (reason: {e}). "
        "Lane visibility will default to 'low'."
    )


# ==================================================
# PUBLIC API
# ==================================================

def predict_lanes(image_path: str) -> Tuple[List, str]:
    """
    Predict lane structure and visibility.

    Args:
        image_path (str): Path to input image

    Returns:
        lanes (list): Detected lanes (empty if unavailable)
        lane_visibility (str): "high" | "medium" | "low"
    """

    # ---------- Load image ----------
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # ---------- Fallback (CPU-safe) ----------
    if not _UFLD_AVAILABLE:
        return [], "low"

    # ---------- Run detection ----------
    try:
        lanes = _lane_detector.detect(image)
    except Exception:
        # Runtime safety
        return [], "low"

    # ---------- Visibility heuristic ----------
    lane_count = len(lanes)

    if lane_count >= 4:
        visibility = "high"
    elif lane_count >= 2:
        visibility = "medium"
    else:
        visibility = "low"

    return lanes, visibility
