# code/pipeline/lane.py
from pathlib import Path
import cv2

from external.code.lane.ufld_detector import UFLDLaneDetector

UFLD_WEIGHTS = Path("external/UFLD/weights/culane_res18.pth")
assert UFLD_WEIGHTS.exists(), f"Missing UFLD weights: {UFLD_WEIGHTS}"

_lane_detector = UFLDLaneDetector(weight_path=str(UFLD_WEIGHTS))

def predict_lanes(image_path: str):
    """
    Returns:
        lanes: list
        lane_visibility: "high" | "medium" | "low"
    """
    image = cv2.imread(image_path)
    assert image is not None, f"Failed to load image: {image_path}"

    lanes = _lane_detector.detect(image)

    lane_count = len(lanes)
    if lane_count >= 4:
        visibility = "high"
    elif lane_count >= 2:
        visibility = "medium"
    else:
        visibility = "low"

    return lanes, visibility
