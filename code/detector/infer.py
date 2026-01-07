"""
YOLO Object Detection (Inference Only)
-------------------------------------
Runs object detection using a pretrained YOLOv8 ONNX model.
The model is loaded once at import time for efficiency.
"""

from pathlib import Path
from typing import List, Dict

from ultralytics import YOLO

from code.detector.config import (
    YOLO_MODEL_PATH,
    IMAGE_SIZE,
    CONF_THRESHOLD,
)

# ==================================================
# LOAD MODEL (ONCE)
# ==================================================

# ONNX model is used strictly for inference (CPU-safe)
_DETECTOR = YOLO(YOLO_MODEL_PATH)


# ==================================================
# PUBLIC API
# ==================================================

def detect_objects(image_path: str) -> List[Dict]:
    """
    Run object detection on a single image.

    Args:
        image_path (str): Path to input image.

    Returns:
        List[Dict]: Each detection has:
            {
                "class": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = _DETECTOR.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    detections: List[Dict] = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class": _DETECTOR.names[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            })

    return detections


# ==================================================
# OPTIONAL LOCAL TEST
# ==================================================
if __name__ == "__main__":
    sample_img = Path("results/showcase/clear/sample.jpg")
    if sample_img.exists():
        preds = detect_objects(str(sample_img))
        print(preds[:3])
