"""
YOLOv8 object detection inference module.

This module provides a lightweight, inference-only wrapper
around a pretrained YOLOv8 model for object detection.

Design goals:
- Stable output format
- CPU-safe execution
- Minimal coupling to YOLO internals
"""

from pathlib import Path
from typing import List, Dict

from ultralytics import YOLO


# ------------------------------------------------------------------
# Model loading (once at import time)
# ------------------------------------------------------------------
MODEL_PATH = Path("models/yolo/yolov8n.pt")
assert MODEL_PATH.exists(), f"YOLO model not found: {MODEL_PATH}"

_model = YOLO(str(MODEL_PATH))


# ------------------------------------------------------------------
# Inference API
# ------------------------------------------------------------------
def run_yolo(image_path: str) -> List[Dict]:
    """
    Run YOLO object detection on a single image.

    Args:
        image_path (str): Path to input image

    Returns:
        List[Dict]: List of detections, each with:
            {
                "label": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
    """

    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"

    results = _model(
        source=str(image_path),
        verbose=False,
        device="cpu"  # explicit CPU for portability
    )

    detections: List[Dict] = []

    if not results or results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    names = _model.names

    for box in boxes:
        cls_id = int(box.cls.item())
        detections.append({
            "label": names.get(cls_id, str(cls_id)),
            "confidence": float(box.conf.item()),
            "bbox": box.xyxy[0].tolist()
        })

    return detections
