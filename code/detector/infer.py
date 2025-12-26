from pathlib import Path
from ultralytics import YOLO

from code.detector.config import YOLO_MODEL_PATH, IMAGE_SIZE, CONF_THRESHOLD


# -----------------------
# LOAD ONNX MODEL ONCE
# -----------------------
_model = YOLO(YOLO_MODEL_PATH)  # ONNX = inference-only


def detect_objects(image_path: str):
    """
    Run YOLO object detection on a single image (ONNX).

    Returns:
        List of dicts:
        [
            {"class": "car", "confidence": 0.93},
            {"class": "person", "confidence": 0.81},
            ...
        ]
    """
    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"

    results = _model.predict(
        source=image_path,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            detections.append({
                "class": _model.names[cls_id],
                "confidence": float(box.conf.item())
            })

    return detections


# Quick sanity test
if __name__ == "__main__":
    out = detect_objects("results/showcase/00a2e3ca-5c856cde.jpg")
    print(out[:5])
