from ultralytics import YOLO

_model = YOLO("models/yolo/yolov8n.pt")


def run_yolo(image_path):
    results = _model(image_path, verbose=False)[0]

    detections = []
    for box in results.boxes:
        detections.append({
            "cls": int(box.cls),
            "conf": float(box.conf),
            "xyxy": box.xyxy[0].tolist()
        })

    return detections
