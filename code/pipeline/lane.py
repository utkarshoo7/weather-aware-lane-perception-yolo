from external.code.lane.ufld_detector import UFLDLaneDetector

_detector = UFLDLaneDetector(
    "external/UFLD/weights/culane_18.pth"
)


def predict_lanes(image_path: str) -> int:
    import cv2
    img = cv2.imread(image_path)
    lanes = _detector.detect(img)
    return len(lanes)
