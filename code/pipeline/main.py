"""
FINAL PIPELINE CONTRACT
======================

This pipeline takes a single road-scene image as input and returns a
structured dictionary with three components:

1. weather:
   - label (str): predicted weather condition
   - confidence (float): model confidence in [0, 1]

2. objects:
   - list of detected objects
   - each object contains:
        - class (str)
        - confidence (float)

3. risk_profile:
   - visibility (str): qualitative visibility estimate
   - weather_confidence (float)
   - num_objects (int)

RETURN FORMAT (GUARANTEED):

{
    "weather": {
        "label": str,
        "confidence": float
    },
    "objects": [
        {"class": str, "confidence": float}
    ],
    "risk_profile": {
        "visibility": str,
        "weather_confidence": float,
        "num_objects": int
    }
}

This contract MUST remain unchanged.
"""

from pathlib import Path

from code.classifier.infer import predict_weather
from code.detector.infer import detect_objects


def _estimate_visibility(weather_label: str) -> str:
    """
    Simple rule-based visibility estimation.
    This is INTENTIONALLY simple and replaceable later.
    """
    if weather_label in {"rainy", "snowy", "foggy"}:
        return "low"
    if weather_label in {"overcast", "night"}:
        return "moderate"
    return "normal"


def run_pipeline(image_path: str):
    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"

    # 1. Weather prediction
    weather_label, weather_conf = predict_weather(image_path)

    # 2. Object detection
    detections = detect_objects(image_path)

    # 3. Risk profiling
    visibility = _estimate_visibility(weather_label)

    # 4. Final contract output
    output = {
        "weather": {
            "label": weather_label,
            "confidence": round(weather_conf, 2)
        },
        "objects": detections,
        "risk_profile": {
            "visibility": visibility,
            "weather_confidence": round(weather_conf, 2),
            "num_objects": len(detections)
        }
    }

    return output


if __name__ == "__main__":
    # Demo image (change if needed)
    demo_image = "results/showcase/00a2e3ca-5c856cde.jpg"

    result = run_pipeline(demo_image)
    print("\n=== FINAL PIPELINE OUTPUT ===")
    print(result)
