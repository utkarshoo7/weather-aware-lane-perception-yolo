"""
FINAL PIPELINE CONTRACT (v3-compatible)

Adds UNet lane reasoning WITHOUT changing output schema.
"""

from pathlib import Path

from code.classifier.infer import predict_weather
from code.detector.infer import detect_objects
from analysis.lane_visibility_metrics import compute_lane_visibility


def _estimate_visibility(weather_label: str) -> str:
    if weather_label in {"rainy", "snowy", "foggy"}:
        return "low"
    if weather_label in {"overcast", "night"}:
        return "moderate"
    return "normal"


def run_pipeline(image_path: str):
    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"

    weather_label, weather_conf = predict_weather(image_path)
    detections = detect_objects(image_path)

    visibility = _estimate_visibility(weather_label)

    # UNet metrics (NOT exposed yet)
    lane_metrics = compute_lane_visibility(str(image_path))

    output = {
        "weather": {
            "label": weather_label,
            "confidence": round(weather_conf, 2),
        },
        "objects": detections,
        "risk_profile": {
            "visibility": visibility,
            "weather_confidence": round(weather_conf, 2),
            "num_objects": len(detections),
        },
    }

    return output


if __name__ == "__main__":
    img = "results/showcase/00a2e3ca-5c856cde.jpg"
    result = run_pipeline(img)

    print("\n=== FINAL PIPELINE OUTPUT ===")
    print(result)
