from pathlib import Path

from code.classifier.infer import predict_weather
from code.detector.infer import detect_objects
from code.unet.infer import predict_lane_mask

from analysis.lane_visibility_metrics import (
    compute_lane_metrics,
    compute_lane_reliability,
)


def run_pipeline(image_path: str):
    image_path = str(image_path)

    # -------------------------
    # WEATHER
    # -------------------------
    weather_label, weather_conf = predict_weather(image_path)

    # -------------------------
    # OBJECT DETECTION
    # -------------------------
    objects = detect_objects(image_path)

    # -------------------------
    # LANE SEGMENTATION
    # -------------------------
    lane_mask = predict_lane_mask(image_path)
    lane_metrics = compute_lane_metrics(lane_mask)
    lane_reliability = compute_lane_reliability(lane_metrics)

    # -------------------------
    # RISK PROFILE (SAFE LOGIC)
    # -------------------------
    if lane_reliability == "unreliable":
        visibility = "low"
    else:
        visibility = lane_metrics["final_visibility"]

    risk_profile = {
        "visibility": visibility,
        "weather_confidence": round(float(weather_conf), 2),
        "num_objects": len(objects),
    }

    # -------------------------
    # FINAL OUTPUT
    # -------------------------
    return {
        "weather": {
            "label": weather_label,
            "confidence": round(float(weather_conf), 2),
        },
        "lane": {
            "visibility": lane_metrics["final_visibility"],
            "reliability": lane_reliability,
            "metrics": lane_metrics,
        },
        "objects": objects,
        "risk_profile": risk_profile,
    }


if __name__ == "__main__":
    # quick sanity test
    test_img = Path("results/showcase/00a2e3ca-5c856cde.jpg")
    if test_img.exists():
        out = run_pipeline(str(test_img))
        print("\n=== FINAL PIPELINE OUTPUT ===")
        print(out)
