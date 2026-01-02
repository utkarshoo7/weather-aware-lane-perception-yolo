from code.classifier.infer import predict_weather
from code.pipeline.lane import predict_lanes
from code.yolo.infer import run_yolo


def run_pipeline(image_path: str):
    weather_label, weather_conf = predict_weather(image_path)

    detections = run_yolo(image_path)

    lane_count = predict_lanes(image_path)

    # Simple risk logic
    if lane_count >= 2 and weather_conf > 0.5:
        visibility = "high"
    elif lane_count >= 1:
        visibility = "medium"
    else:
        visibility = "low"

    return {
        "weather": {
            "label": weather_label,
            "confidence": float(weather_conf),
        },
        "detections": detections,
        "lane_count": lane_count,
        "risk_profile": {
            "visibility": visibility
        }
    }


if __name__ == "__main__":
    img = "results/showcase/00a2e3ca-5c856cde.jpg"
    out = run_pipeline(img)
    print(out)
