# code/pipeline/main.py

from code.classifier.infer import predict_weather
from code.pipeline.lane import predict_lanes
from code.pipeline.decision_policy import decide_action_mode

def run_pipeline(image_path: str) -> dict:
    """
    Single source of truth for inference output
    """

    # Weather
    weather_out = predict_weather(image_path)
    weather_label = weather_out["label"]
    weather_conf = weather_out["confidence"]

    # Canonical mapping
    if weather_label == "overcast":
        weather_label = "clear"

    # Lanes
    lanes, lane_visibility = predict_lanes(image_path)

    # Decision
    decision = decide_action_mode(
        weather=weather_label,
        confidence=weather_conf,
        lane_visibility=lane_visibility
    )

    return {
        "weather": {
            "label": weather_label,
            "confidence": weather_conf
        },
        "lane": {
            "count": len(lanes),
            "visibility": lane_visibility
        },
        "decision": decision
    }
