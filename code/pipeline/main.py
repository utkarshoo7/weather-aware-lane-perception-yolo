# code/pipeline/main.py

from code.classifier.infer import predict_weather
from code.pipeline.decision_policy import decide_action_mode


def run_pipeline(image_path: str) -> dict:
    """
    Single source of truth for inference output.

    This pipeline is intentionally CPU-only and demo-stable.
    Lane detection is disabled for the demo, but the output
    contract is preserved for analysis and reporting.
    """

    # =========================
    # Weather Classification
    # =========================
    weather_out = predict_weather(image_path)

    weather_label = weather_out["label"]
    weather_conf = float(weather_out["confidence"])

    # Canonical normalization (locked)
    if weather_label == "overcast":
        weather_label = "clear"

    # =========================
    # Lane (DISABLED FOR DEMO)
    # =========================
    lane_detected = False
    lane_count = 0

    # IMPORTANT:
    # Use a stable placeholder instead of None
    lane_visibility = "unknown"

    # =========================
    # Decision Policy
    # =========================
    decision = decide_action_mode(
        weather=weather_label,
        confidence=weather_conf,
        lane_visibility=lane_visibility
    )

    # =========================
    # Unified Output Contract
    # =========================
    return {
        "weather": {
            "label": weather_label,
            "confidence": weather_conf
        },
        "lane": {
            "detected": lane_detected,
            "count": lane_count,
            "visibility": lane_visibility
        },
        "decision": decision
    }
