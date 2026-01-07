# code/pipeline/decision_policy.py

def decide_action_mode(
    weather: str,
    confidence: float,
    lane_visibility: str
) -> dict:
    """
    Deterministic decision policy.

    Returns:
        {
            "mode": str,
            "trusted": bool,
            "reason": str
        }
    """

    # -------------------------
    # Normalize lane visibility
    # -------------------------
    if lane_visibility in (None, "unknown"):
        lane_visibility = "medium"

    # -------------------------
    # Confidence gate (dominant)
    # -------------------------
    if confidence < 0.4:
        return {
            "mode": "caution",
            "trusted": False,
            "reason": "low_weather_confidence"
        }

    # -------------------------
    # Lane visibility dominates
    # -------------------------
    if lane_visibility == "low":
        return {
            "mode": "slow",
            "trusted": False,
            "reason": "poor_lane_visibility"
        }

    # -------------------------
    # Weather-based rules
    # -------------------------
    if weather == "snowy":
        return {
            "mode": "slow",
            "trusted": True,
            "reason": "snow_conditions"
        }

    if weather == "rainy":
        return {
            "mode": "normal",
            "trusted": True,
            "reason": "rain_conditions"
        }

    # -------------------------
    # Clear / default
    # -------------------------
    return {
        "mode": "normal",
        "trusted": True,
        "reason": "clear_conditions"
    }
