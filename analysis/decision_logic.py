# analysis/decision_logic.py

"""
Decision Logic
--------------
This module implements the core rule-based reasoning layer
used after perception.

It interprets:
- weather prediction + confidence
- lane visibility
- scene complexity

and produces a compact environment state with an associated risk level.

This logic is intentionally deterministic and transparent.
"""


# =========================
# Weather Normalization
# =========================

def normalize_weather(label: str) -> str:
    """
    Normalize weather labels into canonical classes.

    This prevents semantic duplication (e.g., 'overcast' vs 'clear')
    from influencing downstream logic.
    """
    if label in ("overcast", "cloudy"):
        return "clear"
    return label


# =========================
# Confidence Interpretation
# =========================

def weather_confidence_band(label: str, confidence: float) -> str:
    """
    Convert raw confidence score into an interpretable band.

    These bands are used for decision robustness,
    not for learning or optimization.
    """
    if confidence >= 0.70:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


# =========================
# Environment State Logic
# =========================

def compute_environment_state(
    weather_label: str,
    weather_conf: float,
    lane_visibility: str,
    lane_count: int,
    detections: int,
):
    """
    CORE DECISION LOGIC (STEP 5)

    Returns:
        environment_state (str):
            clear | adverse | trusted_adverse | ambiguous

        risk_level (str):
            low | medium | high
    """

    # -------------------------
    # Normalize inputs
    # -------------------------
    weather_label = normalize_weather(weather_label)
    confidence_band = weather_confidence_band(weather_label, weather_conf)

    # -------------------------
    # Adverse weather (snow)
    # -------------------------
    # Snow with high confidence is considered a trusted adverse scenario
    if weather_label == "snowy" and confidence_band == "high":
        return "trusted_adverse", "high"

    # -------------------------
    # Adverse weather (rain)
    # -------------------------
    # Rain combined with poor lane visibility increases risk
    if weather_label == "rainy" and confidence_band != "low":
        if lane_visibility in ("medium", "low"):
            return "adverse", "high"
        return "adverse", "medium"

    # -------------------------
    # Clear weather (penalized)
    # -------------------------
    # Clear predictions are penalized when confidence or lane quality is weak
    if weather_label == "clear":
        if confidence_band == "low":
            return "ambiguous", "medium"

        if lane_visibility == "low":
            return "ambiguous", "medium"

        return "clear", "low"

    # -------------------------
    # Fallback (uncertain cases)
    # -------------------------
    return "ambiguous", "medium"
