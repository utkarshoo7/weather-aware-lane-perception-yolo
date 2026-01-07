# analysis/decision_engine.py

"""
Decision Engine
---------------
This module defines the final decision logic of the perception pipeline.

It takes structured outputs from perception modules (weather, lane, image stats)
and produces a stable, interpretable decision summary.

This file is intentionally rule-based to ensure:
- determinism
- interpretability
- auditability (important for academic review)
"""

from typing import Dict


# =========================
# Canonical Definitions
# =========================

# Supported weather states used throughout the system
CANONICAL_WEATHER = {"clear", "rainy", "snowy"}


def canonicalize_weather(label: str) -> str:
    """
    Normalize raw weather labels into a canonical set.

    This avoids downstream ambiguity caused by synonymous labels
    (e.g., 'overcast' vs 'clear').
    """
    if label in ("overcast", "cloudy"):
        return "clear"

    if label in CANONICAL_WEATHER:
        return label

    # Fallback for unexpected model outputs
    return "uncertain"


# =========================
# Lighting Estimation
# =========================

def decide_lighting(image_stats: Dict) -> str:
    """
    Determine lighting condition from low-level image statistics.

    This is a deterministic rule-based approximation used
    only for contextual awareness (not learning).
    """
    return "night" if image_stats.get("is_night", False) else "day"


# =========================
# Risk Estimation
# =========================

def decide_risk(lane_visibility: str, weather: str) -> str:
    """
    Estimate driving risk level based on perception quality.

    Priority:
    1. Lane visibility (strongest signal)
    2. Weather condition
    """
    if lane_visibility == "low":
        return "high"

    if weather in ("rainy", "snowy"):
        return "caution"

    return "normal"


# =========================
# Decision Engine (Core)
# =========================

def decision_engine(pipeline_out: Dict) -> Dict:
    """
    SINGLE SOURCE OF TRUTH for decision output.

    This function guarantees:
    - a stable output schema
    - no dependency on raw model internals
    - graceful handling of missing signals
    """

    # -------------------------
    # Weather handling
    # -------------------------
    weather_raw = pipeline_out["weather"]["label"]
    confidence = float(pipeline_out["weather"]["confidence"])

    weather = canonicalize_weather(weather_raw)

    # -------------------------
    # Lighting context
    # -------------------------
    image_stats = pipeline_out.get("image_stats", {})
    lighting = decide_lighting(image_stats)

    # -------------------------
    # Lane context
    # -------------------------
    lane_visibility = pipeline_out["lane"].get("visibility")

    # -------------------------
    # Risk assessment
    # -------------------------
    risk_level = decide_risk(lane_visibility, weather)

    # -------------------------
    # Decision validity check
    # -------------------------
    # A decision is considered valid only if:
    # - weather confidence is sufficient
    # - lane visibility is not critically low
    decision_valid = (confidence >= 0.5) and (lane_visibility != "low")

    # -------------------------
    # Final structured output
    # -------------------------
    return {
        "weather": weather,
        "confidence": confidence,
        "lighting": lighting,
        "lane_visibility": lane_visibility,
        "risk_level": risk_level,
        "decision_valid": decision_valid,
    }
