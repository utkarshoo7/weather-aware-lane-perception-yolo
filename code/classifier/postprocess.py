"""
Weather Post-processing Logic
-----------------------------
Refines raw weather predictions using confidence scores
and scene-level cues (lane visibility).

This module enforces a stable, interpretable output space.
"""

from typing import Tuple


FINAL_WEATHER_CLASSES = {"clear", "rainy", "snowy", "night"}


def refine_weather(
    raw_label: str,
    confidence: float,
    lane_visibility: str,
) -> Tuple[str, float]:
    """
    Refine raw weather prediction into a final, canonical label.

    Args:
        raw_label (str): Raw model output label
        confidence (float): Softmax confidence for raw_label
        lane_visibility (str): Estimated lane visibility ("high", "medium", "low")

    Returns:
        (final_label, adjusted_confidence)
    """

    # --------------------------------------------------
    # Rule 1: Night dominates visual interpretation
    # --------------------------------------------------
    # Extremely low lane visibility is treated as night,
    # regardless of classifier output.
    if lane_visibility == "low":
        return "night", confidence

    # --------------------------------------------------
    # Rule 2: Overcast is an intermediate, not final class
    # --------------------------------------------------
    # Overcast with low confidence behaves like clear.
    # Overcast with high confidence is pushed to rainy
    # due to reduced contrast and diffuse lighting.
    if raw_label == "overcast":
        if confidence < 0.60:
            return "clear", confidence
        return "rainy", confidence

    # --------------------------------------------------
    # Rule 3: Valid classes pass through unchanged
    # --------------------------------------------------
    if raw_label in FINAL_WEATHER_CLASSES:
        return raw_label, confidence

    # --------------------------------------------------
    # Rule 4: Safety fallback
    # --------------------------------------------------
    # Unknown or unsupported labels default to clear
    # with confidence attenuation.
    return "clear", confidence * 0.8
