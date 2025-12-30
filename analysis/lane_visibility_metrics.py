"""
Lane Visibility Metrics (UNet-based)

Quantifies how visible / reliable lanes are.
Used for risk reasoning (v4+).
"""

import numpy as np
from code.unet.infer import predict_lane_mask
from code.classifier.infer import predict_weather


def compute_lane_visibility(image_path: str) -> dict:
    mask = predict_lane_mask(image_path)

    total_pixels = mask.size
    lane_pixels = mask.sum()

    lane_pixel_ratio = lane_pixels / total_pixels if total_pixels > 0 else 0.0

    h = mask.shape[0]
    bottom_half = mask[h // 2 :, :]
    bottom_half_density = (
        bottom_half.sum() / bottom_half.size if bottom_half.size > 0 else 0.0
    )

    vertical_continuity = (
        (mask.sum(axis=1) > 0).mean()
        if mask.shape[0] > 0
        else 0.0
    )

    weather_label, weather_conf = predict_weather(image_path)

    return {
        "weather": weather_label,
        "weather_confidence": round(weather_conf, 2),
        "lane_pixel_ratio": round(lane_pixel_ratio, 4),
        "bottom_half_density": round(bottom_half_density, 4),
        "vertical_continuity": round(vertical_continuity, 4),
    }


if __name__ == "__main__":
    img = "results/showcase/00a2e3ca-5c856cde.jpg"
    metrics = compute_lane_visibility(img)

    print("\n=== LANE VISIBILITY METRICS ===")
    for k, v in metrics.items():
        print(f"{k:22}: {v}")
