# code/pipeline/fusion.py

def fuse_results(weather, lane_metrics, detections):
    """
    Smart confidence-aware fusion logic
    """

    label = weather["label"]
    conf = weather["confidence"]

    lane_vis = lane_metrics["final_visibility"]
    lane_ratio = lane_metrics["lane_pixel_ratio"]

    # --- Rule 1: Low weather confidence → downgrade visibility
    if conf < 0.5:
        if lane_vis == "high":
            lane_vis = "medium"
        elif lane_vis == "medium":
            lane_vis = "low"

    # --- Rule 2: Night is always risky (even with good lanes)
    if label == "night":
        lane_vis = "low"

    # --- Rule 3: Rain + weak lanes → low visibility
    if label == "rainy" and lane_ratio < 0.01:
        lane_vis = "low"

    # --- Rule 4: Overcast is never “perfect”
    if label == "overcast" and lane_vis == "high":
        lane_vis = "medium"

    # --- Final risk profile
    return {
        "visibility": lane_vis,
        "weather_confidence": round(conf, 3),
        "lane_ratio": round(lane_ratio, 4),
        "object_count": len(detections),
    }
