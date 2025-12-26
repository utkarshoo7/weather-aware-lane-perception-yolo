def fuse_results(weather_label, weather_conf, detections):
    # Simple, explainable logic (NO ML here)
    if weather_label in ["rainy", "snowy", "foggy"]:
        visibility = "low"
    elif weather_label in ["overcast"]:
        visibility = "moderate"
    else:
        visibility = "normal"

    return {
        "weather": {
            "label": weather_label,
            "confidence": round(weather_conf, 2)
        },
        "objects": detections,
        "risk_profile": {
            "visibility": visibility,
            "weather_confidence": round(weather_conf, 2),
            "num_objects": len(detections)
        }
    }
