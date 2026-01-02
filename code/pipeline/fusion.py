def fuse_results(weather, detections, lanes):
    """
    Simple rule-based fusion
    """

    visibility = "high"

    if weather["label"] in ["rainy", "foggy", "snowy"]:
        visibility = "medium"

    if len(lanes) < 2:
        visibility = "low"

    return {
        "visibility": visibility
    }
