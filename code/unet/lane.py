import numpy as np


def lane_confidence(mask: np.ndarray) -> float:
    """
    Simple, stable confidence proxy:
    proportion of lane pixels.
    """
    if mask is None:
        return 0.0

    total = mask.size
    lane_pixels = mask.sum()

    conf = lane_pixels / total
    return round(float(conf), 3)
