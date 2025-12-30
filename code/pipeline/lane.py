# code/pipeline/lane.py

from pathlib import Path
import numpy as np

from code.unet.infer import infer_lane_mask


def lane_summary(image_path: Path) -> dict:
    """
    Converts UNet lane mask into lightweight metadata.
    NO raw masks returned.
    """
    mask = infer_lane_mask(image_path)

    lane_pixels = int(mask.sum())
    total_pixels = int(mask.size)
    coverage = lane_pixels / max(total_pixels, 1)

    return {
        "lane_pixels": lane_pixels,
        "lane_coverage": round(coverage, 4)
    }
