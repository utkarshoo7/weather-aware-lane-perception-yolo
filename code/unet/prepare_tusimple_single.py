import json
from pathlib import Path

import cv2
import numpy as np

# ----------------------------
# PATHS (CHANGE ONLY IF NEEDED)
# ----------------------------
TUSIMPLE_ROOT = Path("datasets/tusimple/TUSimple/train_set")
LABEL_FILE = TUSIMPLE_ROOT / "label_data_0313.json"

OUT_IMG_DIR = Path("datasets/tusimple_unet/images/train")
OUT_MASK_DIR = Path("datasets/tusimple_unet/masks/train")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Read ONE label entry
    with open(LABEL_FILE, "r") as f:
        line = f.readline()
        label = json.loads(line)

    # Load image (20th frame)
    img_path = TUSIMPLE_ROOT / label["raw_file"]
    image = cv2.imread(str(img_path))

    if image is None:
        raise RuntimeError(f"Failed to load image: {img_path}")

    h, w, _ = image.shape

    # Create empty mask
    mask = np.zeros((h, w), dtype=np.uint8)

    lanes = label["lanes"]
    h_samples = label["h_samples"]

    # Draw lanes
    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:
                points.append((int(x), int(y)))

        if len(points) >= 2:
            cv2.polylines(
                mask,
                [np.array(points)],
                isClosed=False,
                color=255,
                thickness=6
            )

    # Save outputs
    out_name = img_path.stem + ".png"

    cv2.imwrite(str(OUT_IMG_DIR / out_name), image)
    cv2.imwrite(str(OUT_MASK_DIR / out_name), mask)

    print("Saved:")
    print("Image:", OUT_IMG_DIR / out_name)
    print("Mask :", OUT_MASK_DIR / out_name)


if __name__ == "__main__":
    main()
