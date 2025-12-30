import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
TUSIMPLE_ROOT = Path("datasets/tusimple/TUSimple/train_set")
OUTPUT_ROOT = Path("datasets/tusimple_unet")

IMG_OUT = OUTPUT_ROOT / "images" / "train"
MASK_OUT = OUTPUT_ROOT / "masks" / "train"

IMG_OUT.mkdir(parents=True, exist_ok=True)
MASK_OUT.mkdir(parents=True, exist_ok=True)

LABEL_FILES = [
    "label_data_0313.json",
    "label_data_0531.json",
    "label_data_0601.json",
]

IMG_HEIGHT = 720
IMG_WIDTH = 1280

# ---------------- MASK DRAW ----------------
def draw_lane_mask(label):
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    h_samples = label["h_samples"]
    lanes = label["lanes"]

    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:
                points.append((int(x), int(y)))

        if len(points) >= 2:
            cv2.polylines(
                mask,
                [np.array(points, dtype=np.int32)],
                isClosed=False,
                color=255,
                thickness=5,
            )

    return mask

# ---------------- MAIN ----------------
sample_id = 0
skipped = 0

for label_file in LABEL_FILES:
    label_path = TUSIMPLE_ROOT / label_file
    print(f"Processing {label_file}")

    with open(label_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)

            img_path = TUSIMPLE_ROOT / data["raw_file"]

            if not img_path.exists():
                skipped += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue

            mask = draw_lane_mask(data)

            if mask.sum() == 0:
                skipped += 1
                continue

            out_img = IMG_OUT / f"{sample_id}.png"
            out_mask = MASK_OUT / f"{sample_id}.png"

            cv2.imwrite(str(out_img), img)
            cv2.imwrite(str(out_mask), mask)

            sample_id += 1

print("\n======================")
print(f"Saved samples : {sample_id}")
print(f"Skipped       : {skipped}")
print("======================")
