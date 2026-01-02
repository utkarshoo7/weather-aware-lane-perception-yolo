from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# ================= CONFIG =================
CULANE_ROOT = Path("datasets/CULane")

IMG_OUT = Path("datasets/culane_unet/images")
MASK_OUT = Path("datasets/culane_unet/masks")

TRAIN_LIST = CULANE_ROOT / "list/train.txt"
VAL_LIST   = CULANE_ROOT / "list/val.txt"

SEG_ROOT = CULANE_ROOT / "laneseg_label_w16"

IMG_OUT_TRAIN = IMG_OUT / "train"
IMG_OUT_VAL   = IMG_OUT / "val"
MASK_OUT_TRAIN = MASK_OUT / "train"
MASK_OUT_VAL   = MASK_OUT / "val"

for p in [IMG_OUT_TRAIN, IMG_OUT_VAL, MASK_OUT_TRAIN, MASK_OUT_VAL]:
    p.mkdir(parents=True, exist_ok=True)

# ==========================================


def process_split(list_file, img_out, mask_out, tag):
    saved = 0
    skipped = 0

    with open(list_file, "r") as f:
        lines = f.readlines()

    print(f"Processing {tag}: {len(lines)} samples")

    for line in tqdm(lines):
        rel_path = line.strip()

        # 🔥 CRITICAL FIX: remove leading slash
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]

        img_path = CULANE_ROOT / rel_path
        mask_path = SEG_ROOT / rel_path

        # mask is PNG, not JPG
        mask_path = mask_path.with_suffix(".png")

        if not img_path.exists() or not mask_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            skipped += 1
            continue

        # Convert mask to binary lane mask
        # CULane uses >0 for lane pixels
        mask = (mask > 0).astype(np.uint8) * 255

        name = rel_path.replace("/", "_")

        cv2.imwrite(str(img_out / name), img)
        cv2.imwrite(str(mask_out / name), mask)

        saved += 1

    return saved, skipped


def main():
    assert CULANE_ROOT.exists(), "CULane root not found"
    assert TRAIN_LIST.exists(), "train.txt not found"
    assert VAL_LIST.exists(), "val.txt not found"
    assert SEG_ROOT.exists(), "laneseg_label_w16 not found"

    print(f"[OK] CULane root detected at: {CULANE_ROOT}")

    train_saved, train_skipped = process_split(
        TRAIN_LIST, IMG_OUT_TRAIN, MASK_OUT_TRAIN, "TRAIN"
    )

    val_saved, val_skipped = process_split(
        VAL_LIST, IMG_OUT_VAL, MASK_OUT_VAL, "VAL"
    )

    print("\n======================")
    print(f"Train saved   : {train_saved}")
    print(f"Train skipped : {train_skipped}")
    print(f"Val saved     : {val_saved}")
    print(f"Val skipped   : {val_skipped}")
    print("======================\n")


if __name__ == "__main__":
    main()
