from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# TuSimple (lane segmentation)
TUSIMPLE_ROOT = PROJECT_ROOT / "datasets" / "tusimple" / "TUSimple"
TUSIMPLE_IMAGES = TUSIMPLE_ROOT / "train_set" / "clips"
TUSIMPLE_MASKS = TUSIMPLE_ROOT / "train_set" / "seg_label"
