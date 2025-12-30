"""
UNet Lane Segmentation Inference (Stable, Calibrated)

- Outputs binary lane mask {0,1}
- No double-normalization
- Conservative threshold for TuSimple
"""

from pathlib import Path
import cv2
import numpy as np
import torch

from code.unet.model import UNet

# ========================
# CONFIG
# ========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("models/unet/unet_lane_v1.pth")
LANE_THRESHOLD = 0.20   # calibrated for TuSimple

# ========================
# LOAD MODEL (ONCE)
# ========================
_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    _model = model
    return model


# ========================
# PUBLIC API
# ========================
def predict_lane_mask(image_path: str) -> np.ndarray:
    """
    Returns:
        np.ndarray (H,W) uint8 binary mask {0,1}
    """
    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"

    model = _load_model()

    # --- Load image ---
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # --- Resize to training size ---
    img_resized = cv2.resize(img, (1280, 720))

    # --- Normalize ONCE ---
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(img_tensor)[0, 0].cpu().numpy()

    # --- DEBUG SAFE (keep forever) ---
    print(
        f"[UNet] prob stats → "
        f"min={prob.min():.4f}, "
        f"max={prob.max():.4f}, "
        f"mean={prob.mean():.4f}"
    )

    # --- Threshold ---
    mask = (prob >= LANE_THRESHOLD).astype(np.uint8)

    # --- Resize back ---
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask
