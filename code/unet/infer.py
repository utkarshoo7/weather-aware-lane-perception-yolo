"""
UNet Lane Segmentation Inference
--------------------------------
This module performs lane segmentation using a pretrained UNet model.

Design goals:
- CPU/GPU safe
- Deterministic output
- Explicit assumptions (resolution, thresholding)
- Suitable for academic review and demos
"""

from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn

# =====================================================
# MODEL ARCHITECTURE (MUST MATCH TRAINING)
# =====================================================

class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Lightweight UNet for lane segmentation.
    Input: RGB image
    Output: single-channel lane mask
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_PATH = Path("models/unet/unet_lane_v1.pth")
assert MODEL_PATH.exists(), f"Missing UNet weights: {MODEL_PATH}"

INPUT_SIZE = (256, 256)        # must match training
OUTPUT_SIZE = (1280, 720)      # demo / dataset resolution
THRESHOLD = 0.5                # sigmoid threshold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# LOAD MODEL (ONCE)
# =====================================================

_model = UNet().to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_model.load_state_dict(state_dict)
_model.eval()


# =====================================================
# INFERENCE API
# =====================================================

def predict_lane_mask(image_path: str) -> dict:
    """
    Predict lane segmentation mask for a single image.

    Args:
        image_path (str): path to RGB image

    Returns:
        dict:
            {
                "mask": np.ndarray (H x W, uint8),
                "lane_pixel_ratio": float
            }
    """

    img = cv2.imread(image_path)
    assert img is not None, f"Failed to load image: {image_path}"

    # --- Preprocessing (must match training) ---
    img_resized = cv2.resize(img, INPUT_SIZE)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # --- Inference ---
    with torch.no_grad():
        logits = _model(img_tensor)
        prob = torch.sigmoid(logits)[0, 0]

    mask_small = (prob > THRESHOLD).cpu().numpy().astype(np.uint8)

    # --- Restore original resolution ---
    mask = cv2.resize(mask_small, OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)

    # --- Metrics ---
    lane_ratio = float(mask.sum()) / mask.size

    return {
        "mask": mask,
        "lane_pixel_ratio": round(lane_ratio, 5),
    }
