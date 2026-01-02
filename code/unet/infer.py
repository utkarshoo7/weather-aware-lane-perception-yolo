import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path

# -------------------------
# Simple UNet definition
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


# -------------------------
# Load model ONCE
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path("models/unet/unet_lane_v1.pth")

_model = UNet().to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
_model.load_state_dict(state)
_model.eval()


# -------------------------
# Inference function
# -------------------------
def predict_lane_mask(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)

    x = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = _model(x)
        mask = torch.sigmoid(pred)[0, 0].cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (1280, 720))

    return mask
