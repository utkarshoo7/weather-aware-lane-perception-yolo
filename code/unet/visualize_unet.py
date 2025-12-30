import torch
import cv2
import numpy as np
from pathlib import Path

# ---------------- CONFIG ----------------
MODEL_PATH = Path("models/unet/unet_lane_v1.pth")
IMG_DIR = Path("datasets/tusimple_unet/images/train")
OUT_DIR = Path("results/unet_vis")

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (256, 512)

# ---------------- UNET (same as training) ----------------
class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_c, out_c, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_c, out_c, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)

        self.pool = torch.nn.MaxPool2d(2)

        self.mid = block(256, 512)

        self.up3 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = block(512, 256)

        self.up2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)

        self.up1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)

        self.out = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.mid(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))

# ---------------- LOAD MODEL ----------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- RUN VIS ----------------
images = sorted(list(IMG_DIR.glob("*.png")))[:10]

for img_path in images:
    img = cv2.imread(str(img_path))
    orig = img.copy()

    img_resized = cv2.resize(img, IMG_SIZE)
    img_tensor = torch.from_numpy(img_resized / 255.0).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)[0, 0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

    overlay = orig.copy()
    overlay[mask > 0] = (0, 255, 0)

    blended = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)

    out_path = OUT_DIR / img_path.name
    cv2.imwrite(str(out_path), blended)

print(f"Saved visualizations to {OUT_DIR}")
