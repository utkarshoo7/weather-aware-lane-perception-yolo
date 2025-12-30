import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
DATA_ROOT = Path("datasets/tusimple_unet")
IMG_DIR = DATA_ROOT / "images/train"
MASK_DIR = DATA_ROOT / "masks/train"

BATCH_SIZE = 4
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (256, 512)

MODEL_OUT = Path("models/unet")
MODEL_OUT.mkdir(parents=True, exist_ok=True)

# ---------------- DATASET ----------------
class LaneDataset(Dataset):
    def __init__(self):
        self.images = sorted(IMG_DIR.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = MASK_DIR / img_path.name

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, IMG_SIZE)
        mask = cv2.resize(mask, IMG_SIZE)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

# ---------------- UNET ----------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.mid = block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.mid(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out(d1))

# ---------------- TRAIN ----------------
def train():
    ds = LaneDataset()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    for epoch in range(EPOCHS):
        model.train()
        total = 0

        for imgs, masks in tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total/len(dl):.4f}")

    torch.save(model.state_dict(), MODEL_OUT / "unet_lane_v1.pth")
    print("Saved models/unet/unet_lane_v1.pth")

if __name__ == "__main__":
    train()
