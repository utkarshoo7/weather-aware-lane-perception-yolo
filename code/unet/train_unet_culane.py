"""
Train UNet on CULane lane segmentation dataset
==============================================

This trains a binary lane segmentation UNet using
the prepared CULane UNet dataset.

Input:
- datasets/culane_unet/images/{train,val}
- datasets/culane_unet/masks/{train,val}

Output:
- models/unet/unet_lane_culane_v1.pth

This does NOT modify any pipeline contracts.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from code.unet.model import UNet

# ======================
# CONFIG
# ======================
IMG_DIR = Path("datasets/culane_unet/images")
MASK_DIR = Path("datasets/culane_unet/masks")

MODEL_OUT = Path("models/unet/unet_lane_culane_v1.pth")

EPOCHS = 20
BATCH_SIZE = 4
LR = 1e-4
IMG_SIZE = (512, 1024)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# DATASET
# ======================
class LaneDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.images = sorted(img_dir.glob("*.jpg"))
        self.mask_dir = mask_dir

        self.tf_img = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])

        self.tf_mask = transforms.Compose([
            transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.tf_img(img)
        mask = self.tf_mask(mask)
        mask = (mask > 0).float()  # binary

        return img, mask


# ======================
# TRAIN LOOP
# ======================
def train():
    train_ds = LaneDataset(IMG_DIR / "train", MASK_DIR / "train")
    val_ds = LaneDataset(IMG_DIR / "val", MASK_DIR / "val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"[INFO] Training samples: {len(train_ds)}")
    print(f"[INFO] Validation samples: {len(val_ds)}")
    print(f"[INFO] Device: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, masks in loop:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"[TRAIN] Epoch {epoch} | Loss: {avg_loss:.4f}")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"[SAVED] {MODEL_OUT}")


if __name__ == "__main__":
    train()
