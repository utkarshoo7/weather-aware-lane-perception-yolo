import cv2
import torch
import numpy as np
from pathlib import Path

from code.unet.model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "models/unet/unet_lane_v1.pth"

IMAGE_PATH = "datasets/tusimple_unet/images/train/20.png"
OUTPUT_PATH = "analysis/unet_debug_overlay.png"


def main():
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    img = cv2.imread(IMAGE_PATH)
    assert img is not None, "Failed to load image"

    img_resized = cv2.resize(img, (256, 256))
    img_tensor = (
        torch.from_numpy(img_resized)
        .permute(2, 0, 1)
        .float() / 255.0
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mask = model(img_tensor)[0, 0].cpu().numpy()

    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    overlay = img.copy()
    overlay[mask == 255] = [0, 0, 255]  # red lanes

    cv2.imwrite(OUTPUT_PATH, overlay)
    print(f"Saved overlay to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
