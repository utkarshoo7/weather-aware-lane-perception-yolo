import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image

import torchvision.transforms as transforms

from utils.common import merge_config, get_model


# ---------------------------
# Post-processing
# ---------------------------
def pred2coords(pred, row_anchor, col_anchor,
                local_width=1,
                original_image_width=1640,
                original_image_height=590):

    num_grid_row = pred['loc_row'].shape[1]
    num_cls_row = pred['loc_row'].shape[2]
    num_grid_col = pred['loc_col'].shape[1]
    num_cls_col = pred['loc_col'].shape[2]

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        lane = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    inds = list(range(
                        max(0, max_indices_row[0, k, i] - local_width),
                        min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1
                    ))
                    inds = torch.tensor(inds)

                    out = (pred['loc_row'][0, inds, k, i].softmax(0) * inds.float()).sum() + 0.5
                    x = int(out / (num_grid_row - 1) * original_image_width)
                    y = int(row_anchor[k] * original_image_height)
                    lane.append((x, y))
            coords.append(lane)

    for i in col_lane_idx:
        lane = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    inds = list(range(
                        max(0, max_indices_col[0, k, i] - local_width),
                        min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1
                    ))
                    inds = torch.tensor(inds)

                    out = (pred['loc_col'][0, inds, k, i].softmax(0) * inds.float()).sum() + 0.5
                    y = int(out / (num_grid_col - 1) * original_image_height)
                    x = int(col_anchor[k] * original_image_width)
                    lane.append((x, y))
            coords.append(lane)

    return coords


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to input image")
    parser.add_argument("--weight", required=True, help="Path to culane_18.pth")
    parser.add_argument("--dataset", default="CULane")
    args = parser.parse_args()

    # Load config
    _, cfg = merge_config()
    cfg.dataset = args.dataset
    cfg.batch_size = 1

    # 🔒 HARD FIX for CULane pretrained model
    cfg.train_height = 288
    cfg.train_width = 800
    cfg.crop_ratio = 1.0
    cfg.test_model = args.weight

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading model...")
    net = get_model(cfg).to(device)
    net.eval()

    state = torch.load(cfg.test_model, map_location="cpu")["model"]
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(clean_state, strict=False)

    # Image transform (PIL ONLY)
    transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # Load image
    img_bgr = cv2.imread(args.img)
    assert img_bgr is not None, f"Image not found: {args.img}"

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    print("[INFO] Running inference...")
    with torch.no_grad():
        pred = net(input_tensor)

    coords = pred2coords(
        pred,
        cfg.row_anchor,
        cfg.col_anchor,
        original_image_width=w,
        original_image_height=h
    )

    for lane in coords:
        for x, y in lane:
            cv2.circle(img_bgr, (x, y), 4, (0, 255, 0), -1)

    out_path = "ufld_result.jpg"
    cv2.imwrite(out_path, img_bgr)
    
    print(f"[OK] Result saved to {out_path}")
