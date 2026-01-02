import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
UFLD_ROOT = os.path.join(PROJECT_ROOT, "external", "UFLD")

if UFLD_ROOT not in sys.path:
    sys.path.insert(0, UFLD_ROOT)

import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from addict import Dict



from external.UFLD.utils.common import get_model


class UFLDLaneDetector:
    def __init__(self, weight_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        cfg = Dict()
        cfg.dataset = "CULane"
        cfg.backbone = "18"
        cfg.num_lanes = 4

        cfg.train_height = 288
        cfg.train_width = 800
        cfg.crop_ratio = 1.0

        cfg.num_row = 18
        cfg.num_col = 200
        cfg.num_cell_row = 200
        cfg.num_cell_col = 100

        cfg.row_anchor = [i / cfg.num_row for i in range(cfg.num_row)]
        cfg.col_anchor = [i / cfg.num_col for i in range(cfg.num_col)]

        cfg.use_aux = False
        cfg.fc_norm = False
        cfg.test_model = weight_path

        print("[INFO] Loading UFLD model...")
        self.net = get_model(cfg).to(self.device)

        state = torch.load(weight_path, map_location="cpu")["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.net.load_state_dict(state, strict=False)
        self.net.eval()

        self.cfg = cfg

        self.transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225)
            )
        ])

    def detect(self, image_bgr):
        h, w = image_bgr.shape[:2]

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        inp = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.net(inp)

        return self._pred_to_coords(pred, w, h)

    def _pred_to_coords(self, pred, img_w, img_h):
        coords = []
        loc_row = pred["loc_row"]
        exist_row = pred["exist_row"].argmax(1)
        num_grid = loc_row.shape[1]

        for lane_id in range(loc_row.shape[-1]):
            lane = []
            for r in range(loc_row.shape[2]):
                if exist_row[0, r, lane_id]:
                    idx = loc_row[0, :, r, lane_id].argmax()
                    x = int(idx / (num_grid - 1) * img_w)
                    y = int(self.cfg.row_anchor[r] * img_h)
                    lane.append((x, y))
            if lane:
                coords.append(lane)

        return coords
