# external/UFLD/utils/common.py
# INFERENCE-ONLY SAFE VERSION (NO DALI, NO TRAINING CODE)

import torch
import argparse
from addict import Dict

# -------------------------
# Weight initialization
# -------------------------
def initialize_weights(*models):
    for model in models:
        if model is None:
            continue
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

# -------------------------
# Minimal config loader
# -------------------------
def merge_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='CULane')
    args = parser.parse_args()

    cfg = Dict()
    cfg.dataset = args.dataset
    cfg.backbone = '18'
    cfg.num_lanes = 4
    cfg.train_height = 320
    cfg.train_width = 800
    cfg.crop_ratio = 0.8
    cfg.use_aux = False
    cfg.fc_norm = False
    cfg.test_model = args.weight

    if cfg.dataset == 'CULane':
        cfg.num_row = 18
        cfg.num_col = 200
        cfg.num_cell_row = 200
        cfg.num_cell_col = 100
        cfg.row_anchor = [i / 18 for i in range(18)]
        cfg.col_anchor = [i / 200 for i in range(200)]
    else:
        raise NotImplementedError

    return args, cfg

# -------------------------
# Model loader
# -------------------------
def get_model(cfg):
    if cfg.dataset == 'CULane':
        from model.model_culane import parsingNet
    else:
        raise NotImplementedError

    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
        fc_norm=cfg.fc_norm
    )

    return net.cuda().eval()
