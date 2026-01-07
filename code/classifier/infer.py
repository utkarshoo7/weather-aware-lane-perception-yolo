"""
Weather Classification Inference Module
---------------------------------------
Performs single-image weather classification using a fine-tuned ResNet-18.
This module is intentionally deterministic and CPU-safe for reproducibility.
"""

from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = Path("models/weather/weather_resnet18_ft.pth")

# Device selection (locked to CPU for reproducibility)
DEVICE = torch.device("cpu")

# MUST match training order exactly
RAW_CLASSES = ["clear", "rainy", "snowy", "overcast", "night"]


# =========================
# MODEL INITIALIZATION
# =========================

_model = None


def _load_model() -> torch.nn.Module:
    """
    Loads the fine-tuned ResNet-18 model.
    This function is called once and cached.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Weather model not found: {MODEL_PATH}")

    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(RAW_CLASSES))

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def _get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        _model = _load_model()
    return _model


# =========================
# PREPROCESSING
# =========================

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================
# CANONICAL MAPPING
# =========================

def _canonicalize_label(raw_label: str) -> str:
    """
    Maps raw classifier outputs into canonical weather classes.
    This ensures downstream logic remains stable.
    """
    if raw_label in ("overcast", "night"):
        return "clear"
    return raw_label


# =========================
# PUBLIC API
# =========================

def predict_weather(image_path: str) -> dict:
    """
    Predicts weather condition for a single image.

    Returns:
        {
            label: canonical weather label
            raw_label: original classifier output
            confidence: softmax confidence
        }
    """
    model = _get_model()

    img = Image.open(image_path).convert("RGB")
    x = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    conf, idx = probs.max(dim=1)
    raw_label = RAW_CLASSES[idx.item()]
    final_label = _canonicalize_label(raw_label)

    return {
        "label": final_label,
        "raw_label": raw_label,
        "confidence": float(conf.item()),
    }
