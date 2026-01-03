from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18

# =========================
# CONFIG
# =========================
MODEL_PATH = Path("models/weather/weather_resnet18_ft.pth")
DEVICE = "cpu"

# MUST MATCH TRAINING
RAW_CLASSES = ["clear", "rainy", "snowy", "overcast", "night"]

# =========================
# BUILD MODEL
# =========================
model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(RAW_CLASSES))

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# =========================
# TRANSFORMS
# =========================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# =========================
# API
# =========================
def predict_weather(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    conf, idx = probs.max(dim=1)
    raw_label = RAW_CLASSES[idx.item()]

    # CANONICAL MAPPING (LOCKED)
    if raw_label == "overcast":
        final_label = "clear"
    elif raw_label == "night":
        final_label = "clear"
    else:
        final_label = raw_label

    return {
        "label": final_label,
        "raw_label": raw_label,
        "confidence": float(conf.item()),
        "logits": logits.squeeze().tolist(),
    }
