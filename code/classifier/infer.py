import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("models/weather/weather_resnet18_ft.pth")
TEMP_PATH = Path("analysis/weather_temperature.txt")

CLASSES = ["clear", "overcast", "partly_cloudy", "rainy", "snowy"]

_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

_model = None
_temperature = None


def _load_model():
    global _model, _temperature

    if _model is None:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model

    if _temperature is None:
        if TEMP_PATH.exists():
            _temperature = float(TEMP_PATH.read_text().strip())
        else:
            _temperature = 1.0


def predict_weather(image_path):
    _load_model()

    img = Image.open(image_path).convert("RGB")
    x = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(x)
        logits = logits / _temperature
        probs = F.softmax(logits, dim=1)[0]

    idx = int(torch.argmax(probs))
    return CLASSES[idx], float(probs[idx])
