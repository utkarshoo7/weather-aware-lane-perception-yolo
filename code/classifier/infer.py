import torch
import torchvision.transforms as T
from PIL import Image

MODEL_PATH = "models/weather/weather_resnet18.pth"

_classes = [
    "clear",
    "overcast",
    "partly_cloudy",
    "rainy",
    "snowy"
]

_device = "cuda" if torch.cuda.is_available() else "cpu"

_model = torch.hub.load(
    "pytorch/vision",
    "resnet18",
    pretrained=False
)
_model.fc = torch.nn.Linear(_model.fc.in_features, len(_classes))

_ckpt = torch.load(MODEL_PATH, map_location="cpu")
_model.load_state_dict(_ckpt, strict=False)
_model.to(_device).eval()

_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


def predict_weather(image_path):
    img = Image.open(image_path).convert("RGB")
    x = _tf(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1)

    conf, idx = probs.max(1)
    return _classes[idx.item()], conf.item()
