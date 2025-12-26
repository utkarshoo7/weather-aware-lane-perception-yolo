from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from code.classifier.model import WeatherCNN

# -----------------------
# CONFIG
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "weather" / "weather_cnn.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["clear", "overcast", "partly_cloudy", "rainy", "snowy"]

# -----------------------
# LOAD MODEL (FROZEN)
# -----------------------
model = WeatherCNN(num_classes=len(CLASSES))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()  # 🔒 FREEZE

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------
# INFERENCE
# -----------------------
def predict_weather(image_path):
    image_path = Path(image_path)
    assert image_path.exists(), f"Image not found: {image_path}"

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    conf, idx = torch.max(probs, dim=1)
    return CLASSES[idx.item()], conf.item()
