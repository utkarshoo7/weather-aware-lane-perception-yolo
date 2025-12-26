import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("datasets/weather")
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 6
LR = 1e-5  # VERY IMPORTANT (low LR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# -----------------------
# TRANSFORMS
# -----------------------
train_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------
# DATA
# -----------------------
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

NUM_CLASSES = len(train_ds.classes)

# -----------------------
# MODEL
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Load previous head-trained weights
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("models/weather/weather_resnet18.pth", map_location=DEVICE))

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze LAST block + classifier
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# -----------------------
# TRAIN
# -----------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100.0 * correct / total
    print(f"[FT] Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss:.4f} Val Acc: {val_acc:.2f}%")

# -----------------------
# SAVE
# -----------------------
torch.save(model.state_dict(), "models/weather/weather_resnet18_ft.pth")
print("Saved models/weather/weather_resnet18_ft.pth")
