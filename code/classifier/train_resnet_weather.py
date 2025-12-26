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
EPOCHS = 8
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# -----------------------
# TRANSFORMS (IMPORTANT)
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
# DATASETS
# -----------------------
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfms)

print("Classes:", train_ds.classes)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

NUM_CLASSES = len(train_ds.classes)

# -----------------------
# MODEL (TRANSFER LEARNING)
# -----------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# -----------------------
# TRAIN LOOP
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
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss:.4f} Val Acc: {val_acc:.2f}%")

# -----------------------
# SAVE
# -----------------------
os.makedirs("models/weather", exist_ok=True)
torch.save(model.state_dict(), "models/weather/weather_resnet18.pth")
print("Saved models/weather/weather_resnet18.pth")
