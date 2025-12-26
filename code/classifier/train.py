import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from code.classifier.model import WeatherCNN


# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("datasets/weather")
MODEL_DIR = Path("models/weather")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("SCRIPT STARTED")
    print("Using device:", DEVICE)

    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tfms)

    print("Classes:", train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = WeatherCNN(len(train_ds.classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
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

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {train_loss:.4f}  Val Acc: {acc:.2f}%")

    torch.save(model.state_dict(), MODEL_DIR / "weather_cnn.pth")
    print("Model saved to models/weather/weather_cnn.pth")


if __name__ == "__main__":
    main()
