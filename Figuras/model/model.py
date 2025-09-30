import os
import shutil
import random
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms, datasets

# =============================
# 1. Configuración
# =============================
data_dir = "data"   # carpeta original con VERDE, ROJO, AMARILLO
train_ratio = 0.8   # 80% train / 20% val
batch_size = 8
epochs = 10

# =============================
# 2. Transformaciones
# =============================
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =============================
# 3. Cargar dataset y dividirlo
# =============================
# Dataset completo
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# División train/val
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Clases detectadas:", full_dataset.classes)
print(f"Total imágenes: {len(full_dataset)} → Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# =============================
# 4. Modelo DenseNet para clasificación
# =============================
def get_model(num_classes=3):
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)  # 3 clases
    )
    return model

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes=3).to(device)

# =============================
# 5. Entrenamiento
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Acc: {acc:.3f}")

    return model

# =============================
# 6. Ejecutar entrenamiento
# =============================
model = train_model(model, train_loader, val_loader, epochs=epochs)
torch.save(model.state_dict(), "rcft_classifier.pth")
print("Modelo guardado como rcft_classifier.pth")