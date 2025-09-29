import os
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# =============================
# 1. Dataset y Preprocesamiento
# =============================

class RCFTDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def preprocess_image(self, img_path):
        # Cargar y convertir a escala de grises
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Filtro de mediana
        gray = cv2.medianBlur(gray, 5)

        # Binarización adaptativa
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 8
        )

        # Detección de contornos y recorte
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped = gray[y:y+h, x:x+w]
        else:
            cropped = gray

        # Redimensionar a 512x512 y convertir a RGB
        resized = cv2.resize(cropped, (512, 512))
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    def __getitem__(self, idx):
        img = self.preprocess_image(self.image_paths[idx])
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.image_paths)


# =============================
# 2. Modelo DenseNet
# =============================

def get_model():
    model = models.densenet121(pretrained=True)  # DenseNet backbone
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1000),
        nn.ReLU(),
        nn.Linear(1000, 128),
        nn.ReLU(),
        nn.Linear(128, 1)  # salida regresión (score RCFT)
    )
    return model


# =============================
# 3. Entrenamiento
# =============================

def train_model(model, train_loader, val_loader, device, epochs=20):
    criterion = nn.SmoothL1Loss()  # Huber loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)

    best_model_wts = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validación
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
                outputs = model(imgs)
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_mae = mean_absolute_error(val_true, val_preds)
        val_r2 = r2_score(val_true, val_preds)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val MAE: {val_mae:.3f} - R2: {val_r2:.3f}")

        # Guardar mejor modelo
        if val_mae < best_val_loss:
            best_val_loss = val_mae
            best_model_wts = model.state_dict()

    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model


# =============================
# 4. Ejemplo de uso
# =============================

if __name__ == "__main__":
    # Rutas de ejemplo (cámbialas a tu dataset real)
    train_images = ["data/train/img1.png", "data/train/img2.png"]
    train_labels = [28.0, 32.0]  # puntajes RCFT (0-36)
    val_images = ["data/val/img3.png", "data/val/img4.png"]
    val_labels = [12.0, 18.0]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = RCFTDataset(train_images, train_labels, transform=transform)
    val_dataset = RCFTDataset(val_images, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model()
    model = train_model(model, train_loader, val_loader, device, epochs=20)

    torch.save(model.state_dict(), "rcft_model.pth")
    print("Modelo guardado como rcft_model.pth")