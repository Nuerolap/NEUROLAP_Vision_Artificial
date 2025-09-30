import os
import cv2
import numpy as np
from PIL import Image

# =============================
# Preprocesamiento (paper)
# =============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Binarización adaptativa
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )

    # Recorte con contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = gray[y:y+h, x:x+w]
    else:
        cropped = gray

    # Resize a 512x512 y convertir a RGB
    resized = cv2.resize(cropped, (512, 512))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb

# =============================
# Data Augmentation
# =============================
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.uint8)
    noisy = cv2.add(image, gauss)
    return noisy

def add_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def add_dilation(image):
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    return dilated

# =============================
# Main
# =============================
def process_and_augment(input_dir="data", output_dir="processed_data"):
    classes = ["VERDE", "ROJO", "AMARILLO"]
    os.makedirs(output_dir, exist_ok=True)

    for cls in classes:
        input_path = os.path.join(input_dir, cls)
        output_path = os.path.join(output_dir, cls)
        os.makedirs(output_path, exist_ok=True)

        for fname in os.listdir(input_path):
            fpath = os.path.join(input_path, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                base = preprocess_image(fpath)

                # Guardar original preprocesada
                out_name = os.path.splitext(fname)[0] + "_preproc.png"
                cv2.imwrite(os.path.join(output_path, out_name), base)

                # Augmentations
                aug1 = add_gaussian_noise(base)
                aug2 = add_gaussian_blur(base)
                aug3 = add_dilation(base)

                cv2.imwrite(os.path.join(output_path, os.path.splitext(fname)[0] + "_noise.png"), aug1)
                cv2.imwrite(os.path.join(output_path, os.path.splitext(fname)[0] + "_blur.png"), aug2)
                cv2.imwrite(os.path.join(output_path, os.path.splitext(fname)[0] + "_dilate.png"), aug3)

            except Exception as e:
                print(f"Error procesando {fpath}: {e}")

if __name__ == "__main__":
    process_and_augment()
    print("Preprocesado y data augmentation completados en processed_data/")