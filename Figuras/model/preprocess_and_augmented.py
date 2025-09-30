import os
import cv2
import numpy as np
from PIL import Image
import pillow_heif  # Para HEIC en Windows/Mac

# Registrar soporte HEIC en Pillow
pillow_heif.register_heif_opener()

# =============================
# Preprocesamiento calibrado
# =============================
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")

    # Rotar 90 grados
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    # Filtro bilateral
    gray_filtered = cv2.bilateralFilter(gray_rotated, 9, 75, 75)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_filtered)

    # Umbral adaptativo
    binary = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 10
    )

    # Closing morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Invertir (fondo blanco, trazos negros)
    final = cv2.bitwise_not(morph)

    # Resize a 512x512 y convertir a RGB
    resized = cv2.resize(final, (512, 512))
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    return rgb

# =============================
# Data Augmentation (sin ruido)
# =============================
def add_gaussian_blur(image):
    return cv2.GaussianBlur(image, (3, 3), 0)

def add_erosion(image):
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def add_clahe_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def adjust_brightness(image, factor=1.2):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,2] = np.clip(hsv[...,2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

# =============================
# Main
# =============================
def process_and_augment(input_dir="data", output_dir="processed_data"):
    classes = ["Verde", "Rojo", "Amarillo"]
    os.makedirs(output_dir, exist_ok=True)

    for cls in classes:
        input_path = os.path.join(input_dir, cls)

        # Carpeta base preprocesada
        preproc_path = os.path.join(output_dir, cls)
        os.makedirs(preproc_path, exist_ok=True)

        # Carpeta para augmentations
        aug_path = os.path.join(output_dir, cls + "_Augmented")
        os.makedirs(aug_path, exist_ok=True)

        for fname in os.listdir(input_path):
            fpath = os.path.join(input_path, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                base = preprocess_image(fpath)

                # Guardar preprocesada
                out_name = os.path.splitext(fname)[0] + "_preproc.png"
                cv2.imwrite(os.path.join(preproc_path, out_name), base)
                cv2.imwrite(os.path.join(aug_path, out_name), base)  # también en augmented

                # Augmentations
                aug_blur = add_gaussian_blur(base)
                aug_erode = add_erosion(base)
                aug_contrast = add_clahe_contrast(base)
                aug_bright = adjust_brightness(base, factor=1.2)

                cv2.imwrite(os.path.join(aug_path, os.path.splitext(fname)[0] + "_blur.png"), aug_blur)
                cv2.imwrite(os.path.join(aug_path, os.path.splitext(fname)[0] + "_erode.png"), aug_erode)
                cv2.imwrite(os.path.join(aug_path, os.path.splitext(fname)[0] + "_contrast.png"), aug_contrast)
                cv2.imwrite(os.path.join(aug_path, os.path.splitext(fname)[0] + "_bright.png"), aug_bright)

            except Exception as e:
                print(f"Error procesando {fpath}: {e}")

if __name__ == "__main__":
    process_and_augment()
    print("✅ Preprocesado completado")
    print("📂 Carpeta con preprocesadas en processed_data/CLASE")
    print("📂 Carpeta con aumentadas en processed_data/CLASE_Augmented (incluye la preprocesada)")
