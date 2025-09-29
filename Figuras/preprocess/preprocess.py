import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pillow_heif  # Librería recomendada para manejar HEIC en Windows/Mac

# Registrar soporte HEIC en Pillow
pillow_heif.register_heif_opener()

# === CONFIGURACIÓN ===
folder_path = r"C:\Users\Andres\OneDrive\Documentos\Atom\Python"
input_filename = "IMG_7730.HEIC"   # Imagen HEIC original
converted_filename = "converted7730.jpg"  # Nombre del archivo convertido

heic_path = os.path.join(folder_path, input_filename)
jpg_path = os.path.join(folder_path, converted_filename)

# === CONVERSIÓN HEIC → JPG ===
def convert_heic_to_jpg(heic_file, output_file):
    image = Image.open(heic_file)  # Ahora PIL abre HEIC directamente
    image.save(output_file, "JPEG")

# Convertir solo si aún no existe
if not os.path.exists(jpg_path):
    convert_heic_to_jpg(heic_path, jpg_path)
    print("Imagen HEIC convertida a JPG.")
else:
    print("Imagen JPG ya existe. No se vuelve a convertir.")

# === CARGA DE IMAGEN ===
def load_color_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    return img

input_img = load_color_image(jpg_path)

# Escala de grises
gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

# 👉 Rotar 90 grados como primer paso
gray_rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
input_rotated = cv2.rotate(input_img, cv2.ROTATE_90_CLOCKWISE)

# Filtro bilateral (reduce ruido pero preserva bordes)
gray_filtered = cv2.bilateralFilter(gray_rotated, 9, 75, 75)

# CLAHE (mejora contraste adaptativo)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(gray_filtered)

# Umbral adaptativo
binary = cv2.adaptiveThreshold(
    clahe_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    35, 10
)

# Closing morfológico (dilatación + erosión) para reforzar líneas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

# Invertimos para que quede fondo blanco y líneas negras
final = cv2.bitwise_not(morph)

# === Mostrar resultados (6 imágenes) ===
titles = [
    'Original',
    'Original Rotada',
    'CLAHE',
    'Binarización Adaptativa',
    'Closing',
    'Resultado Final'
]
images = [input_img, input_rotated, clahe_img, binary, morph, final]

plt.figure(figsize=(14,8))
for i in range(6):
    plt.subplot(2,3,i+1)
    if i in [0,1]:  # imágenes en color
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:  # procesadas en escala de grises
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
