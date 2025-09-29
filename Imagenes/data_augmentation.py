import os
import random
import cv2
from PIL import Image
import pillow_heif
import numpy as np
import matplotlib.pyplot as plt

folders = ["Verde", "Amarillo", "Rojo"]

output_folder = "output_augmented"
os.makedirs(output_folder, exist_ok=True)

selected_folder = random.choice(folders)
print(f"Carpeta seleccionada: {selected_folder}")

images = os.listdir(selected_folder)
if not images:
    raise ValueError(f"No hay imágenes en la carpeta {selected_folder}")

selected_image = random.choice(images)
print(f"Imagen seleccionada: {selected_image}")

img_path = os.path.join(selected_folder, selected_image)
image_pil = Image.open(img_path).convert("RGB")
img = np.array(image_pil)

if img is None:
    raise ValueError("No se pudo leer la imagen. Verifica formato y ruta.")

# Rotaciones de la imagen original
rot90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rot180 = cv2.rotate(img, cv2.ROTATE_180)
rot270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Kernel morfológico
kernel = np.ones((3, 3), np.uint8)

# Escala de grises + binarización para dilatación
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
binary = cv2.bitwise_not(binary)

dilatacion = cv2.dilate(binary, kernel, iterations=1)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Rotaciones de dilatación
dilat_rot90 = cv2.rotate(dilatacion, cv2.ROTATE_90_CLOCKWISE)
dilat_rot180 = cv2.rotate(dilatacion, cv2.ROTATE_180)
dilat_rot270 = cv2.rotate(dilatacion, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Rotaciones de blur
blur_rot90 = cv2.rotate(blur, cv2.ROTATE_90_CLOCKWISE)
blur_rot180 = cv2.rotate(blur, cv2.ROTATE_180)
blur_rot270 = cv2.rotate(blur, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Guardado de imágenes
base_name = os.path.splitext(selected_image)[0]
cv2.imwrite(os.path.join(output_folder, f"{base_name}_rot90.jpg"), rot90)
cv2.imwrite(os.path.join(output_folder, f"{base_name}_rot180.jpg"), rot180)
cv2.imwrite(os.path.join(output_folder, f"{base_name}_rot270.jpg"), rot270)

cv2.imwrite(os.path.join(output_folder, f"{base_name}_dilatacion.jpg"), dilatacion)
cv2.imwrite(os.path.join(output_folder, f"{base_name}_dilat_rot90.jpg"), dilat_rot90)
cv2.imwrite(os.path.join(output_folder, f"{base_name}_dilat_rot180.jpg"), dilat_rot180)
cv2.imwrite(os.path.join(output_folder, f"{base_name}_dilat_rot270.jpg"), dilat_rot270)

cv2.imwrite(os.path.join(output_folder, f"{base_name}_blur.jpg"), blur)
cv2.imwrite(os.path.join(output_folder, f"{base_name}_blur_rot90.jpg"), blur_rot90)
cv2.imwrite(os.path.join(output_folder, f"{base_name}_blur_rot180.jpg"), blur_rot180)
cv2.imwrite(os.path.join(output_folder, f"{base_name}_blur_rot270.jpg"), blur_rot270)

print("Data augmentation completado. Imágenes guardadas en 'output_augmented'.")

# Lista para el plot
generated_images = [
    ("original", img_path),
    ("rot90", os.path.join(output_folder, f"{base_name}_rot90.jpg")),
    ("rot180", os.path.join(output_folder, f"{base_name}_rot180.jpg")),
    ("rot270", os.path.join(output_folder, f"{base_name}_rot270.jpg")),
    ("dilatacion", os.path.join(output_folder, f"{base_name}_dilatacion.jpg")),
    ("dilat_rot90", os.path.join(output_folder, f"{base_name}_dilat_rot90.jpg")),
    ("dilat_rot180", os.path.join(output_folder, f"{base_name}_dilat_rot180.jpg")),
    ("dilat_rot270", os.path.join(output_folder, f"{base_name}_dilat_rot270.jpg")),
    ("blur", os.path.join(output_folder, f"{base_name}_blur.jpg")),
    ("blur_rot90", os.path.join(output_folder, f"{base_name}_blur_rot90.jpg")),
    ("blur_rot180", os.path.join(output_folder, f"{base_name}_blur_rot180.jpg")),
    ("blur_rot270", os.path.join(output_folder, f"{base_name}_blur_rot270.jpg")),
]

# Plot dinámico (ajusta filas/columnas según cantidad)
cols = 4
rows = (len(generated_images) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
axes = axes.flatten()

for ax, (title, img_file) in zip(axes, generated_images):
    img_disp = cv2.imread(img_file)
    # Mantener colores correctos en RGB
    if len(img_disp.shape) == 2:  # grayscale
        ax.imshow(img_disp, cmap="gray")
    else:
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        ax.imshow(img_disp)
    ax.set_title(title)
    ax.axis("off")

for ax in axes[len(generated_images):]:
    ax.axis("off")

plt.tight_layout()
final_plot_path = os.path.join(output_folder, f"{base_name}_all_plots.jpg")
plt.savefig(final_plot_path)
plt.close()

print(f"Imagen resumen generada en: {final_plot_path}")