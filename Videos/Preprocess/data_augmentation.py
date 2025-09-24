import cv2
import numpy as np
import os

# ==========================
# VARIABLES
# ==========================

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VIDEO_PATH = os.path.join(BASE_PATH, "processed-videos", "proc1_15fps.mp4")
OUTPUT_DIR = os.path.join(BASE_PATH, "augmented-videos")

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Valores para augmentation
BRILLO_DELTA = 40
CONTRASTE_DELTA = 1.5
HORIZONTAL_PIXELS = 47
VERTICAL_PIXELS = 26
SCALE_OUT = 0.8
SCALE_DOWN = 0.85 # alejar (para crear margen antes de shift)
SCALE_STRETCH = 0.83 # alejar para estirar
STRETCH = 1.2  # estirar

# ==========================
# FUNCIONES DE AUGMENTACIÓN
# ==========================
def ajustar_brillo(img, delta):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=delta)

def ajustar_contraste(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def escalar_centro(img, scale):
    """Escalado uniforme alrededor del centro"""
    filas, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, filas / 2), 0, scale)
    return cv2.warpAffine(img, M, (cols, filas))

def desplazar_centro(img, dx, dy, pre_scale=1.0):
    """Shift con margen gracias al pre_scale < 1"""
    reducido = escalar_centro(img, pre_scale)
    filas, cols = reducido.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(reducido, M, (cols, filas))

def estirar_centro(img, scale_x, scale_y, pre_scale=1.0):
    """Stretch alrededor del centro con margen previo"""
    reducido = escalar_centro(img, pre_scale)
    filas, cols = reducido.shape[:2]
    cx, cy = cols / 2, filas / 2
    M = np.float32([
        [scale_x, 0, cx - scale_x * cx],
        [0, scale_y, cy - scale_y * cy]
    ])
    return cv2.warpAffine(reducido, M, (cols, filas))

# ==========================
# MAIN
# ==========================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ No se pudo abrir el video.")
    exit()

# Obtener propiedades del video original
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Procesando {total_frames} frames — {width}x{height} @ {fps:.2f}fps")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# Definir variantes como funciones con nombres
variantes = {
    "original": lambda f: f,
    "mas_brillo": lambda f: ajustar_brillo(f, BRILLO_DELTA),
    "menos_brillo": lambda f: ajustar_brillo(f, -BRILLO_DELTA),
    "mas_cont": lambda f: ajustar_contraste(f, CONTRASTE_DELTA),
    "menos_cont": lambda f: ajustar_contraste(f, CONTRASTE_DELTA / 2),
    "shift_der": lambda f: desplazar_centro(f, HORIZONTAL_PIXELS, 0, pre_scale=SCALE_DOWN),
    "shift_izq": lambda f: desplazar_centro(f, -HORIZONTAL_PIXELS, 0, pre_scale=SCALE_DOWN),
    "shift_arriba": lambda f: desplazar_centro(f, 0, -VERTICAL_PIXELS, pre_scale=SCALE_DOWN),
    "shift_abajo": lambda f: desplazar_centro(f, 0, VERTICAL_PIXELS, pre_scale=SCALE_DOWN),
    "est_hori": lambda f: estirar_centro(f, STRETCH, 1.0, pre_scale=SCALE_STRETCH),
    "est_vert": lambda f: estirar_centro(f, 1.0, STRETCH, pre_scale=SCALE_STRETCH),
    "alejado": lambda f: escalar_centro(f, SCALE_OUT),
}
# Crear un VideoWriter por cada variante
writers = {
    nombre: cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_{nombre}.mp4"),
        fourcc,
        fps,
        (width, height)
    )
    for nombre in variantes
}

# Procesar frame a frame
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Generar y guardar cada variante
    for nombre, func in variantes.items():
        mod = func(frame)
        writers[nombre].write(mod)

    if frame_idx % 50 == 0 or frame_idx == total_frames:
        print(f"Procesados {frame_idx}/{total_frames} frames")

cap.release()
for w in writers.values():
    w.release()

print(f"✅ Proceso terminado. Videos guardados en: {OUTPUT_DIR}")