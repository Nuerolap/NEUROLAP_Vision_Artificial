import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def analyze_first_frame(video_path, save=True, show=False):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Error: no se pudo leer el primer frame de {video_path}")
        return

    # Convertir a otros espacios de color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Brillo y contraste
    brightness = np.mean(hsv[:, :, 2])
    contrast = np.std(hsv[:, :, 2])

    print(f"📊 Análisis del primer frame de '{video_path}'")
    print(f"   Brillo promedio (V): {brightness:.2f}")
    print(f"   Contraste (σ de V): {contrast:.2f}")

    # Graficar
    plt.figure(figsize=(15, 10))

    # Imagen original
    plt.subplot(2, 3, 1)
    plt.imshow(rgb)
    plt.title("Primer frame (RGB)")
    plt.axis("off")

    # Histogramas RGB (usar la imagen rgb)
    plt.subplot(2, 3, 2)
    for i, col in enumerate(("r","g","b")):
        hist = cv2.calcHist([rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.xlim([0,256])
    plt.title("Histograma RGB")
    plt.xlabel("Intensidad")
    plt.ylabel("Frecuencia")

    # Histogramas HSV
    plt.subplot(2, 3, 3)
    labels = ["H (Tono)", "S (Saturación)", "V (Brillo)"]
    for i, label in enumerate(labels):
        hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
        plt.plot(hist, label=label)
    plt.title("Histogramas HSV")
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.legend()

    # Distribución de H (tono) más detallada (0–180 en OpenCV)
    plt.subplot(2, 3, 4)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    plt.bar(range(180), hist_h, color="orange")
    plt.title("Distribución de Tono (H)")
    plt.xlabel("Tono (0-180)")
    plt.ylabel("Frecuencia")

    # Distribución de S
    plt.subplot(2, 3, 5)
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    plt.plot(hist_s, color="blue")
    plt.title("Distribución de Saturación (S)")
    plt.xlabel("Saturación")
    plt.ylabel("Frecuencia")

    # Mapa 2D H-S (tono vs saturación)
    plt.subplot(2, 3, 6)
    h_vals = hsv[:, :, 0].flatten()
    s_vals = hsv[:, :, 1].flatten()
    plt.hexbin(h_vals, s_vals, gridsize=50, cmap="plasma")
    plt.title("Mapa 2D: Tono vs Saturación")
    plt.xlabel("H (Tono)")
    plt.ylabel("S (Saturación)")
    plt.colorbar(label="Densidad")

    
    plt.tight_layout()
    if save:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{os.path.splitext(os.path.basename(video_path))[0]}_diagnostic.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    INPUT_VIDEO = os.path.join(BASE_PATH, "training-videos", "IMG_8309.MOV")

    analyze_first_frame(INPUT_VIDEO, save=True, show=False)
