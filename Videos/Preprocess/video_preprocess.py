import cv2
import os

# =========================
# Parámetros configurables
# =========================
TARGET_FPS = 10
TARGET_RESOLUTION = (640, 360)   # (ancho, alto) → 360p
CODEC = "mp4v"                   # "mp4v" → mp4, "XVID" → avi

# =========================
# Función principal
# =========================
def convert_video(input_path, output_path, target_fps=TARGET_FPS,
                  resolution=TARGET_RESOLUTION, codec=CODEC):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir el video: {input_path}")

    # Propiedades del video original
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_in  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0  # fallback si OpenCV no detecta FPS

    print(f"📹 FPS original: {orig_fps:.2f} | Objetivo: {target_fps} fps")
    print(f"   Resolución entrada: {width_in}x{height_in} → salida: {resolution[0]}x{resolution[1]}")
    print(f"   Total de frames: {total_frames}")

    # Configurar salida
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, float(target_fps), resolution)
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"❌ No se pudo crear el archivo de salida: {output_path}")

    # Algoritmo de resampleo temporal
    ratio = float(target_fps) / float(orig_fps)
    accum = 0.0
    in_idx, out_count = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        accum += ratio
        while accum >= 1.0:
            # Redimensionar si es necesario
            if (frame.shape[1], frame.shape[0]) != resolution:
                frame_resized = cv2.resize(frame, resolution)
            else:
                frame_resized = frame

            out.write(frame_resized)
            out_count += 1
            accum -= 1.0

        in_idx += 1

    cap.release()
    out.release()

    # Reporte final
    dur_in  = total_frames / orig_fps
    dur_out = out_count / target_fps
    print(f"✅ Guardado en: {output_path}")
    print(f"   Frames escritos: {out_count}")
    print(f"   Duración original ≈ {dur_in:.2f}s | salida ≈ {dur_out:.2f}s")


# =========================
# Ejecución
# =========================
if __name__ == "__main__":
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    INPUT_VIDEO  = os.path.join(BASE_PATH, "training-videos", "IMG_8309.MOV")
    OUTPUT_VIDEO = os.path.join(BASE_PATH, "processed-videos", "processed3_10fps.mp4")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    convert_video(INPUT_VIDEO, OUTPUT_VIDEO)
