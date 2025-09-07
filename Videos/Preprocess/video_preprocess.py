import cv2
import os

# Parámetros configurables
TARGET_RESOLUTION = (640, 360)  # ancho, alto
TARGET_FPS = 15                 # FPS reducido para salida

def preprocess_video(input_path, output_path, resolution=TARGET_RESOLUTION, target_fps=TARGET_FPS):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: no se pudo abrir {input_path}")
        return

    # FPS original y cálculo de skip_rate
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    skip_rate = max(int(original_fps // target_fps), 1)

    width, height = resolution
    # Usar codec XVID y mantener FPS original
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Procesando '{input_path}'")
    print(f"   Resolución: {width}x{height} | FPS original: {original_fps:.1f} → salida: {target_fps}")
    print(f"   Total frames en video: {total_frames}")

    frame_idx, saved_idx = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Saltar frames para reducir FPS
        if frame_idx % skip_rate != 0:
            frame_idx += 1
            continue

        # Redimensionar frame
        frame = cv2.resize(frame, resolution)

        # Guardar frame procesado
        out.write(frame)
        saved_idx += 1
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Guardado en '{output_path}' ({saved_idx} frames procesados)")

if __name__ == "__main__":
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    INPUT_VIDEO = os.path.join(BASE_PATH, "training-videos", "IMG_8217.MOV")
    OUTPUT_VIDEO = os.path.join(BASE_PATH, "processed-videos", "processed3.mp4")

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

    preprocess_video(INPUT_VIDEO, OUTPUT_VIDEO)
