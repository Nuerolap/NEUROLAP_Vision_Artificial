import cv2
import os
import numpy as np

# =========================
# Parámetros configurables
# =========================
TARGET_FPS = 15
TARGET_RESOLUTION = (640, 360)   # (ancho, alto) → 360p letterboxed
CODEC = "mp4v"                   # "mp4v" → mp4, "XVID" → avi
OVERWRITE = False                # True para reescribir salidas existentes

def resize_letterbox(frame, target=(640, 360)):
    tw, th = target
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=frame.dtype)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def convert_video(input_path, output_path, target_fps=TARGET_FPS,
                  resolution=TARGET_RESOLUTION, codec=CODEC):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir el video: {input_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_in  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not orig_fps or orig_fps <= 0:
        orig_fps = 30.0  # fallback

    print(f"📹 {os.path.basename(input_path)} | FPS in: {orig_fps:.2f} → out: {target_fps}")
    print(f"   Resolución in: {width_in}x{height_in} → out: {resolution[0]}x{resolution[1]}")
    print(f"   Total frames in: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, float(target_fps), resolution)
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"❌ No se pudo crear el archivo de salida: {output_path}")

    ratio = float(target_fps) / float(orig_fps)
    accum = 0.0
    out_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        accum += ratio
        while accum >= 1.0:
            frame_resized = resize_letterbox(frame, resolution)
            out.write(frame_resized)
            out_count += 1
            accum -= 1.0

    cap.release()
    out.release()

    dur_in  = total_frames / orig_fps
    dur_out = out_count / target_fps
    print(f"✅ Guardado: {output_path}")
    print(f"   Frames escritos: {out_count} | Dur in ≈ {dur_in:.2f}s | Dur out ≈ {dur_out:.2f}s\n")

if __name__ == "__main__":
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_folder  = os.path.join(BASE_PATH, "training-videos")
    output_folder = os.path.join(BASE_PATH, "processed-videos")
    os.makedirs(output_folder, exist_ok=True)

    videos = [f for f in os.listdir(input_folder)
              if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

    if not videos:
        print("⚠️ No se encontraron videos en training-videos/")
    for file in videos:
        in_path  = os.path.join(input_folder, file)
        out_name = f"{os.path.splitext(file)[0]}_15fps.mp4"
        out_path = os.path.join(output_folder, out_name)

        if (not OVERWRITE) and os.path.exists(out_path):
            print(f"⏭️  Saltando (ya existe): {out_name}")
            continue

        print(f"➡️ Procesando: {file}")
        convert_video(in_path, out_path)
