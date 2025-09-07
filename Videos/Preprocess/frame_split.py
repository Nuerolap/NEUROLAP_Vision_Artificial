import os
import cv2
from tqdm import tqdm

# Obtener la carpeta base del proyecto (un nivel arriba del script actual)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Rutas relativas
VIDEOS_FOLDER = os.path.join(BASE_PATH, "training-videos")
FRAMES_FOLDER = os.path.join(BASE_PATH, "video-frames")

# Máximo de FPS a extraer (para evitar miles de frames innecesarios en webcams)
MAX_FPS_OUTPUT = 20


def split_video(output_folder, video_path):
    # Si ya existe la carpeta con frames, saltar para no reprocesar
    if os.path.exists(output_folder) and os.listdir(output_folder):
        print(f"Skipping '{video_path}' (frames already extracted).")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}.")
        return

    # Obtener FPS y total de frames del video
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25  # fallback si webcam no guarda FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ajustar intervalo de extracción
    effective_fps = min(video_fps, MAX_FPS_OUTPUT)
    frame_interval = max(int(round(video_fps / effective_fps)), 1)

    print(f"\n📹 Processing '{video_path}'")
    print(f"   Original FPS: {video_fps:.2f} | Saving ~{effective_fps} fps")
    print(f"   Total frames in video: {total_frames}")

    frame_count = 0
    saved_count = 0

    # Barra de progreso
    with tqdm(total=total_frames, desc="Extracting frames", unit="f") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Fin del video

            # Guardar solo cada n frames
            if frame_count % frame_interval == 0:
                frame_name = f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(os.path.join(output_folder, frame_name), frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"   ✅ Saved {saved_count} frames to '{output_folder}'\n")


def main():
    os.makedirs(FRAMES_FOLDER, exist_ok=True)

    # Procesar cada archivo de video
    files = os.listdir(VIDEOS_FOLDER)
    for file in files:
        if file.endswith((".mp4", ".avi", ".MOV", ".mkv")):
            video_name = os.path.splitext(file)[0]
            output_folder = os.path.join(FRAMES_FOLDER, video_name)
            video_path = os.path.join(VIDEOS_FOLDER, file)

            split_video(output_folder, video_path)


if __name__ == "__main__":
    main()
