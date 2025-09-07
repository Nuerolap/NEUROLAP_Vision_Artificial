import os
import cv2

# Obtener la carpeta base del proyecto (un nivel arriba del script actual)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Rutas relativas
VIDEOS_FOLDER = os.path.join(BASE_PATH, "training-videos")
FRAMES_FOLDER = os.path.join(BASE_PATH, "video-frames")

def split_video(output_folder, video_path):
    # Crear la carpeta de salida para los frames
    os.makedirs(output_folder, exist_ok=True)

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    # Obtener FPS del video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing '{video_path}' - FPS: {video_fps}")

    # Calcular el intervalo de frames (para guardar aproximadamente 25 frames por segundo)
    frame_interval = int(round(video_fps / 30)) if video_fps > 25 else 1

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video

        # Guardar solo cada n frames, dependiendo del intervalo
        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_folder}'")


def main():
    # Crear la carpeta para guardar frames si no existe
    os.makedirs(FRAMES_FOLDER, exist_ok=True)

    # Recorrer los archivos de la carpeta de videos
    files = os.listdir(VIDEOS_FOLDER)
    for file in files:
        if file.endswith((".mp4", ".avi", ".mov", ".mkv")):  # Filtrar solo archivos de video
            video_name = os.path.splitext(file)[0]  # Nombre del video sin extensión
            output_folder = os.path.join(FRAMES_FOLDER, video_name)  # Carpeta de salida
            video_path = os.path.join(VIDEOS_FOLDER, file)  # Ruta completa del video

            # Dividir el video en frames
            split_video(output_folder, video_path)


if __name__ == "__main__":
    main()
