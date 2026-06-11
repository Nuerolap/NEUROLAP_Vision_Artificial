import os
import json
import cv2

# Función para obtener los metadatos de un video
def get_video_metadata(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"No se pudo abrir el archivo de video: {video_path}")
        return None

    metadata = {
        "fps": video.get(cv2.CAP_PROP_FPS),
        "width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    video.release()
    return metadata

# Función para extraer el nombre base de un archivo de video (antes del número)
def extract_gloss_name(file_name):
    name_parts = file_name.split("_")
    if len(name_parts) > 1:
        return "_".join(name_parts[:-1])  # Une todas las partes antes del número
    return os.path.splitext(file_name)[0]  # Devuelve el nombre sin extensión si no hay guión bajo

# Función para crear el JSON estructurado
def create_training_json(base_path, videos_folder="training-videos", output_json="training.json"):
    # Construir rutas relativas
    videos_path = os.path.join(base_path, videos_folder)
    output_json_path = os.path.join(os.path.dirname(__file__), output_json)  # Guardar el JSON en el mismo directorio del script
    data = {}

    # Recorrer todas las subcarpetas y archivos
    for root, dirs, files in os.walk(videos_path):
        for file_name in files:
            if file_name.endswith((".mp4", ".avi", ".mov", ".mkv")):  # Extensiones de video
                file_path = os.path.join(root, file_name)
                metadata = get_video_metadata(file_path)

                if metadata:
                    # Obtener el nombre base para gloss
                    gloss_name = extract_gloss_name(file_name)

                    # Si no existe el gloss en el diccionario, inicializarlo
                    if gloss_name not in data:
                        data[gloss_name] = {
                            "gloss": gloss_name,
                            "instances": []
                        }

                    # Crear una instancia para el video
                    video_instance = {
                        "bbox": [
                            int(metadata["width"] * 1),  # Coordenada x inicial (central)
                            int(metadata["height"] * 0.20),  # Coordenada y inicial (central)
                            int(metadata["width"] * 1),  # Coordenada x final
                            int(metadata["height"] * 0.80)   # Coordenada y final
                        ],
                        "fps": metadata["fps"],
                        "frame_end": -1,  # Valor predeterminado
                        "frame_start": 1,  # Valor predeterminado
                        "split": "train",  # Valor predeterminado
                        "path": os.path.relpath(file_path, base_path).replace("\\", "/"),  # Ruta relativa al proyecto
                        "video_id": os.path.splitext(file_name)[0]  # Nombre sin extensión
                    }

                    # Agregar la instancia al gloss correspondiente
                    data[gloss_name]["instances"].append(video_instance)

    # Asignar los splits a cada gloss
    for gloss_name, gloss_data in data.items():
        instances = gloss_data["instances"]
        num_instances = len(instances)

        # Los últimos 4 se asignan a 'test'
        for instance in instances[-7:]:
            instance["split"] = "test"

        # Los penúltimos 4 se asignan a 'val'
        for instance in instances[-14:-7]:
            instance["split"] = "val"

        # El resto de los videos son para 'train'
        for instance in instances[:-7]:
            instance["split"] = "train"

    # Convertir el diccionario en una lista
    output_data = list(data.values())

    # Guardar en un archivo JSON
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(output_data, json_file, indent=4, ensure_ascii=False)

    print(f"Archivo JSON generado: {output_json_path}")

# Obtener la ruta base del proyecto
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Ir un nivel hacia arriba

# Generar el archivo JSON
create_training_json(base_path)
