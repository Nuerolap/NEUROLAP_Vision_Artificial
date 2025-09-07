import cv2
import os
import mediapipe as mp

# Inicializar Mediapipe Hands con umbrales más bajos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.4,   # más sensible
                       min_tracking_confidence=0.4)    # más tolerante
mp_drawing = mp.solutions.drawing_utils

# Paths
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_video_path = os.path.join(BASE_PATH, "processed-videos", "processed3_10fps.mp4")
output_video_path = os.path.join(BASE_PATH, "processed-videos", "mediapipe3_10fps.mp4")

# Verificar entrada
if not os.path.exists(input_video_path):
    raise FileNotFoundError(f"No se encontró el video: {input_video_path}")

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise Exception(f"No se pudo abrir el video: {input_video_path}")

# Propiedades del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps is None:
    fps = 30  # valor seguro por defecto

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Procesando {frame_count} frames...")

# Guardar los últimos keypoints válidos para suavizado
last_hand_landmarks = None

current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Actualizar último frame válido
        last_hand_landmarks = results.multi_hand_landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # Usar los últimos keypoints válidos si existen (suavizado)
        if last_hand_landmarks:
            for hand_landmarks in last_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    out.write(frame)

    current_frame += 1
    if current_frame % 30 == 0:
        print(f"Procesados {current_frame}/{frame_count} frames...")

# Liberar recursos
cap.release()
out.release()
hands.close()
cv2.destroyAllWindows()
print(f"Video procesado guardado en {output_video_path}")
