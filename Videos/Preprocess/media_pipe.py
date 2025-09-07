import cv2
import os
import mediapipe as mp

# Inicializar Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Abrir el video de entrada
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_video_path = os.path.join(BASE_PATH, "training-videos", "IMG_8217.MOV")
cap = cv2.VideoCapture(input_video_path)

# Obtener propiedades del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Crear video de salida
output_video_path = os.path.join(BASE_PATH, "processed-videos", "mediapipe0.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Dibujar los keypoints si se detecta alguna mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Escribir el frame en el video de salida
    out.write(frame)

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video procesado guardado en {output_video_path}")
