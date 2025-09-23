import os
import csv
import cv2
import os
import math
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


# =========================
# CONFIGURACIÓN
# =========================

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_PATH = os.path.join(BASE_PATH, "processed-videos", "proc3_15fps.mp4")
OUTPUT_PATH = os.path.join(BASE_PATH, "processed-videos", "ema3_15fps.mp4")

# Parámetros de detección y tracking
MAX_HANDS = 2
MIN_DET_CONF = 0.15     # Confianza mínima para detección inicial
MIN_TRK_CONF = 0.4      # Confianza mínima para tracking entre frames

# Padding vertical solo para mejorar la detección (no se refleja en la salida)
DETECTION_TOP_PAD = 150  # píxeles adicionales arriba

# Suavizado de landmarks con media exponencial (EMA)
SMOOTH_ALPHA = 0.15      # 0 = sin suavizado, 0.9 = muy estable pero lento

# Tracking temporal de manos
MATCH_THRESHOLD = 0.45   # Máxima distancia (normalizada) para emparejar detecciones
MAX_MISSES = 15          # Frames permitidos sin detección antes de descartar mano


# =========================
# UTILIDADES
# =========================

def centroid_of(landmarks):
    """Calcula el centroide promedio (x,y) de una lista de landmarks normalizados."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def euclidean_distance(a, b):
    """Distancia euclidiana entre dos puntos (x,y)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_from_padded_to_original(lm, pad_top, padded_h, orig_w, orig_h):
    """
    Convierte un landmark detectado en imagen con padding superior
    a coordenadas normalizadas respecto al frame original.
    """
    px = lm.x * orig_w
    py_padded = lm.y * padded_h
    py_orig = py_padded - pad_top

    # Re-normalizar y recortar a [0,1]
    x_norm = max(0.0, min(1.0, px / orig_w))
    y_norm = max(0.0, min(1.0, py_orig / orig_h))

    return landmark_pb2.NormalizedLandmark(x=x_norm, y=y_norm, z=lm.z)


def smooth_landmark_list(prev_list, curr_list, alpha):
    """
    Aplica suavizado EMA entre dos listas de landmarks (previo y actual).
    Asume listas del mismo largo.
    """
    smoothed = []
    for p, c in zip(prev_list, curr_list):
        sx = alpha * p.x + (1 - alpha) * c.x
        sy = alpha * p.y + (1 - alpha) * c.y
        sz = alpha * p.z + (1 - alpha) * c.z
        smoothed.append(landmark_pb2.NormalizedLandmark(x=sx, y=sy, z=sz))
    return smoothed


# =========================
# INICIALIZACIÓN
# =========================

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"No se encontró el video de entrada: {INPUT_PATH}")

cap = cv2.VideoCapture(INPUT_PATH)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir el video: {INPUT_PATH}")

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_w, frame_h))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"No se pudo crear el archivo de salida: {OUTPUT_PATH}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Lista de manos activas
# Cada slot = {'landmarks': [NormalizedLandmark], 'centroid': (x,y), 'missed': int}
slots = []

print(f"Procesando {total_frames} frames — {frame_w}x{frame_h} @ {fps:.2f}fps")


# =========================
# PROCESAMIENTO PRINCIPAL
# =========================

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRK_CONF
) as hands:

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Convertir a RGB para MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Añadir padding superior solo para la detección
            pad_top = DETECTION_TOP_PAD if DETECTION_TOP_PAD > 0 else 0
            if pad_top > 0:
                detect_img = cv2.copyMakeBorder(
                    rgb, pad_top, 0, 0, 0,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )
                padded_h = frame_h + pad_top
            else:
                detect_img = rgb
                padded_h = frame_h

            # Detección de manos
            results = hands.process(detect_img)

            # Landmarks detectados en coordenadas normalizadas del frame original
            detected = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mapped = []
                    for lm in hand_landmarks.landmark:
                        if pad_top > 0:
                            mapped.append(
                                map_from_padded_to_original(lm, pad_top, padded_h, frame_w, frame_h)
                            )
                        else:
                            mapped.append(
                                landmark_pb2.NormalizedLandmark(
                                    x=max(0.0, min(1.0, lm.x)),
                                    y=max(0.0, min(1.0, lm.y)),
                                    z=lm.z
                                )
                            )
                    detected.append(mapped)

            # ========================
            # Emparejamiento y suavizado
            # ========================
            det_centroids = [centroid_of(d) for d in detected]
            assigned_slot_for_det = [-1] * len(detected)
            used_slots = set()

            # Emparejar detecciones con slots previos (mínima distancia)
            for di, centroid in enumerate(det_centroids):
                best_slot, best_dist = -1, float("inf")
                for si, slot in enumerate(slots):
                    if si in used_slots:
                        continue
                    dist = euclidean_distance(centroid, slot['centroid'])
                    if dist < best_dist:
                        best_dist = dist
                        best_slot = si
                if best_slot != -1 and best_dist <= MATCH_THRESHOLD:
                    assigned_slot_for_det[di] = best_slot
                    used_slots.add(best_slot)

            new_slots = []
            slots_used_next = set()

            # Actualizar slots emparejados
            for di, mapped in enumerate(detected):
                si = assigned_slot_for_det[di]
                if si != -1:
                    prev = slots[si]['landmarks']
                    if len(prev) != len(mapped):
                        prev = mapped.copy()
                    smoothed = smooth_landmark_list(prev, mapped, SMOOTH_ALPHA)
                    cent = centroid_of(smoothed)
                    new_slots.append({'landmarks': smoothed, 'centroid': cent, 'missed': 0})
                    slots_used_next.add(si)

            # Mantener slots no emparejados (si no excedieron MAX_MISSES)
            for si, slot in enumerate(slots):
                if si in slots_used_next:
                    continue
                missed = slot['missed'] + 1
                if missed <= MAX_MISSES:
                    new_slots.append({
                        'landmarks': slot['landmarks'],
                        'centroid': slot['centroid'],
                        'missed': missed
                    })

            # Crear nuevos slots para detecciones no emparejadas
            for di, mapped in enumerate(detected):
                if assigned_slot_for_det[di] == -1:
                    cent = centroid_of(mapped)
                    init = [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in mapped]
                    new_slots.append({'landmarks': init, 'centroid': cent, 'missed': 0})

            # Limitar a MAX_HANDS
            slots = new_slots[:MAX_HANDS]

            # Dibujar resultados suavizados
            for slot in slots:
                lm_list = landmark_pb2.NormalizedLandmarkList(landmark=slot['landmarks'])
                mp_drawing.draw_landmarks(frame, lm_list, mp_hands.HAND_CONNECTIONS)

            out.write(frame)

            # Mostrar progreso en consola
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                print(f"Procesados {frame_idx}/{total_frames} frames. Slots activos: {len(slots)}")

    finally:
        cap.release()
        out.release()

print(f"Proceso terminado. Salida en: {OUTPUT_PATH}")
