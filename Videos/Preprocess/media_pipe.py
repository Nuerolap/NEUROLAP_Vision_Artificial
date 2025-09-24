import cv2
import os
import math
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# =========================
# CONFIGURACIÓN
# =========================

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(BASE_PATH, "augmented-videos")
OUTPUT_DIR = os.path.join(BASE_PATH, "mp-videos")

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parámetros de detección y tracking
MAX_HANDS = 2
MIN_DET_CONF = 0.15   # Confianza mínima para detección inicial
MIN_TRK_CONF = 0.4    # Confianza mínima para tracking entre frames
DETECTION_TOP_PAD = 150  # Padding superior (solo para detección, no en salida)
SMOOTH_ALPHA = 0.15   # Suavizado EMA
MATCH_THRESHOLD = 0.45
MAX_MISSES = 15


# =========================
# UTILIDADES
# =========================

def centroid_of(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def map_from_padded_to_original(lm, pad_top, padded_h, orig_w, orig_h):
    px = lm.x * orig_w
    py_padded = lm.y * padded_h
    py_orig = py_padded - pad_top
    x_norm = max(0.0, min(1.0, px / orig_w))
    y_norm = max(0.0, min(1.0, py_orig / orig_h))
    return landmark_pb2.NormalizedLandmark(x=x_norm, y=y_norm, z=lm.z)

def smooth_landmark_list(prev_list, curr_list, alpha):
    smoothed = []
    for p, c in zip(prev_list, curr_list):
        sx = alpha * p.x + (1 - alpha) * c.x
        sy = alpha * p.y + (1 - alpha) * c.y
        sz = alpha * p.z + (1 - alpha) * c.z
        smoothed.append(landmark_pb2.NormalizedLandmark(x=sx, y=sy, z=sz))
    return smoothed


# =========================
# PROCESAMIENTO DE UN VIDEO
# =========================
def procesar_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir {input_path}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not out.isOpened():
        cap.release()
        print(f"❌ No se pudo crear {output_path}")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    slots = []

    print(f"▶ Procesando {os.path.basename(input_path)} ({frame_w}x{frame_h} @ {fps:.1f}fps, {total_frames} frames)")

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

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Padding superior
                if DETECTION_TOP_PAD > 0:
                    detect_img = cv2.copyMakeBorder(
                        rgb, DETECTION_TOP_PAD, 0, 0, 0,
                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    padded_h = frame_h + DETECTION_TOP_PAD
                else:
                    detect_img = rgb
                    padded_h = frame_h

                # Detección
                results = hands.process(detect_img)
                detected = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mapped = []
                        for lm in hand_landmarks.landmark:
                            if DETECTION_TOP_PAD > 0:
                                mapped.append(map_from_padded_to_original(lm, DETECTION_TOP_PAD, padded_h, frame_w, frame_h))
                            else:
                                mapped.append(
                                    landmark_pb2.NormalizedLandmark(
                                        x=max(0.0, min(1.0, lm.x)),
                                        y=max(0.0, min(1.0, lm.y)),
                                        z=lm.z
                                    )
                                )
                        detected.append(mapped)

                # Emparejamiento y suavizado
                det_centroids = [centroid_of(d) for d in detected]
                assigned_slot_for_det = [-1] * len(detected)
                used_slots = set()

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

                # Mantener slots no emparejados
                for si, slot in enumerate(slots):
                    if si in slots_used_next:
                        continue
                    missed = slot['missed'] + 1
                    if missed <= MAX_MISSES:
                        new_slots.append({**slot, 'missed': missed})

                # Crear nuevos slots
                for di, mapped in enumerate(detected):
                    if assigned_slot_for_det[di] == -1:
                        cent = centroid_of(mapped)
                        init = [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in mapped]
                        new_slots.append({'landmarks': init, 'centroid': cent, 'missed': 0})

                slots = new_slots[:MAX_HANDS]

                # Dibujar landmarks suavizados
                for slot in slots:
                    lm_list = landmark_pb2.NormalizedLandmarkList(landmark=slot['landmarks'])
                    mp_drawing.draw_landmarks(frame, lm_list, mp_hands.HAND_CONNECTIONS)

                out.write(frame)

                if frame_idx % 100 == 0 or frame_idx == total_frames:
                    print(f"  {frame_idx}/{total_frames} frames procesados")

        finally:
            cap.release()
            out.release()

    print(f"✅ Guardado en {output_path}\n")


# =========================
# PROCESAR TODOS LOS VIDEOS
# =========================
def main():
    videos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
    if not videos:
        print("❌ No se encontraron videos en", INPUT_DIR)
        return

    for video in videos:
        input_path = os.path.join(INPUT_DIR, video)
        output_path = os.path.join(OUTPUT_DIR, video)
        procesar_video(input_path, output_path)


if __name__ == "__main__":
    main()
