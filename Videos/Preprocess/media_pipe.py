import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# -------------------------
# Config
# -------------------------
MIN_DET_CONF = 0.55
MIN_TRK_CONF = 0.80
MODEL_COMPLEXITY = 1
MISSING_DECAY = 24          # sigue siendo útil para suavizado visual (aunque no guardamos video)
CROP_MARGIN = 0.35
CROP_UPSCALE = 1.8
OF_MAX_LEVEL = 3
OF_WIN = (21, 21)
OF_MAX_ITER = 30
OF_EPS = 0.03
OF_EXPAND = 1.4
COUNT_LONG_GAP_SEC = 0.5     # umbral de "pérdida larga"

# -------------------------
# Rutas
# -------------------------
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FOLDER  = os.path.join(BASE_PATH, "processed-videos")
REPORTS_DIR   = os.path.join(BASE_PATH, "reports", "hand_tracking")
PER_VIDEO_DIR = os.path.join(REPORTS_DIR, "per_video")
os.makedirs(PER_VIDEO_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(REPORTS_DIR, "summary.csv")

# -------------------------
# MediaPipe base
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
WRIST_IDX = 0

def run_hands(frame_bgr, static_mode, min_det=MIN_DET_CONF, min_trk=MIN_TRK_CONF):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if static_mode:
        tmp = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk
        )
        res = tmp.process(rgb)
        tmp.close()
    else:
        # Instancia persistente (mejor performance)
        if not hasattr(run_hands, "_persistent"):
            run_hands._persistent = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=MODEL_COMPLEXITY,
                min_detection_confidence=min_det,
                min_tracking_confidence=min_trk
            )
        res = run_hands._persistent.process(rgb)
    return res

def lm_to_bbox(landmarks, W, H, margin=CROP_MARGIN):
    xs = [lm.x * W for lm in landmarks.landmark]
    ys = [lm.y * H for lm in landmarks.landmark]
    x1, x2 = max(0, min(xs)), min(W-1, max(xs))
    y1, y2 = max(0, min(ys)), min(H-1, max(ys))
    w = max(1, x2 - x1); h = max(1, y2 - y1)
    x1 = int(max(0, x1 - margin*w)); y1 = int(max(0, y1 - margin*h))
    x2 = int(min(W-1, x2 + margin*w)); y2 = int(min(H-1, y2 + margin*h))
    return [x1, y1, x2, y2]

def remap_crop_landmarks_to_full(res_crop, x1, y1, W, H, scale=CROP_UPSCALE, crop_shape=None):
    mapped = []
    ch, cw = crop_shape[:2]
    for hlm in res_crop.multi_hand_landmarks:
        lm_list = []
        for lm in hlm.landmark:
            x_up = lm.x * (cw * scale)
            y_up = lm.y * (ch * scale)
            x = (x_up / scale) + x1
            y = (y_up / scale) + y1
            nl = landmark_pb2.NormalizedLandmark(x=float(x / W), y=float(y / H), z=lm.z)
            lm_list.append(nl)
        mapped.append(landmark_pb2.NormalizedLandmarkList(landmark=lm_list))
    class Obj: pass
    out = Obj()
    out.multi_hand_landmarks = mapped
    out.multi_handedness = None
    return out

def recover_in_roi(frame_bgr, roi, W, H):
    (x1, y1, x2, y2) = roi
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_up = cv2.resize(crop, None, fx=CROP_UPSCALE, fy=CROP_UPSCALE, interpolation=cv2.INTER_CUBIC)
    res_crop = run_hands(crop_up, static_mode=True)
    if res_crop and res_crop.multi_hand_landmarks:
        return remap_crop_landmarks_to_full(res_crop, x1, y1, W, H, scale=CROP_UPSCALE, crop_shape=crop.shape)
    return None

def wrist_pixel(lm_list, W, H):
    lm = lm_list.landmark[WRIST_IDX]
    return np.array([[lm.x * W, lm.y * H]], dtype=np.float32)

def expand_bbox_towards(bbox, vec, W, H, factor=OF_EXPAND):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)/2, (y1+y2)/2
    dx, dy = vec
    nx1 = int(max(0, min(cx, cx + factor*dx) - (x2-x1)/2))
    ny1 = int(max(0, min(cy, cy + factor*dy) - (y2-y1)/2))
    nx2 = int(min(W-1, max(cx, cx + factor*dx) + (x2-x1)/2))
    ny2 = int(min(H-1, max(cy, cy + factor*dy) + (y2-y1)/2))
    return [nx1, ny1, nx2, ny2]

def get_handedness_labels(results):
    labels = []
    if results and results.multi_handedness:
        for h in results.multi_handedness:
            labels.append(h.classification[0].label)  # "Left"/"Right"
    return labels

def process_one_video(video_path, per_video_dir=PER_VIDEO_DIR):
    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_csv = os.path.join(per_video_dir, f"{basename}_frames.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {video_path}")
        return None

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    N  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    long_gap_thr = int(round(COUNT_LONG_GAP_SEC * fps))

    # Estado por mano
    state = {
        "Left":  {"last": None, "miss": 0, "last_bbox": None, "last_wrist_px": None},
        "Right": {"last": None, "miss": 0, "last_bbox": None, "last_wrist_px": None},
    }
    prev_gray = None
    lk_params = dict(
        winSize=OF_WIN,
        maxLevel=OF_MAX_LEVEL,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, OF_MAX_ITER, OF_EPS)
    )

    # contadores
    total_frames = 0
    left_ok_frames = 0
    right_ok_frames = 0
    partial_2to1_events = 0

    # gaps largos (contados por mano)
    left_long_gaps = 0
    right_long_gaps = 0
    # acumuladores de run-length actual
    left_run_zero = 0
    right_run_zero = 0

    prev_hands_count = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["frame_idx", "left_detected", "right_detected", "hands_detected"])

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = run_hands(frame, static_mode=False)
            labels = get_handedness_labels(res)
            curr_hands_count = len(res.multi_hand_landmarks) if res and res.multi_hand_landmarks else 0

            # Mapear detecciones a Left/Right
            detected = {"Left": None, "Right": None}
            if res and res.multi_hand_landmarks:
                for hlm, lab in zip(res.multi_hand_landmarks, labels):
                    detected[lab] = hlm

            # actualización por mano + recuperación ligera
            for hand in ("Left", "Right"):
                if detected[hand] is not None:
                    hlm = detected[hand]
                    state[hand]["last"] = hlm
                    state[hand]["miss"] = 0
                    state[hand]["last_bbox"] = lm_to_bbox(hlm, W, H, CROP_MARGIN)
                    state[hand]["last_wrist_px"] = wrist_pixel(hlm, W, H)
                else:
                    st = state[hand]
                    st["miss"] += 1
                    # intento global ligero
                    if st["last"] is not None:
                        rec = recover_in_roi(frame, [0, 0, W-1, H-1], W, H)
                        if rec and rec.multi_hand_landmarks:
                            choose = rec.multi_hand_landmarks[0]
                            st["last"] = choose
                            st["miss"] = 0
                            st["last_bbox"] = lm_to_bbox(choose, W, H, CROP_MARGIN)
                            st["last_wrist_px"] = wrist_pixel(choose, W, H)
                        else:
                            # optical flow del wrist para mover bbox y reintentar
                            bbox = st["last_bbox"] or [0,0,W-1,H-1]
                            if prev_gray is not None and st["last_wrist_px"] is not None:
                                new_wrist, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray,
                                                                                  st["last_wrist_px"], None, **lk_params)
                                if status is not None and status[0][0] == 1:
                                    move = (new_wrist[0] - st["last_wrist_px"][0]).ravel()
                                    bbox = expand_bbox_towards(bbox, move, W, H, factor=OF_EXPAND)
                            rec2 = recover_in_roi(frame, bbox, W, H)
                            if rec2 and rec2.multi_hand_landmarks:
                                choose = rec2.multi_hand_landmarks[0]
                                st["last"] = choose
                                st["miss"] = 0
                                st["last_bbox"] = lm_to_bbox(choose, W, H, CROP_MARGIN)
                                st["last_wrist_px"] = wrist_pixel(choose, W, H)

            # métricas por frame
            left_present  = 1 if state["Left"]["last"]  is not None else 0
            right_present = 1 if state["Right"]["last"] is not None else 0
            hands_present = left_present + right_present

            # eventos 2->1
            if prev_hands_count == 2 and hands_present == 1:
                partial_2to1_events += 1
            prev_hands_count = hands_present

            # gaps largos: run-length de ceros por mano
            left_run_zero  = 0 if left_present  else (left_run_zero + 1)
            right_run_zero = 0 if right_present else (right_run_zero + 1)
            if left_run_zero == long_gap_thr:
                left_long_gaps += 1
            if right_run_zero == long_gap_thr:
                right_long_gaps += 1

            writer.writerow([idx, left_present, right_present, hands_present])

            # contadores globales
            total_frames += 1
            left_ok_frames  += left_present
            right_ok_frames += right_present

            prev_gray = gray
            idx += 1

    cap.release()

    # porcentajes
    left_miss_pct  = 100.0 * (total_frames - left_ok_frames)  / max(1, total_frames)
    right_miss_pct = 100.0 * (total_frames - right_ok_frames) / max(1, total_frames)

    print(f"📄 {os.path.basename(video_path)} → frames: {total_frames} | "
          f"%miss L: {left_miss_pct:.2f}% | %miss R: {right_miss_pct:.2f}% | "
          f"2→1: {partial_2to1_events} | long_gaps L/R: {left_long_gaps}/{right_long_gaps}")

    return {
        "video": os.path.basename(video_path),
        "frames": total_frames,
        "fps": float(fps),
        "miss_left_pct": left_miss_pct,
        "miss_right_pct": right_miss_pct,
        "partial_2to1": partial_2to1_events,
        "long_gaps_left": left_long_gaps,
        "long_gaps_right": right_long_gaps
    }

if __name__ == "__main__":
    videos = [os.path.join(INPUT_FOLDER, f)
              for f in os.listdir(INPUT_FOLDER)
              if f.lower().endswith(".mp4")]
              # si solo quieres *_15fps.mp4: if f.endswith("_15fps.mp4")

    if not videos:
        print("⚠️ No hay mp4 en processed-videos/")
        raise SystemExit(0)

    summaries = []
    for vp in videos:
        print(f"\n▶️ Analizando: {os.path.basename(vp)}")
        try:
            one = process_one_video(vp, PER_VIDEO_DIR)
            if one:
                summaries.append(one)
        except Exception as e:
            print(f"❌ Error en {os.path.basename(vp)}: {e}")

    # guardar resumen maestro
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as fsum:
        writer = csv.writer(fsum)
        writer.writerow(["video","frames","fps",
                         "miss_left_pct","miss_right_pct",
                         "partial_2to1","long_gaps_left","long_gaps_right"])
        for s in summaries:
            writer.writerow([
                s["video"], s["frames"], f'{s["fps"]:.2f}',
                f'{s["miss_left_pct"]:.2f}', f'{s["miss_right_pct"]:.2f}',
                s["partial_2to1"], s["long_gaps_left"], s["long_gaps_right"]
            ])

    # también imprime promedio global
    if summaries:
        avg_miss_left  = np.mean([s["miss_left_pct"] for s in summaries])
        avg_miss_right = np.mean([s["miss_right_pct"] for s in summaries])
        print("\n===== RESUMEN GLOBAL =====")
        print(f"Promedio %miss Left : {avg_miss_left:.2f}%")
        print(f"Promedio %miss Right: {avg_miss_right:.2f}%")
        print(f"CSV resumen: {SUMMARY_CSV}")
        print(f"CSVs por video: {PER_VIDEO_DIR}")
