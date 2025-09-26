import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple

# =========================
# Config
# =========================
FILTER_15FPS = False
         # True: solo procesa *_15fps.mp4
WRITE_ALL_FRAMES = True      # siempre escribe Left y Right por frame (con mask=0 si no hay mano)
OUTPUT_DIR_NAME = os.path.join("datasets", "kp_v1")
VALID_EXTS = (".mp4", ".mov", ".mkv", ".avi")
LABELS = ["rojo", "amarillo", "verde"]  # subcarpetas esperadas

# =========================
# Rutas
# =========================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_ROOT   = os.path.join(BASE_PATH, "processed-videos")   # processed-videos/<label>/*.mp4
OUTPUT_ROOT  = os.path.join(BASE_PATH, OUTPUT_DIR_NAME)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# MediaPipe
# =========================
mp_hands = mp.solutions.hands

def get_persistent_hands():
    if not hasattr(get_persistent_hands, "_hands"):
        get_persistent_hands._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.80
        )
    return get_persistent_hands._hands

def handedness_labels(results):
    labs = []
    if results and results.multi_handedness:
        for h in results.multi_handedness:
            labs.append(h.classification[0].label)  # "Left"/"Right"
    return labs

# =========================
# Utilidades CSV
# =========================
LANDMARK_POINTS = 21
LM_COLS   = [f"{axis}{i}" for i in range(LANDMARK_POINTS) for axis in ("x","y","z")]
MASK_COLS = [f"m{i}" for i in range(LANDMARK_POINTS)]
CSV_HEADER = ["video_id","hand","frame_idx","t_sec"] + LM_COLS + MASK_COLS

def write_row(writer, video_id, hand, frame_idx, t_sec, lm_list):
    if lm_list is None:
        coords = [0.0]*(LANDMARK_POINTS*3)
        mask   = [0]*LANDMARK_POINTS
    else:
        coords, mask = [], []
        for lm in lm_list.landmark:
            coords.extend([float(lm.x), float(lm.y), float(lm.z)])
            mask.append(1)
    writer.writerow([video_id, hand, frame_idx, t_sec] + coords + mask)

# =========================
# Descubrimiento de videos
# =========================
def list_labeled_videos() -> List[Tuple[str, str]]:
    """
    Retorna [(ruta_video, label)] recorriendo processed-videos/<label>/...
    """
    pairs = []
    for lab in LABELS:
        lab_dir = os.path.join(INPUT_ROOT, lab)
        if not os.path.isdir(lab_dir):
            continue
        for f in os.listdir(lab_dir):
            if f.lower().endswith(VALID_EXTS):
                if FILTER_15FPS and not f.endswith("_15fps.mp4"):
                    continue
                pairs.append((os.path.join(lab_dir, f), lab))
    return sorted(pairs)

# =========================
# Procesamiento por video
# =========================
def process_one_video(video_path: str, label: str):
    video_id = os.path.splitext(os.path.basename(video_path))[0]  # ej: VID_..._15fps
    out_dir  = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    out_csv  = os.path.join(out_dir, f"{video_id}.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hands = get_persistent_hands()
    total = 0
    left_present_frames = 0
    right_present_frames = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(CSV_HEADER)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_sec = frame_idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            lms_left = None
            lms_right = None
            if res and res.multi_hand_landmarks:
                labels = handedness_labels(res)
                for hlm, lab in zip(res.multi_hand_landmarks, labels):
                    if lab == "Left":
                        lms_left = hlm
                    elif lab == "Right":
                        lms_right = hlm

            if lms_left is not None:
                left_present_frames += 1
            if lms_right is not None:
                right_present_frames += 1

            if WRITE_ALL_FRAMES:
                write_row(writer, video_id, "Left",  frame_idx, t_sec, lms_left)
                write_row(writer, video_id, "Right", frame_idx, t_sec, lms_right)
            else:
                if lms_left  is not None: write_row(writer, video_id, "Left",  frame_idx, t_sec, lms_left)
                if lms_right is not None: write_row(writer, video_id, "Right", frame_idx, t_sec, lms_right)

            total += 1
            frame_idx += 1

    cap.release()

    miss_left_pct  = 100.0 * (total - left_present_frames)  / max(1, total)
    miss_right_pct = 100.0 * (total - right_present_frames) / max(1, total)

    print(f"✅ {label}/{os.path.basename(video_path)} → frames: {total} | "
          f"%miss L: {miss_left_pct:.2f}% | %miss R: {miss_right_pct:.2f}% | "
          f"csv: {os.path.relpath(out_csv, BASE_PATH)}")

    return {
        "label": label,
        "video": os.path.basename(video_path),
        "video_id": video_id,
        "frames": total,
        "fps": float(fps),
        "miss_left_pct": miss_left_pct,
        "miss_right_pct": miss_right_pct,
        "csv_path": out_csv
    }

# =========================
# Main
# =========================
def main():
    videos = list_labeled_videos()
    if not videos:
        print("⚠️ No hay videos en processed-videos/<rojo|amarillo|verde>/")
        return

    summaries = []
    for vp, lab in videos:
        try:
            s = process_one_video(vp, lab)
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"❌ Error en {lab}/{os.path.basename(vp)}: {e}")

    # Manifiesto
    manifest_csv = os.path.join(OUTPUT_ROOT, "_manifest.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as fman:
        writer = csv.writer(fman)
        writer.writerow(["label","video","video_id","frames","fps","miss_left_pct","miss_right_pct","csv_path"])
        for s in summaries:
            writer.writerow([
                s["label"], s["video"], s["video_id"], s["frames"], f'{s["fps"]:.2f}',
                f'{s["miss_left_pct"]:.2f}', f'{s["miss_right_pct"]:.2f}',
                os.path.relpath(s["csv_path"], BASE_PATH)
            ])

    print(f"\n📄 Manifiesto: {os.path.relpath(manifest_csv, BASE_PATH)}")
    print(f"🗂  CSVs por video en: {os.path.relpath(OUTPUT_ROOT, BASE_PATH)}")

if __name__ == "__main__":
    main()
