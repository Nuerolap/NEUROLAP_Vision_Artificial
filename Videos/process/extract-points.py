import os
import csv
import cv2
import numpy as np
import mediapipe as mp

# =========================
# Config
# =========================
# Si solo quieres procesar archivos terminados en _15fps.mp4, pon FILTER_15FPS = True
FILTER_15FPS = False
WRITE_ALL_FRAMES = True  # True = escribe fila por mano SIEMPRE, aunque no aparezca (mask=0 y coords=0)
OUTPUT_DIR_NAME = os.path.join("datasets", "kp_v1")

# =========================
# Rutas
# =========================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_FOLDER  = os.path.join(BASE_PATH, "processed-videos")
OUTPUT_FOLDER = os.path.join(BASE_PATH, OUTPUT_DIR_NAME)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# MediaPipe
# =========================
mp_hands = mp.solutions.hands

def get_persistent_hands():
    # Una sola instancia persistente (mejor rendimiento)
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
            labs.append(h.classification[0].label)  # "Left" / "Right"
    return labs

# =========================
# Utilidades
# =========================
LANDMARK_POINTS = 21
LM_COLS = [f"{axis}{i}" for i in range(LANDMARK_POINTS) for axis in ("x","y","z")]
MASK_COLS = [f"m{i}" for i in range(LANDMARK_POINTS)]

CSV_HEADER = ["video_id","hand","frame_idx","t_sec"] + LM_COLS + MASK_COLS

def write_row(writer, video_id, hand, frame_idx, t_sec, lm_list):
    """
    lm_list:
      - lista de 21 landmarks mediapipe (NormalizedLandmark)  -> generamos coords y mask=1
      - None -> coords 0 y mask=0
    """
    if lm_list is None:
        coords = [0.0]*(LANDMARK_POINTS*3)
        mask   = [0]*LANDMARK_POINTS
        row = [video_id, hand, frame_idx, t_sec] + coords + mask
        writer.writerow(row)
        return

    coords = []
    mask   = []
    # MediaPipe da coordenadas normalizadas [0..1] ya; z es relativa. Las dejamos tal cual.
    for lm in lm_list.landmark:
        coords.extend([float(lm.x), float(lm.y), float(lm.z)])
        mask.append(1)
    row = [video_id, hand, frame_idx, t_sec] + coords + mask
    writer.writerow(row)

def process_one_video(video_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_csv  = os.path.join(OUTPUT_FOLDER, f"{video_id}.csv")

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

            # Mapeamos a Left/Right
            lms_left = None
            lms_right = None
            if res and res.multi_hand_landmarks:
                labels = handedness_labels(res)
                for hlm, lab in zip(res.multi_hand_landmarks, labels):
                    if lab == "Left":
                        lms_left = hlm
                    elif lab == "Right":
                        lms_right = hlm

            # contar presencia por frame
            if lms_left is not None:
                left_present_frames += 1
            if lms_right is not None:
                right_present_frames += 1

            if WRITE_ALL_FRAMES:
                # siempre escribir dos filas (Left y Right)
                write_row(writer, video_id, "Left",  frame_idx, t_sec, lms_left)
                write_row(writer, video_id, "Right", frame_idx, t_sec, lms_right)
            else:
                # solo escribir si la mano está presente (menos cómodo para ventanas fijas)
                if lms_left is not None:
                    write_row(writer, video_id, "Left", frame_idx, t_sec, lms_left)
                if lms_right is not None:
                    write_row(writer, video_id, "Right", frame_idx, t_sec, lms_right)

            total += 1
            frame_idx += 1

    cap.release()

    miss_left_pct  = 100.0 * (total - left_present_frames)  / max(1, total)
    miss_right_pct = 100.0 * (total - right_present_frames) / max(1, total)

    print(f"✅ {os.path.basename(video_path)} → frames: {total} | "
          f"%miss L: {miss_left_pct:.2f}% | %miss R: {miss_right_pct:.2f}% | "
          f"csv: {os.path.relpath(out_csv, BASE_PATH)}")

    return {
        "video": os.path.basename(video_path),
        "frames": total,
        "fps": float(fps),
        "miss_left_pct": miss_left_pct,
        "miss_right_pct": miss_right_pct,
        "csv_path": out_csv
    }

def main():
    # Listar videos
    videos = [os.path.join(INPUT_FOLDER, f)
              for f in os.listdir(INPUT_FOLDER)
              if f.lower().endswith(".mp4")]

    if FILTER_15FPS:
        videos = [v for v in videos if v.endswith("_15fps.mp4")]

    if not videos:
        print("⚠️ No hay mp4 en processed-videos/")
        return

    summaries = []
    for vp in sorted(videos):
        try:
            s = process_one_video(vp)
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"❌ Error en {os.path.basename(vp)}: {e}")

    # Manifiesto
    manifest_csv = os.path.join(OUTPUT_FOLDER, "_manifest.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as fman:
        writer = csv.writer(fman)
        writer.writerow(["video","frames","fps","miss_left_pct","miss_right_pct","csv_path"])
        for s in summaries:
            writer.writerow([
                s["video"], s["frames"], f'{s["fps"]:.2f}',
                f'{s["miss_left_pct"]:.2f}', f'{s["miss_right_pct"]:.2f}',
                os.path.relpath(s["csv_path"], BASE_PATH)
            ])

    print(f"\n📄 Manifiesto: {os.path.relpath(manifest_csv, BASE_PATH)}")
    print(f"🗂  CSVs por video en: {os.path.relpath(OUTPUT_FOLDER, BASE_PATH)}")

if __name__ == "__main__":
    main()
