import os
import csv
import cv2
import glob
import argparse
import numpy as np
import mediapipe as mp
from typing import List, Tuple

# =========================
# Constantes base
# =========================
LANDMARK_POINTS = 21
LM_COLS   = [f"{axis}{i}" for i in range(LANDMARK_POINTS) for axis in ("x","y","z")]
MASK_COLS = [f"m{i}" for i in range(LANDMARK_POINTS)]
CSV_HEADER = ["video_id","hand","frame_idx","t_sec"] + LM_COLS + MASK_COLS
VALID_EXTS = (".mp4", ".mov", ".mkv", ".avi")
DEFAULT_LABELS = ["rojo", "amarillo", "verde"]

BASE_PATH   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_ROOT  = os.path.join(BASE_PATH, "processed-videos")
OUTPUT_ROOT = os.path.join(BASE_PATH, "datasets", "kp_v1")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =========================
# MediaPipe (singleton)
# =========================
mp_hands = mp.solutions.hands
def get_hands():
    if not hasattr(get_hands, "_h"):
        get_hands._h = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.80
        )
    return get_hands._h

def handedness_labels(results):
    labs = []
    if results and results.multi_handedness:
        for h in results.multi_handedness:
            labs.append(h.classification[0].label)  # "Left"/"Right"
    return labs

def write_row(writer, video_id, hand, frame_idx, t_sec, lm_list):
    if lm_list is None:
        coords = [0.0]*(LANDMARK_POINTS*3)
        mask   = [0]*LANDMARK_POINTS
    else:
        coords, mask = [], []
        for lm in lm_list.landmark:
            coords.extend([float(lm.x), float(lm.y), float(lm.z)])
            mask.append(1)
    writer.writerow([video_id, hand, frame_idx, float(t_sec)] + coords + mask)

# =========================
# Descubrimiento
# =========================
def list_labeled_videos(labels, pattern=None):
    pairs = []
    for lab in labels:
        lab_dir = os.path.join(INPUT_ROOT, lab)
        if not os.path.isdir(lab_dir):
            continue
        if pattern:
            cand = glob.glob(os.path.join(lab_dir, pattern))
        else:
            cand = [os.path.join(lab_dir, f) for f in os.listdir(lab_dir)]
        for p in sorted(cand):
            if os.path.splitext(p)[1].lower() in VALID_EXTS and os.path.isfile(p):
                pairs.append((p, lab))
    return pairs

# =========================
# Procesamiento
# =========================
def process_one_video(video_path: str, label: str, write_all_frames: bool = True):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_dir  = os.path.join(OUTPUT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    out_csv  = os.path.join(out_dir, f"{video_id}.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {video_path}")
        return None

    # Info previa (para logging)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n_raw   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"▶️  Procesando: {label}/{os.path.basename(video_path)}")
    print(f"    - Resolución: {w}x{h} | fps_raw: {fps_raw:.2f} | frames(declarados): {n_raw}")
    csv_rel = os.path.relpath(out_csv, BASE_PATH).replace("\\", "/")
    print(f"    - CSV salida: {csv_rel}")


    hands = get_hands()
    total = 0
    left_present_frames = 0
    right_present_frames = 0

    t0_msec = None
    last_msec = None

    with open(out_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(CSV_HEADER)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms and pos_ms > 0:
                t_sec = float(pos_ms) / 1000.0
                if t0_msec is None:
                    t0_msec = pos_ms
                last_msec = pos_ms
            else:
                t_sec = (frame_idx / fps_raw) if fps_raw > 0 else float(frame_idx)

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

            if write_all_frames:
                write_row(writer, video_id, "Left",  frame_idx, t_sec, lms_left)
                write_row(writer, video_id, "Right", frame_idx, t_sec, lms_right)
            else:
                if lms_left  is not None: write_row(writer, video_id, "Left",  frame_idx, t_sec, lms_left)
                if lms_right is not None: write_row(writer, video_id, "Right", frame_idx, t_sec, lms_right)

            total += 1
            frame_idx += 1

    cap.release()

    # duración y fps efectivos
    if t0_msec is not None and last_msec is not None and last_msec > t0_msec:
        dur_sec = (last_msec - t0_msec) / 1000.0
    else:
        dur_sec = (total / fps_raw) if fps_raw > 0 else float(total)
    fps_eff = float(total / dur_sec) if dur_sec > 0 else (fps_raw if fps_raw > 0 else 0.0)

    miss_left_pct  = 100.0 * (total - left_present_frames)  / max(1, total)
    miss_right_pct = 100.0 * (total - right_present_frames) / max(1, total)

    csv_rel = os.path.relpath(out_csv, BASE_PATH).replace("\\","/")

    print(f"    ✔ frames_leídos: {total} | fps_eff: {fps_eff:.2f} | "
          f"missL: {miss_left_pct:.2f}% | missR: {miss_right_pct:.2f}%")
    print()

    return {
        "label": label,
        "video": os.path.basename(video_path),
        "video_id": video_id,
        "frames": total,
        "fps": float(fps_eff),
        "miss_left_pct": miss_left_pct,
        "miss_right_pct": miss_right_pct,
        "csv_path": csv_rel
    }

# =========================
# Main con logs/flags
# =========================
def main():
    ap = argparse.ArgumentParser(description="Extractor de keypoints con logs y control.")
    ap.add_argument("--only-labels", type=str, default="rojo,amarillo,verde",
                    help="Filtra etiquetas (coma). Ej: rojo,verde")
    ap.add_argument("--pattern", type=str, default=None,
                    help="Patrón de archivo a incluir (glob). Ej: *_15fps.mp4")
    ap.add_argument("--confirm-each", action="store_true",
                    help="Pedir confirmación por cada video (y/n).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Saltar videos cuyo CSV ya exista.")
    ap.add_argument("--list", action="store_true",
                    help="Sólo listar lo que se procesaría (no extrae).")
    ap.add_argument("--write-all-frames", action="store_true",
                    help="Escribir Left y Right por frame (default on).")
    args = ap.parse_args()

    labels = [s.strip() for s in (args.only_labels or "").split(",") if s.strip()]
    if not labels:
        labels = DEFAULT_LABELS

    videos = list_labeled_videos(labels, pattern=args.pattern)
    if not videos:
        print("⚠️  No hay videos que cumplan el filtro.")
        return

    print(f"Encontrados {len(videos)} videos:")
    for vp, lab in videos:
        print(f"  - {lab}/{os.path.basename(vp)}")
    print()

    if args.list:
        print("Modo --list: no se procesará nada.")
        return

    summaries = []
    for vp, lab in videos:
        video_id = os.path.splitext(os.path.basename(vp))[0]
        out_csv  = os.path.join(OUTPUT_ROOT, lab, f"{video_id}.csv")
        if args.skip_existing and os.path.exists(out_csv):
            print(f"⏭️  Saltando (CSV ya existe): {lab}/{os.path.basename(vp)}")
            continue
        if args.confirm_each:
            resp = input(f"¿Procesar {lab}/{os.path.basename(vp)}? [y/n] ").strip().lower()
            if resp not in ("y","yes","s","si","sí"):
                print("  → omitido por usuario.\n")
                continue
        try:
            s = process_one_video(vp, lab, write_all_frames=args.write_all_frames)
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"❌ Error en {lab}/{os.path.basename(vp)}: {e}\n")

    if not summaries:
        print("⚠️  No hay resultados para escribir manifiesto.")
        return

    # Manifiesto
    manifest_csv = os.path.join(OUTPUT_ROOT, "_manifest_clean.csv")
    with open(manifest_csv, "w", newline="", encoding="utf-8") as fman:
        writer = csv.writer(fman)
        writer.writerow(["label","video","video_id","frames","fps",
                         "miss_left_pct","miss_right_pct","csv_path"])
        for s in summaries:
            writer.writerow([
                s["label"], s["video"], s["video_id"], s["frames"],
                f'{float(s["fps"]):.2f}',
                f'{float(s["miss_left_pct"]):.2f}', f'{float(s["miss_right_pct"]):.2f}',
                s["csv_path"]
            ])

    print("\n📄 Manifiesto escrito en:",
          os.path.relpath(manifest_csv, BASE_PATH).replace("\\","/"))
    print(f"🗂  CSVs por video en:",
          os.path.relpath(OUTPUT_ROOT, BASE_PATH).replace("\\","/"))

if __name__ == "__main__":
    main()
