import os
import cv2
import csv
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple

# =========================
# Parámetros por defecto
# =========================
DEF_TARGET_FPS = 15
DEF_RESOLUTION = (640, 360)   # (w, h)
DEF_CODEC = "mp4v"
DEF_OVERWRITE = False
DEF_LABELS = ["rojo", "amarillo", "verde"]  # puedes cambiar/añadir

VALID_EXTS = (".mp4", ".mov", ".mkv", ".avi")

def resize_letterbox(frame, target=(640, 360)):
    tw, th = target
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=frame.dtype)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

def normalize_video_id(basename_noext: str) -> str:
    """
    ID normalizado para empatar en pipeline: usamos el nombre de salida sin extensión.
    """
    return f"{basename_noext}_15fps"

def convert_video(input_path: str, output_path: str,
                  target_fps=DEF_TARGET_FPS,
                  resolution=DEF_RESOLUTION,
                  codec=DEF_CODEC) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {input_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not in_fps or in_fps <= 0:
        in_fps = 30.0  # fallback

    print(f"➡️ {os.path.basename(input_path)} | FPS in {in_fps:.2f} → out {target_fps} | "
          f"{in_w}x{in_h} → {resolution[0]}x{resolution[1]} | frames in {in_frames}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, float(target_fps), resolution)
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear el archivo de salida: {output_path}")

    ratio = float(target_fps) / float(in_fps)
    accum = 0.0
    out_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        accum += ratio
        while accum >= 1.0:
            out.write(resize_letterbox(frame, resolution))
            out_count += 1
            accum -= 1.0

    cap.release()
    out.release()

    dur_in  = (in_frames / in_fps) if in_fps > 0 else 0.0
    dur_out = (out_count / target_fps) if target_fps > 0 else 0.0
    print(f"✅ Guardado: {output_path} | frames out {out_count} | dur in≈{dur_in:.2f}s dur out≈{dur_out:.2f}s\n")

    return {
        "in_fps": in_fps, "in_frames": in_frames, "in_w": in_w, "in_h": in_h,
        "out_fps": target_fps, "out_frames": out_count,
        "out_w": resolution[0], "out_h": resolution[1],
        "dur_in_s": dur_in, "dur_out_s": dur_out,
    }

def collect_labeled_videos(in_dir: str, labels: List[str]) -> List[Tuple[str, str]]:
    """
    Devuelve lista de (ruta_video, label). Solo dentro de subcarpetas cuyo nombre sea un label.
    """
    pairs = []
    for lab in labels:
        lab_dir = os.path.join(in_dir, lab)
        if not os.path.isdir(lab_dir):
            continue
        for f in os.listdir(lab_dir):
            if f.lower().endswith(VALID_EXTS):
                pairs.append((os.path.join(lab_dir, f), lab))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Preprocess (letterbox + fps) con estructura por labels")
    parser.add_argument("--input",  default=None, help="Carpeta de entrada (por defecto training-videos)")
    parser.add_argument("--output", default=None, help="Carpeta de salida (por defecto processed-videos)")
    parser.add_argument("--labels", nargs="*", default=DEF_LABELS, help="Lista de labels (subcarpetas)")
    parser.add_argument("--fps",    type=float, default=DEF_TARGET_FPS, help="FPS objetivo")
    parser.add_argument("--width",  type=int,   default=DEF_RESOLUTION[0], help="Ancho objetivo")
    parser.add_argument("--height", type=int,   default=DEF_RESOLUTION[1], help="Alto objetivo")
    parser.add_argument("--overwrite", action="store_true", help="Reescribir salidas existentes")
    args = parser.parse_args()

    BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_dir  = args.input  or os.path.join(BASE, "training-videos")
    out_dir = args.output or os.path.join(BASE, "processed-videos")
    os.makedirs(out_dir, exist_ok=True)

    # artifacts
    art_dir = os.path.join(BASE, "artifacts")
    os.makedirs(art_dir, exist_ok=True)
    manifest_path = os.path.join(art_dir, "preprocess_manifest.csv")
    labels_csv    = os.path.join(art_dir, "labels_from_manifest.csv")
    write_header = not os.path.exists(manifest_path)

    videos = collect_labeled_videos(in_dir, args.labels)
    if not videos:
        print("⚠️  No se encontraron videos en subcarpetas:", args.labels, "dentro de", in_dir)
        return

    # Para escribir labels_from_manifest.csv sin duplicar, iremos acumulando aquí
    label_rows = []

    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "ts","label","in_file","out_file","video_id_norm",
                "in_fps","in_frames","in_w","in_h",
                "out_fps","out_frames","out_w","out_h",
                "dur_in_s","dur_out_s","status","error"
            ])

        for in_path, lab in videos:
            base_noext = os.path.splitext(os.path.basename(in_path))[0]
            out_name   = f"{base_noext}_15fps.mp4"
            # salida en carpeta por label
            lab_out_dir = os.path.join(out_dir, lab)
            os.makedirs(lab_out_dir, exist_ok=True)
            out_path = os.path.join(lab_out_dir, out_name)

            vid_norm = normalize_video_id(base_noext)

            if (not args.overwrite) and os.path.exists(out_path):
                print(f"⏭️  Saltando (ya existe): {lab}/{out_name}")
                w.writerow([datetime.now().isoformat(), lab, in_path, out_path, vid_norm,
                            "", "", "", "", "", "", "", "", "", "", "skipped_exists", ""])
                # también añadimos al mapping label
                label_rows.append([vid_norm, lab])
                continue

            try:
                meta = convert_video(
                    in_path, out_path,
                    target_fps=args.fps,
                    resolution=(args.width, args.height)
                )
                w.writerow([datetime.now().isoformat(), lab, in_path, out_path, vid_norm,
                            f"{meta['in_fps']:.4f}", meta["in_frames"], meta["in_w"], meta["in_h"],
                            f"{meta['out_fps']:.4f}", meta["out_frames"], meta["out_w"], meta["out_h"],
                            f"{meta['dur_in_s']:.3f}", f"{meta['dur_out_s']:.3f}", "ok", ""])
                label_rows.append([vid_norm, lab])
            except Exception as e:
                print(f"❌ Error con {in_path}: {e}")
                w.writerow([datetime.now().isoformat(), lab, in_path, out_path, vid_norm,
                            "", "", "", "", "", "", "", "", "", "", "error", str(e)])

    # Generar labels_from_manifest.csv (video_id_norm, label)
    # Evitar duplicados manteniendo el último visto
    if label_rows:
        # deduplicar por video_id_norm (simple dict)
        dedup = {}
        for vid_norm, lab in label_rows:
            dedup[vid_norm] = lab
        with open(labels_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id_norm", "label"])
            for vid_norm, lab in sorted(dedup.items()):
                w.writerow([vid_norm, lab])
        print(f"[OK] labels_from_manifest.csv -> {labels_csv}")

    print(f"[OK] Manifest -> {manifest_path}")
    print("🎯 Listo. Salidas en:", out_dir)

if __name__ == "__main__":
    main()
