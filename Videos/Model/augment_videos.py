import os
import cv2
import numpy as np
import random
import argparse
from typing import List, Dict, Tuple

# ---------------- Paths ----------------
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_ROOT   = os.path.join(BASE_PATH, "processed-videos")
OUT_ROOT  = os.path.join(BASE_PATH, "augmented-videos")
os.makedirs(OUT_ROOT, exist_ok=True)

LABELS = ["amarillo", "rojo", "verde"]
VALID_EXTS = (".mp4", ".mov", ".mkv", ".avi")

# --------- Variantes (exactamente las 12 que pasaste) ---------
BRILLO_DELTA = 40
CONTRASTE_DELTA = 1.5
HORIZONTAL_PIXELS = 47
VERTICAL_PIXELS = 26
SCALE_OUT = 0.8
SCALE_DOWN = 0.85
SCALE_STRETCH = 0.83
STRETCH = 1.2
VARIANT_LIMITS = {
    "amarillo": 12,
    "rojo": 4,
    "verde": 3
}

def ajustar_brillo(img, delta): return cv2.convertScaleAbs(img, alpha=1.0, beta=delta)
def ajustar_contraste(img, alpha): return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

def escalar_centro(img, scale):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    return cv2.warpAffine(img, M, (w, h))

def desplazar_centro(img, dx, dy, pre_scale=1.0):
    red = escalar_centro(img, pre_scale)
    h, w = red.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(red, M, (w, h))

def estirar_centro(img, sx, sy, pre_scale=1.0):
    red = escalar_centro(img, pre_scale)
    h, w = red.shape[:2]
    cx, cy = w/2, h/2
    M = np.float32([[sx, 0, cx - sx*cx],
                    [0, sy, cy - sy*cy]])
    return cv2.warpAffine(red, M, (w, h))

VARIANTES = [
    ("mas_brillo",   lambda f: ajustar_brillo(f, BRILLO_DELTA)),
    ("menos_brillo", lambda f: ajustar_brillo(f, -BRILLO_DELTA)),
    ("mas_cont",     lambda f: ajustar_contraste(f, CONTRASTE_DELTA)),
    ("menos_cont",   lambda f: ajustar_contraste(f, CONTRASTE_DELTA/2)),
    ("shift_der",    lambda f: desplazar_centro(f, HORIZONTAL_PIXELS, 0, pre_scale=SCALE_DOWN)),
    ("shift_izq",    lambda f: desplazar_centro(f, -HORIZONTAL_PIXELS, 0, pre_scale=SCALE_DOWN)),
    ("shift_arriba", lambda f: desplazar_centro(f, 0, -VERTICAL_PIXELS, pre_scale=SCALE_DOWN)),
    ("shift_abajo",  lambda f: desplazar_centro(f, 0, VERTICAL_PIXELS, pre_scale=SCALE_DOWN)),
    ("est_hori",     lambda f: estirar_centro(f, STRETCH, 1.0, pre_scale=SCALE_STRETCH)),
    ("est_vert",     lambda f: estirar_centro(f, 1.0, STRETCH, pre_scale=SCALE_STRETCH)),
    ("alejado",      lambda f: escalar_centro(f, SCALE_OUT)),
    # (no añadimos más, respetamos tus 12)
]

# ---------------- Utils ----------------
def list_videos(root: str, label: str) -> List[str]:
    d = os.path.join(root, label)
    if not os.path.isdir(d): return []
    return sorted([os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(VALID_EXTS)])

def writable_variants(in_path: str, out_dir: str, chosen: List[str]) -> Dict[str, cv2.VideoWriter]:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir: {in_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    base = os.path.splitext(os.path.basename(in_path))[0]
    writers = {}
    for name in chosen:
        out_path = os.path.join(out_dir, f"{base}__aug_{name}.mp4")
        if os.path.exists(out_path):
            continue
        writers[name] = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    cap.release()
    return writers

def write_augmented(in_path: str, writers: Dict[str, cv2.VideoWriter]):
    if not writers: return 0
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened(): return 0

    name_to_fn = {n: fn for (n, fn) in VARIANTES}
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        for name, wv in writers.items():
            wv.write(name_to_fn[name](frame))
        frames += 1
    cap.release()
    for wv in writers.values(): wv.release()
    return len(writers)

def distribute_plan(n_orig: int, n_target: int, n_files: int, cap_per_video: int) -> List[int]:
    """Cuántas variantes por video (no exceder cap_per_video)."""
    need = max(0, n_target - n_orig)
    if need == 0 or n_files == 0: return [0]*n_files
    base = min(need // n_files, cap_per_video)
    rem  = need - base*n_files
    plan = [base]*n_files
    idxs = list(range(n_files))
    random.shuffle(idxs)
    for i in idxs:
        if rem == 0: break
        if plan[i] < cap_per_video:
            plan[i] += 1
            rem -= 1
    return plan

def max_total_with_cap(n_orig: int, cap: int) -> int:
    return n_orig + n_orig*cap

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Augment balanceado con caps por clase.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap-amarillo", type=int, default=12)
    ap.add_argument("--cap-rojo",     type=int, default=5)
    ap.add_argument("--cap-verde",    type=int, default=4)
    ap.add_argument("--target", type=int, default=None,
                    help="Objetivo por clase. Si no se pasa, se usa el máximo conteo actual.")
    args = ap.parse_args()
    random.seed(args.seed)

    caps = {"amarillo": args.cap_amarillo, "rojo": args.cap_rojo, "verde": args.cap_verde}
    for k,v in caps.items():
        if v > len(VARIANTES): caps[k] = len(VARIANTES)

    files = {lab: list_videos(IN_ROOT, lab) for lab in LABELS}
    counts = {lab: len(files[lab]) for lab in LABELS}
    print("[INFO] Conteo actual:", counts)
    print("[INFO] Caps por video:", caps)

    # Si no pasas --target, igualamos al máximo actual (p. ej. 75)
    raw_default_target = max(counts.values()) if counts else 0

    # Para que TODAS las clases puedan alcanzar el target con sus caps
    # el target no debe superar min( max_total_con_cap(clase) )
    max_totals = {lab: max_total_with_cap(counts[lab], caps[lab]) for lab in LABELS}
    reachable_ceiling = min(max_totals.values()) if max_totals else 0

    target = args.target if args.target is not None else raw_default_target
    if target > reachable_ceiling:
        print(f"[WARN] target={target} no alcanzable por todas las clases con los caps.")
        print(f"       Se ajusta a {reachable_ceiling}.")
        target = reachable_ceiling

    print(f"[INFO] Target por clase: {target}")
    print(f"[INFO] Techos alcanzables con caps: {max_totals}")

    variant_names = [n for (n, _) in VARIANTES]

    for lab in LABELS:
        in_files = files[lab]
        n_orig   = counts[lab]
        cap_lab  = caps[lab]
        out_dir  = os.path.join(OUT_ROOT, lab)
        os.makedirs(out_dir, exist_ok=True)

        plan = distribute_plan(n_orig, target, len(in_files), cap_lab)
        if sum(plan) == 0:
            print(f"▶ {lab}: ya está en target ({n_orig}).")
            continue

        for in_path, k in zip(in_files, plan):
            if k <= 0: continue
            chosen = random.sample(variant_names, k)  # k variantes distintas
            writers = writable_variants(in_path, out_dir, chosen)
            n_written = write_augmented(in_path, writers)
            print(f"  {lab} | {os.path.basename(in_path)} -> {n_written} variantes")

    # Resumen rápido
    final_counts = {}
    for lab in LABELS:
        base = counts.get(lab,0)
        out_dir = os.path.join(OUT_ROOT, lab)
        aug = len([f for f in os.listdir(out_dir) if "__aug_" in f])
        final_counts[lab] = base + aug
    print("[INFO] Totales aprox tras augment:", final_counts)
    print(f"✅ Aumentos en: {OUT_ROOT}")

if __name__ == "__main__":
    main()
