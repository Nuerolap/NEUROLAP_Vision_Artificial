# clean_manifest.py
import os
import argparse
import pandas as pd
import numpy as np

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(BASE_PATH, "datasets", "kp_v1")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_csv",  default=os.path.join(DATA_DIR, "_manifest_clean.csv"),
                    help="Ruta al manifest original (_manifest_clean.csv)")
    ap.add_argument("--out", dest="out_csv", default=os.path.join(DATA_DIR, "_manifest_clean.csv"),
                    help="Ruta de salida para el manifest filtrado")
    ap.add_argument("--min_frames", type=int,   default=90,
                    help="Descartar videos con menos de este número de frames")
    ap.add_argument("--max_miss_pct", type=float, default=90.0,
                    help="Descartar si miss_left_pct y miss_right_pct superan este porcentaje (ambas manos)")
    ap.add_argument("--min_fps", type=float,   default=5.0,
                    help="Descartar si fps es menor a este valor (protección contra fps inválidos)")
    ap.add_argument("--check_files", action="store_true",
                    help="Si se activa, descarta entradas cuyo csv_path no exista")
    args = ap.parse_args()

    if not os.path.exists(args.in_csv):
        raise FileNotFoundError(f"No existe manifest de entrada: {args.in_csv}")

    df = pd.read_csv(args.in_csv)
    # normaliza columnas esperadas
    needed = ["label","video","video_id","frames","fps","miss_left_pct","miss_right_pct","csv_path"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Faltan columnas en el manifest: {missing}")

    # Asegurar tipos numéricos
    for c in ["frames","fps","miss_left_pct","miss_right_pct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Reemplaza separadores en csv_path y absolutiza para chequeo
    df["csv_path_norm"] = df["csv_path"].astype(str).str.replace("\\", "/", regex=False)
    df["csv_abs"] = df["csv_path_norm"].apply(lambda p: os.path.abspath(os.path.join(BASE_PATH, p)))

    n0 = len(df)

    # Criterios de descarte
    cond_frames = df["frames"] < args.min_frames
    cond_fps    = df["fps"].isna() | (df["fps"] < args.min_fps)
    cond_miss   = (df["miss_left_pct"] >= args.max_miss_pct) & (df["miss_right_pct"] >= args.max_miss_pct)
    cond_files  = (~df["csv_abs"].apply(os.path.exists)) if args.check_files else pd.Series(False, index=df.index)

    drop = cond_frames | cond_fps | cond_miss | cond_files
    kept = df[~drop].copy()
    dropped = df[drop].copy()

    # Reporte
    print("=== LIMPIEZA DE MANIFEST ===")
    print(f"Total entradas: {n0}")
    print(f"Conservadas   : {len(kept)}")
    print(f"Descartadas   : {len(dropped)}\n")

    def show_examples(mask, title, max_n=8):
        ex = df[mask][["label","video","frames","fps","miss_left_pct","miss_right_pct","csv_path_norm"]].head(max_n)
        if len(ex) > 0:
            print(f"-- {title} ({len(df[mask])}) --")
            print(ex.to_string(index=False))
            print()

    show_examples(cond_frames, f"frames < {args.min_frames}")
    show_examples(cond_fps,    f"fps < {args.min_fps} o NaN")
    show_examples(cond_miss,   f"miss_left_pct y miss_right_pct >= {args.max_miss_pct}%")
    if args.check_files:
        show_examples(cond_files, "csv_path inexistente")

    # Guardar limpio
    out_cols = ["label","video","video_id","frames","fps","miss_left_pct","miss_right_pct","csv_path"]
    kept[out_cols].to_csv(args.out_csv, index=False)
    print(f"[OK] Manifest limpio -> {os.path.relpath(args.out_csv, BASE_PATH)}")

if __name__ == "__main__":
    main()
