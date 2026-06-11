# predict_quality.py (IDs -> usa artefactos, con DEBUG fuerte)
import os, argparse, warnings, joblib
import pandas as pd
import numpy as np

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR   = os.path.join(BASE_PATH, "artifacts")
MODEL_P   = os.path.join(ART_DIR, "video_quality_clf.joblib")
RECON     = os.path.join(ART_DIR, "reconstruction_by_video_hand.csv")
EMB_WIN   = os.path.join(ART_DIR, "kmeans_window_assignments.csv")

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Helpers ----------
def find_manifest():
    candidates = [
        os.path.join(BASE_PATH, "datasets", "kp_v1", "_manifest_clean.csv"),
        os.path.abspath(os.path.join(ART_DIR, "..", "datasets", "kp_v1", "_manifest_clean.csv")),
        os.path.join(BASE_PATH, "_manifest_clean.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

def _lower_cols(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def load_signal_features_from_recon(recon_csv: str, manifest_csv: str, debug=False) -> pd.DataFrame:
    # --- Recon p95 por mano ---
    r = pd.read_csv(recon_csv)
    r = _lower_cols(r)
    need = {"video_id","hand","p95"}
    if not need.issubset(r.columns):
        raise RuntimeError("reconstruction_by_video_hand.csv debe tener (video_id, hand, p95).")

    p95w = r.pivot_table(index="video_id", columns="hand", values="p95", aggfunc="mean").reset_index()
    p95w.columns.name = None
    p95w = p95w.rename(columns={"left":"p95_left","right":"p95_right"})
    for c in ("p95_left","p95_right"):
        if c not in p95w.columns: 
            p95w[c] = 0.0

    # --- Manifest con frames/fps/miss ---
    if manifest_csv is None or not os.path.exists(manifest_csv):
        raise FileNotFoundError("No se encontró _manifest_clean.csv (no puedo leer frames/fps).")

    man = pd.read_csv(manifest_csv)
    man = _lower_cols(man)

    # 👇 REGLA NUEVA:
    # - Si el manifest YA trae 'video_id', úsalo tal cual (no toques el sufijo _15fps).
    # - Si no trae 'video_id', derivarlo de 'video' quitando solo la extensión (conserva _15fps).
    if "video_id" in man.columns and man["video_id"].notna().any():
        # normaliza a string y déjalo como viene (con _15fps)
        man["video_id"] = man["video_id"].astype(str)
    elif "video" in man.columns:
        man["video_id"] = man["video"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
    else:
        raise RuntimeError("_manifest_clean.csv debe tener 'video_id' o 'video' para derivarlo.")

    keep = ["video_id","frames","fps","miss_left_pct","miss_right_pct"]
    missing = [c for c in keep if c not in man.columns]
    if missing:
        raise RuntimeError(f"Faltan columnas en _manifest_clean.csv: {missing}")

    # Merge 1:1 por video_id (idéntico a RECON: con _15fps)
    df = p95w.merge(man[keep], on="video_id", how="left")

    # Derivadas
    df["max_p95"]  = df[["p95_left","p95_right"]].max(axis=1)
    df["min_p95"]  = df[["p95_left","p95_right"]].min(axis=1)
    df["mean_p95"] = df[["p95_left","p95_right"]].mean(axis=1)
    df["max_miss"]  = df[["miss_left_pct","miss_right_pct"]].max(axis=1)
    df["min_miss"]  = df[["miss_left_pct","miss_right_pct"]].min(axis=1)
    df["mean_miss"] = df[["miss_left_pct","miss_right_pct"]].mean(axis=1)

    # No imputes aquí salvo que sea estrictamente necesario
    # (si aún tienes NaN aislados, puedes hacer df = df.fillna(0.0) como red).
    df = df.fillna(0.0)

    if debug:
        print("\n[DEBUG] Ejemplos de df mergeado (p95 + manifest):")
        print(df.head(3).to_string(index=False))

    return df


def build_cluster_features(kwin_csv: str) -> pd.DataFrame:
    k = pd.read_csv(kwin_csv)
    k = _lower_cols(k)
    if "video_id" not in k.columns or "cluster" not in k.columns:
        raise RuntimeError("kmeans_window_assignments.csv debe tener 'video_id' y 'cluster'.")

    total = k.groupby("video_id").size().rename("n").reset_index()
    counts = k.groupby(["video_id","cluster"]).size().rename("cnt").reset_index()
    m = counts.merge(total, on="video_id", how="left")
    m["prop"] = m["cnt"] / m["n"].replace(0, np.nan)

    piv = m.pivot(index="video_id", columns="cluster", values="prop").fillna(0.0).reset_index()
    piv.columns.name = None
    # renombrar numéricos -> prop_c0/1/2
    rename = {}
    for c in piv.columns:
        if isinstance(c, (int, np.integer)):
            rename[c] = f"prop_c{c}"
    piv = piv.rename(columns=rename)
    for kcol in ("prop_c0","prop_c1","prop_c2"):
        if kcol not in piv.columns:
            piv[kcol] = 0.0
    return piv

def normalize_ids(wanted_ids, known_ids):
    known = set(known_ids)
    norm, missing = [], []
    for vid in wanted_ids:
        cands = [vid, f"{vid}_15fps"] if not vid.endswith("_15fps") else [vid]
        match = next((c for c in cands if c in known), None)
        if match: norm.append(match)
        else:     missing.append(vid)
    return norm, missing

def build_features_for(video_id_list, manifest_csv: str, debug=False):
    # p95/miss/frames/fps
    sig = load_signal_features_from_recon(RECON, manifest_csv, debug=debug)
    # prop_c*
    clu = build_cluster_features(EMB_WIN)
    X   = sig.merge(clu, on="video_id", how="left")

    # --------- IMPUTACIÓN CORRECTA (cruda, no usar training_features.csv) ---------
    # frames/fps <=1 o NaN -> imputar con mediana cruda del conjunto o defaults
    for col, default in (("frames", 450.0), ("fps", 15.0)):
        if col in X.columns:
            xcol = pd.to_numeric(X[col], errors="coerce")
            xcol = xcol.where(xcol > 1, pd.NA)      # 0,1 o negativos -> NA
            med  = float(xcol.median(skipna=True)) if xcol.notna().any() else default
            X[col] = xcol.fillna(med)
        else:
            X[col] = default

    # rellenar proporciones faltantes y cualquier otra col numérica con 0.0
    for kcol in ("prop_c0","prop_c1","prop_c2"):
        if kcol not in X.columns:
            X[kcol] = 0.0
    # NaN generales -> 0 para no romper el scaler/modelo
    X = X.fillna(0.0)

    if debug:
        print("\n--- FEATURES CRUDOS (antes de escalar) ---")
        cols_dbg = ["video_id","p95_left","p95_right","max_p95","mean_p95",
                    "miss_left_pct","miss_right_pct","mean_miss","frames","fps",
                    "prop_c0","prop_c1","prop_c2"]
        cols_dbg = [c for c in cols_dbg if c in X.columns]
        mask = X["video_id"].astype(str).str.contains("|".join(video_id_list), case=False, regex=True)
        print(X.loc[mask, cols_dbg].head(10).to_string(index=False))

    want_norm, not_found = normalize_ids(video_id_list, X["video_id"].unique())
    if not_found:
        print("⚠️  IDs no encontrados en artefactos:", ", ".join(not_found))

    Xs = X[X["video_id"].isin(want_norm)].copy()
    return Xs, want_norm

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="*", help="IDs sin sufijo o con _15fps (ej: VID_20250829_100122)")
    ap.add_argument("--save", help="Ruta CSV para guardar predicciones (opcional)")
    ap.add_argument("--debug", action="store_true", help="Imprime diagnósticos detallados")
    args = ap.parse_args()

    if not os.path.exists(MODEL_P):
        print(f"❌ No existe el modelo: {MODEL_P}")
        return
    pack = joblib.load(MODEL_P)
    clf      = pack["model"]
    feat_cols= list(pack["feat_cols"])
    classes  = list(pack["classes"])
    scaler   = pack.get("scaler", None)

    manifest_csv = find_manifest()
    if args.debug:
        print(f"[DEBUG] BASE_PATH: {BASE_PATH}")
        print(f"[DEBUG] ART_DIR  : {ART_DIR}")
        print(f"[DEBUG] RECON    : {RECON} -> exists={os.path.exists(RECON)}")
        print(f"[DEBUG] EMB_WIN  : {EMB_WIN} -> exists={os.path.exists(EMB_WIN)}")
        print(f"[DEBUG] MANIFEST : {manifest_csv} -> exists={os.path.exists(manifest_csv) if manifest_csv else False}")

    if manifest_csv is None:
        print("❌ No se encontró _manifest_clean.csv. Verifica datasets/kp_v1/_manifest_clean.csv")
        return

    X, want = build_features_for(args.video_ids, manifest_csv, debug=args.debug)
    if X.empty:
        print("❌ No hay features para esos video_id (revisa los nombres).")
        return

    missing_feats = [c for c in feat_cols if c not in X.columns]
    if missing_feats:
        print("❌ Faltan columnas en X:", missing_feats)
        print("Disponibles:", list(X.columns))
        return

    Xv = X[feat_cols].values
    if scaler is not None:
        Xv = scaler.transform(Xv)
        if args.debug:
            df_scaled = pd.DataFrame(Xv, columns=feat_cols)
            print("\n--- STATS features escalados (post-scaler) ---")
            print(pd.DataFrame({
                "min": df_scaled.min(),
                "median": df_scaled.median(),
                "max": df_scaled.max()
            }).T)
        Xv = X[feat_cols].values

    # Anti-NaN por si las dudas
    if np.isnan(Xv).any():
        nan_cols = np.array(feat_cols)[np.isnan(Xv).any(axis=0)]
        print("[WARN] Se detectaron NaN en columnas:", list(nan_cols))
        Xv = np.nan_to_num(Xv, nan=0.0)

    if scaler is not None:
        Xv = scaler.transform(Xv)
        if args.debug:
            df_scaled = pd.DataFrame(Xv, columns=feat_cols)
            print("\n--- STATS features escalados (post-scaler) ---")
            print(pd.DataFrame({
                "min": df_scaled.min(),
                "median": df_scaled.median(),
                "max": df_scaled.max()
            }).T)

    probs = clf.predict_proba(Xv)
    preds = clf.predict(Xv)

    out = X[["video_id"]].copy()
    out["pred"] = preds
    for i, c in enumerate(classes):
        out[f"p_{c}"] = probs[:, i]

    for _, row in out.iterrows():
        vid = row["video_id"]
        pr  = row["pred"]
        ps  = ", ".join([f"{c}={row[f'p_{c}']:.3f}" for c in classes])
        print(f"🎬 {vid} -> Predicción: {pr} | Probabilidades: {ps}")

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        out.to_csv(args.save, index=False)
        print(f"[OK] guardado -> {args.save}")

if __name__ == "__main__":
    main()
