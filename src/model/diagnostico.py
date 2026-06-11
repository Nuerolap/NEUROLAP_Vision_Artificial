import os
import argparse
import joblib
import numpy as np
import pandas as pd

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR   = os.path.join(BASE_PATH, "artifacts")
DATA_DIR  = os.path.join(BASE_PATH, "datasets", "kp_v1")

MODEL_P   = os.path.join(ART_DIR, "video_quality_clf.joblib")
RECON_CSV = os.path.join(ART_DIR, "reconstruction_by_video_hand.csv")
EMB_WIN   = os.path.join(ART_DIR, "kmeans_window_assignments.csv")
MAN_CLEAN = os.path.join(DATA_DIR, "_manifest_clean.csv")
TRAIN_FEATS = os.path.join(ART_DIR, "training_features.csv")

def _lower_cols(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def load_signal_features(recon_csv: str, manifest_csv: str) -> pd.DataFrame:
    r = pd.read_csv(recon_csv)
    r = _lower_cols(r)
    if not {"video_id","hand","p95"}.issubset(r.columns):
        raise RuntimeError("reconstruction_by_video_hand.csv debe tener (video_id, hand, p95).")

    p95w = r.pivot_table(index="video_id", columns="hand", values="p95", aggfunc="mean").reset_index()
    p95w.columns.name = None
    p95w = p95w.rename(columns={"Left":"p95_left","Right":"p95_right"})
    for c in ("p95_left","p95_right"):
        if c not in p95w.columns: p95w[c] = 0.0

    man = pd.read_csv(manifest_csv)
    man = _lower_cols(man)
    man["video_id"] = man["video"].str.replace("_15fps.mp4","", regex=False)

    keep = ["video_id","frames","fps","miss_left_pct","miss_right_pct"]
    df = p95w.merge(man[keep], on="video_id", how="left")

    df["max_p95"]  = df[["p95_left","p95_right"]].max(axis=1)
    df["min_p95"]  = df[["p95_left","p95_right"]].min(axis=1)
    df["mean_p95"] = df[["p95_left","p95_right"]].mean(axis=1)
    df["max_miss"]  = df[["miss_left_pct","miss_right_pct"]].max(axis=1)
    df["min_miss"]  = df[["miss_left_pct","miss_right_pct"]].min(axis=1)
    df["mean_miss"] = df[["miss_left_pct","miss_right_pct"]].mean(axis=1)

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

def build_current_features():
    sig = load_signal_features(RECON_CSV, MAN_CLEAN)
    clu = build_cluster_features(EMB_WIN)
    X = sig.merge(clu, on="video_id", how="left")
    # imputación defensiva (por si quedaron NaN o fps/frames inválidos)
    for col, default in (("frames", 450.0), ("fps", 15.0)):
        if col in X.columns:
            xcol = pd.to_numeric(X[col], errors="coerce")
            xcol = xcol.where(xcol > 1, pd.NA)
            med  = float(xcol.median(skipna=True)) if xcol.notna().any() else default
            X[col] = xcol.fillna(med)
        else:
            X[col] = default
    X = X.fillna(0.0)
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+", help="IDs sin o con _15fps (ej: VID_20250903_103119)")
    args = ap.parse_args()

    # Carga modelo
    pack = joblib.load(MODEL_P)
    clf       = pack["model"]
    feat_cols = list(pack["feat_cols"])
    classes   = list(pack["classes"])
    scaler    = pack.get("scaler", None)

    # Features actuales (artefactos)
    X_now = build_current_features()

    # Training features (lo que vio el modelo)
    if not os.path.exists(TRAIN_FEATS):
        raise FileNotFoundError(f"No existe {TRAIN_FEATS}")
    T = pd.read_csv(TRAIN_FEATS)

    # Normalización de IDs: en train los ids están normalizados (sin _15fps y lowercase)
    def norm(s: str) -> str:
        b = os.path.basename(str(s))
        b = os.path.splitext(b)[0]
        b = b.replace("-", "_")
        b = b.replace(" ", "")
        for suf in ("_15fps","-15fps"):
            if b.lower().endswith(suf):
                b = b[: -len(suf)]
        return b.strip().lower()

    # Preparar tablas auxiliares para train
    T = T.copy()
    if "video_id_norm" in T.columns:
        T_idcol = "video_id_norm"
    elif "video_id" in T.columns:
        T_idcol = "video_id"
        T["video_id_norm"] = T["video_id"].apply(norm)
    else:
        raise RuntimeError("training_features.csv debe tener 'video_id' o 'video_id_norm'.")
    # Map: id_norm -> fila de train
    T_idx = T.set_index("video_id_norm")

    # Map actual: id -> fila actual + id_norm
    X_now = X_now.copy()
    X_now["video_id_norm"] = X_now["video_id"].apply(norm)
    X_idx = X_now.set_index("video_id")

    # Medianas de train (para referencia)
    train_medians = T[feat_cols].median()

    for vid in args.video_ids:
        # Resolver a *_15fps si hace falta
        cands = [vid, f"{vid}_15fps"] if not vid.endswith("_15fps") else [vid]
        vid_use = next((c for c in cands if c in X_idx.index), None)
        if vid_use is None:
            print(f"⚠️  {vid}: no encontrado en features actuales.")
            continue

        row_now = X_idx.loc[vid_use, feat_cols].astype(float)
        id_norm = norm(vid_use)
        has_train = id_norm in T_idx.index

        print("\n" + "="*80)
        print(f"🎬 VIDEO: {vid_use}   (id_norm='{id_norm}')")
        if has_train:
            row_tr = T_idx.loc[id_norm, feat_cols].astype(float)
            lbl_tr = T_idx.loc[id_norm, "label"] if "label" in T_idx.columns else "(sin label)"
            print(f"Etiqueta en training_features.csv: {lbl_tr}")
        else:
            print("Etiqueta en training_features.csv: (no encontrada)")

        # Z-scores (scaler del modelo)
        if scaler is not None:
            z_now = pd.Series(scaler.transform([row_now.values])[0], index=feat_cols)
            if has_train:
                z_tr  = pd.Series(scaler.transform([row_tr.values])[0],  index=feat_cols)
        else:
            z_now = (row_now - train_medians)  # fallback: delta vs mediana
            if has_train:
                z_tr = (row_tr - train_medians)

        # Predicción actual
        Xv_now = scaler.transform([row_now.values]) if scaler is not None else [row_now.values]
        proba  = clf.predict_proba(Xv_now)[0] if hasattr(clf,"predict_proba") else None
        pred   = clf.predict(Xv_now)[0]
        probs_txt = ", ".join([f"{classes[i]}={proba[i]:.3f}" for i in range(len(classes))]) if proba is not None else "(sin proba)"

        print(f"\n→ Predicción actual: {pred} | Probabilidades: {probs_txt}")

        # Cuadro lado-a-lado (train vs ahora)
        cols_show = ["p95_left","p95_right","max_p95","min_p95","mean_p95",
                     "miss_left_pct","miss_right_pct","max_miss","min_miss","mean_miss",
                     "frames","fps","prop_c0","prop_c1","prop_c2"]
        cols_show = [c for c in cols_show if c in feat_cols]

        def fmt_series(s):
            return pd.Series({k: float(s[k]) for k in cols_show})

        now_vals = fmt_series(row_now)
        now_z    = fmt_series(z_now)

        if has_train:
            tr_vals = fmt_series(row_tr)
            tr_z    = fmt_series(z_tr)
            delta   = now_vals - tr_vals

            table = pd.DataFrame({
                "train": tr_vals,
                "now":   now_vals,
                "Δ(now-train)": delta,
                "z_now": now_z,
                "z_train": tr_z
            })
        else:
            table = pd.DataFrame({
                "now": now_vals,
                "z_now": now_z
            })

        print("\n--- FEATURES (train vs now) ---")
        print(table.to_string(float_format=lambda x: f"{x:.4f}"))

        # Importancias de features (GBDT)
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
            print("\nTop-10 importancias del modelo:")
            print(imp.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n⇒ Sugerencia: fíjate si 'prop_c*' o 'p95_*' cambiaron fuerte vs train, o si fps/frames difieren.")
    print("   Si el video estaba verde en train pero ahora sale rojo, suele deberse a cambios en proporciones de clúster o en p95.")
    print("   También revisa que el ID que pasas corresponde exactamente al clip base vs sus aug_*. \n")

if __name__ == "__main__":
    main()
