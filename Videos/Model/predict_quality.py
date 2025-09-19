# predict_quality.py (robusto a esquemas ancho/largo)
import os, sys, argparse, warnings
import joblib
import pandas as pd

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR   = os.path.join(BASE_PATH, "artifacts")
MODEL_P   = os.path.join(ART_DIR, "video_quality_clf.joblib")
RECON     = os.path.join(ART_DIR, "reconstruction_by_video_hand.csv")
EMB_WIN   = os.path.join(ART_DIR, "kmeans_window_assignments.csv")

warnings.filterwarnings("ignore", category=UserWarning)

def _lower_cols(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def load_signal_features_from_recon(recon_csv: str) -> pd.DataFrame:
    r = pd.read_csv(recon_csv)
    r = _lower_cols(r)

    # --- Caso 1: ya viene ancho (p95_left/right) ---
    if {"p95_left","p95_right"}.issubset(r.columns):
        # nombre de columnas esperadas en este caso
        need = ["video_id","p95_left","p95_right","frames","fps"]
        for c in need:
            if c not in r.columns:
                raise RuntimeError(f"Falta columna '{c}' en {os.path.basename(recon_csv)}")

        # miss puede venir como miss_left_pct/right_pct o no venir
        if {"miss_left_pct","miss_right_pct"}.issubset(r.columns):
            miss_left  = r["miss_left_pct"]
            miss_right = r["miss_right_pct"]
        else:
            # si no existe, asumimos 0
            miss_left  = 0.0
            miss_right = 0.0

        sig = pd.DataFrame({
            "video_id":   r["video_id"],
            "p95_left":   r["p95_left"],
            "p95_right":  r["p95_right"],
            "miss_left_pct":  miss_left,
            "miss_right_pct": miss_right,
            "frames":     r["frames"],
            "fps":        r["fps"],
        })

    # --- Caso 2: formato largo: columnas por mano (hand: Left/Right) ---
    else:
        # columnas mínimas para pivotar
        if not {"video_id","hand","p95"}.issubset(r.columns):
            raise RuntimeError(
                "El CSV de reconstrucción debe tener (video_id, hand, p95) o las columnas anchas p95_left/p95_right."
            )
        # p95 -> ancho
        p95w = r.pivot_table(index="video_id", columns="hand", values="p95", aggfunc="max")
        # renombrar columnas a left/right robusto
        p95w.columns = [f"p95_{str(c).lower()}" for c in p95w.columns]
        p95w = p95w.reset_index()

        # miss -> ancho (si existe miss_pct por mano)
        if "miss_pct" in r.columns:
            mw = r.pivot_table(index="video_id", columns="hand", values="miss_pct", aggfunc="max").fillna(0.0)
            mw.columns = [f"miss_{str(c).lower()}_pct" for c in mw.columns]
            mw = mw.reset_index()
        else:
            mw = pd.DataFrame({"video_id": p95w["video_id"]})
            mw["miss_left_pct"] = 0.0
            mw["miss_right_pct"] = 0.0

        # frames/fps (repetidos por video): usa max o first
        agg = r.groupby("video_id").agg(
            frames=("frames","max") if "frames" in r.columns else ("video_id","size"),
            fps   =("fps","max")    if "fps" in r.columns else ("video_id", "size")
        ).reset_index()
        if "fps" not in agg.columns:   agg["fps"] = 15.0  # por defecto si no estaba
        if "frames" not in agg.columns: agg["frames"] = r.groupby("video_id").size().values

        sig = p95w.merge(mw, on="video_id", how="left").merge(agg, on="video_id", how="left")
        # si falta alguna de p95_left/right porque no hubo esa mano, rellena con 0
        for c in ["p95_left","p95_right","miss_left_pct","miss_right_pct"]:
            if c not in sig.columns:
                sig[c] = 0.0
        sig = sig.fillna(0.0)

    # features agregadas
    sig["max_p95"]  = sig[["p95_left","p95_right"]].max(axis=1)
    sig["min_p95"]  = sig[["p95_left","p95_right"]].min(axis=1)
    sig["mean_p95"] = sig[["p95_left","p95_right"]].mean(axis=1)
    sig["max_miss"]  = sig[["miss_left_pct","miss_right_pct"]].max(axis=1)
    sig["min_miss"]  = sig[["miss_left_pct","miss_right_pct"]].min(axis=1)
    sig["mean_miss"] = sig[["miss_left_pct","miss_right_pct"]].mean(axis=1)
    return sig

def build_cluster_features(kwin_csv: str) -> pd.DataFrame:
    k = pd.read_csv(kwin_csv)
    k = _lower_cols(k)
    if "video_id" not in k.columns or "cluster" not in k.columns:
        raise RuntimeError("kmeans_window_assignments.csv debe tener columnas 'video_id' y 'cluster'.")

    total = k.groupby("video_id").size().rename("n").reset_index()
    counts = k.groupby(["video_id","cluster"]).size().rename("cnt").reset_index()
    m = counts.merge(total, on="video_id", how="left")
    m["prop"] = m["cnt"] / m["n"]
    piv = m.pivot(index="video_id", columns="cluster", values="prop").fillna(0.0)
    piv.columns = [f"prop_c{int(c)}" for c in piv.columns]
    piv = piv.reset_index()
    return piv

def normalize_ids(wanted_ids, known_ids):
    known = set(known_ids)
    norm, missing = [], []
    for vid in wanted_ids:
        cands = [vid, f"{vid}_15fps"] if not vid.endswith("_15fps") else [vid]
        match = next((c for c in cands if c in known), None)
        (norm if match else missing).append(match or vid)
    return [x for x in norm if x], missing

def build_features_for(video_id_list):
    sig = load_signal_features_from_recon(RECON)          # p95/miss/frames/fps + derivados
    clu = build_cluster_features(EMB_WIN)                 # prop_c*
    X   = sig.merge(clu, on="video_id", how="left").fillna(0.0)
    want_norm, not_found = normalize_ids(video_id_list, X["video_id"].unique())
    if not_found:
        print("⚠️  IDs no encontrados en artefactos:", ", ".join([v for v in not_found if v not in want_norm]))
    return X[X["video_id"].isin(want_norm)].copy(), want_norm

def main(video_ids, save_path=None):
    if not os.path.exists(MODEL_P):
        print(f"❌ No existe el modelo: {MODEL_P}")
        return
    pack = joblib.load(MODEL_P)
    clf, feat_cols, classes = pack["model"], pack["feat_cols"], pack["classes"]

    X, want = build_features_for(video_ids)
    if X.empty:
        print("❌ No hay features para esos video_id (revisa los nombres).")
        return

    missing_feats = [c for c in feat_cols if c not in X.columns]
    if missing_feats:
        print("❌ Faltan columnas de features en X:", missing_feats)
        print("Disponibles:", list(X.columns))
        return

    probs = clf.predict_proba(X[feat_cols].values)
    preds = clf.predict(X[feat_cols].values)

    out = X[["video_id"]].copy()
    out["pred"] = preds
    for i, c in enumerate(classes):
        out[f"p_{c}"] = probs[:, i]

    cols = ["video_id","pred"] + [c for c in out.columns if c.startswith("p_")]
    out = out[cols]
    print(out.to_string(index=False))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out.to_csv(save_path, index=False)
        print(f"[OK] guardado -> {save_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="*", help="IDs sin sufijo o con _15fps. Ej: VID_20250829_100122")
    ap.add_argument("--save", help="Ruta CSV para guardar predicciones (opcional)")
    args = ap.parse_args()
    if not args.video_ids:
        print("Proveer al menos un video_id. Ej: python predict_quality.py VID_20250829_100122 VID_20250829_103350")
    else:
        main(args.video_ids, save_path=args.save)
