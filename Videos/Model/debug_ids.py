# debug_ids.py
import os, sys, argparse, joblib
import pandas as pd
import numpy as np

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR   = os.path.join(BASE_PATH, "artifacts")
MODEL_P   = os.path.join(ART_DIR, "video_quality_clf.joblib")
RECON     = os.path.join(ART_DIR, "reconstruction_by_video_hand.csv")
EMB_WIN   = os.path.join(ART_DIR, "kmeans_window_assignments.csv")
TRN_FEAT  = os.path.join(ART_DIR, "training_features.csv")

def lower_cols(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df

def load_signal_features_from_recon(recon_csv):
    r = lower_cols(pd.read_csv(recon_csv))
    if {"p95_left","p95_right"}.issubset(r.columns):
        need = ["video_id","p95_left","p95_right","frames","fps"]
        for c in need:
            if c not in r.columns:
                raise RuntimeError(f"Falta '{c}' en {os.path.basename(recon_csv)}")
        miss_left  = r["miss_left_pct"]  if "miss_left_pct"  in r.columns else 0.0
        miss_right = r["miss_right_pct"] if "miss_right_pct" in r.columns else 0.0
        sig = pd.DataFrame({
            "video_id": r["video_id"],
            "p95_left": r["p95_left"],
            "p95_right": r["p95_right"],
            "miss_left_pct": miss_left,
            "miss_right_pct": miss_right,
            "frames": r["frames"],
            "fps": r["fps"],
        })
    else:
        need = {"video_id","hand","p95"}
        if not need.issubset(r.columns):
            raise RuntimeError("reconstruction_by_video_hand.csv debe tener (video_id, hand, p95) o p95_left/right.")
        p95w = r.pivot_table(index="video_id", columns="hand", values="p95", aggfunc="max")
        p95w.columns = [f"p95_{str(c).lower()}" for c in p95w.columns]
        p95w = p95w.reset_index()
        if "miss_pct" in r.columns:
            mw = r.pivot_table(index="video_id", columns="hand", values="miss_pct", aggfunc="max").fillna(0.0)
            mw.columns = [f"miss_{str(c).lower()}_pct" for c in mw.columns]
            mw = mw.reset_index()
        else:
            mw = pd.DataFrame({"video_id": p95w["video_id"]})
            mw["miss_left_pct"] = 0.0; mw["miss_right_pct"] = 0.0
        agg = r.groupby("video_id").agg(
            frames=("frames","max") if "frames" in r.columns else ("video_id","size"),
            fps   =("fps","max")    if "fps"    in r.columns else ("video_id","size")
        ).reset_index()
        if "fps" not in agg.columns:    agg["fps"] = 15.0
        if "frames" not in agg.columns: agg["frames"] = r.groupby("video_id").size().values
        sig = p95w.merge(mw, on="video_id", how="left").merge(agg, on="video_id", how="left")
        for c in ["p95_left","p95_right","miss_left_pct","miss_right_pct"]:
            if c not in sig.columns: sig[c] = 0.0
        sig = sig.fillna(0.0)

    sig["max_p95"]  = sig[["p95_left","p95_right"]].max(axis=1)
    sig["min_p95"]  = sig[["p95_left","p95_right"]].min(axis=1)
    sig["mean_p95"] = sig[["p95_left","p95_right"]].mean(axis=1)
    sig["max_miss"]  = sig[["miss_left_pct","miss_right_pct"]].max(axis=1)
    sig["min_miss"]  = sig[["miss_left_pct","miss_right_pct"]].min(axis=1)
    sig["mean_miss"] = sig[["miss_left_pct","miss_right_pct"]].mean(axis=1)
    return sig

def build_cluster_features(kwin_csv):
    k = lower_cols(pd.read_csv(kwin_csv))
    if "video_id" not in k.columns or "cluster" not in k.columns:
        raise RuntimeError("kmeans_window_assignments.csv debe tener 'video_id' y 'cluster'.")
    total = k.groupby("video_id").size().rename("n").reset_index()
    counts = k.groupby(["video_id","cluster"]).size().rename("cnt").reset_index()
    m = counts.merge(total, on="video_id", how="left")
    m["prop"] = m["cnt"] / m["n"]
    piv = m.pivot(index="video_id", columns="cluster", values="prop").fillna(0.0).reset_index()
    piv.columns = [f"prop_c{int(c)}" if isinstance(c, (int, np.integer)) else c for c in piv.columns]
    for c in ["prop_c0","prop_c1","prop_c2"]:
        if c not in piv.columns: piv[c] = 0.0
    return piv

def norm_candidates(vid):
    return [vid, f"{vid}_15fps"] if not vid.endswith("_15fps") else [vid]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="+")
    args = ap.parse_args()

    pack = joblib.load(MODEL_P)
    clf       = pack["model"]
    feat_cols = list(pack["feat_cols"])
    classes   = list(pack["classes"])
    scaler    = pack.get("scaler", None)

    sig = load_signal_features_from_recon(RECON)
    clu = build_cluster_features(EMB_WIN)
    X   = sig.merge(clu, on="video_id", how="left").fillna(0.0)

    # mapear ids pedidas -> existentes (con/sin _15fps)
    exists = set(X["video_id"])
    wanted = []
    missing = []
    for v in args.video_ids:
        m = next((c for c in norm_candidates(v) if c in exists), None)
        (wanted if m else missing).append(m or v)
    if missing:
        print("⚠️  IDs no encontrados en artefactos:", ", ".join(missing))

    Xpick = X[X["video_id"].isin(wanted)].copy()
    if Xpick.empty:
        print("❌ No hay filas para esos IDs en artefactos.")
        return

    # enseñar features crudos que usa el modelo
    cols_show = ["video_id","p95_left","p95_right","max_p95","mean_p95",
                 "miss_left_pct","miss_right_pct","mean_miss","frames","fps",
                 "prop_c0","prop_c1","prop_c2"]
    print("\n--- FEATURES CRUDOS ---")
    print(Xpick[cols_show].to_string(index=False))

    # escalar y predecir
    Xv = Xpick[feat_cols].values
    if scaler is not None:
        Xv = scaler.transform(Xv)
    probs = clf.predict_proba(Xv)
    preds = clf.predict(Xv)

    print("\n--- MODELO ---")
    print("Clases del modelo:", classes)
    for i, vid in enumerate(Xpick["video_id"].values):
        pstr = ", ".join([f"{classes[j]}={probs[i,j]:.3f}" for j in range(len(classes))])
        print(f"🎬 {vid} -> pred={preds[i]} | {pstr}")

    # intentar cruzar contra training_features.csv para ver la etiqueta usada en train
    if os.path.exists(TRN_FEAT):
        tf = pd.read_csv(TRN_FEAT)
        # archivo guarda video_id_norm; armonizamos minúsculas
        tf["video_id_norm"] = tf["video_id_norm"].astype(str).str.lower()
        def norm_id(s):
            s = str(s).lower()
            if s.endswith("_15fps"): s = s[:-6]
            return s
        Xpick["video_id_norm"] = Xpick["video_id"].astype(str).str.lower().map(norm_id)
        jj = Xpick.merge(tf[["video_id_norm","label"]], on="video_id_norm", how="left")
        print("\n--- ETIQUETA EN TRAIN (si existía) ---")
        print(jj[["video_id","video_id_norm","label"]].to_string(index=False))
    else:
        print("\n(No se encontró artifacts/training_features.csv para cruzar etiquetas de entrenamiento.)")

if __name__ == "__main__":
    main()
