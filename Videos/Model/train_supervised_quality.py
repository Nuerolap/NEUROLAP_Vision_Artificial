import os
import re
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# ==== Rutas base ====
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR   = os.path.join(BASE_PATH, "artifacts")
DATA_DIR  = os.path.join(BASE_PATH, "datasets", "kp_v1")

RECON_CSV   = os.path.join(ART_DIR, "reconstruction_by_video_hand.csv")  # p95 por video/hand
EMB_WIN_CSV = os.path.join(ART_DIR, "kmeans_window_assignments.csv")     # cluster por ventana
MANIFEST    = os.path.join(DATA_DIR, "_manifest.csv")                    # miss_% y paths

LABELS_DIR  = os.path.join(BASE_PATH, "labels")  # rojo/amarillo/verde
MODEL_OUT   = os.path.join(ART_DIR, "video_quality_clf.joblib")
FEATURES_OUT= os.path.join(ART_DIR, "training_features.csv")

# ---------- Normalización de IDs ----------
def normalize_id(s: str) -> str:
    """
    Reglas:
      - minúsculas
      - quitar carpeta/extension
      - quitar sufijos como _15fps o -15fps
      - reemplazar guiones por underscores
      - quitar espacios
      - quitar sufijos como " (2)" o "_(2)"
    """
    if not isinstance(s, str):
        return s
    b = os.path.basename(s)
    b = re.sub(r'\.[A-Za-z0-9]+$', '', b)  # quitar extensión
    b = re.sub(r'(?i)[_-]?15fps$', '', b)  # quitar _15fps o -15fps al final
    b = b.replace('-', '_')
    b = b.replace(' ', '')
    b = re.sub(r'[\(\[]\s*\d+\s*[\)\]]$', '', b)  # quitar (2), [2], etc. al final
    return b.strip().lower()

# ---------- Carga de etiquetas ----------
def load_labels_from_folders(labels_dir):
    rows = []
    for lbl in ("rojo", "amarillo", "verde"):
        d = os.path.join(labels_dir, lbl)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".mp4",".mov",".mkv",".avi")):
                vid_norm = normalize_id(f)
                rows.append({"video_id_norm": vid_norm, "label": lbl, "orig_file": f})
    df = pd.DataFrame(rows).drop_duplicates("video_id_norm")
    if df.empty:
        raise RuntimeError(f"No se encontraron etiquetas en {labels_dir}. Revisa rutas y extensiones.")
    print(f"[INFO] Etiquetas cargadas: {Counter(df['label'])}")
    return df

# ---------- Features de clusters ----------
def build_cluster_features(emb_win_csv):
    emb = pd.read_csv(emb_win_csv)
    if "video_id" not in emb.columns or "cluster" not in emb.columns:
        raise RuntimeError("kmeans_window_assignments.csv debe tener columnas (video_id, cluster).")
    emb["video_id_norm"] = emb["video_id"].apply(normalize_id)
    grp = emb.groupby(["video_id_norm","cluster"]).size().reset_index(name="n")
    tot = grp.groupby("video_id_norm")["n"].sum().reset_index(name="N")
    m = grp.merge(tot, on="video_id_norm", how="left")
    m["prop"] = m["n"] / m["N"].replace(0, np.nan)
    pv = m.pivot(index="video_id_norm", columns="cluster", values="prop").fillna(0.0).reset_index()
    pv.columns.name = None
    # renombrar
    rename = {}
    for c in pv.columns:
        if isinstance(c, (int, np.integer)):
            rename[c] = f"prop_c{c}"
    pv = pv.rename(columns=rename)
    for k in ("prop_c0","prop_c1","prop_c2"):
        if k not in pv.columns:
            pv[k] = 0.0
    return pv

def load_signal_features(recon_csv, manifest_csv):
    r = pd.read_csv(recon_csv)

    # Normalizar ID y mano (por si vienen como 'left', 'RIGHT', etc.)
    r["video_id_norm"] = r["video_id"].apply(normalize_id)
    r["hand"] = r["hand"].str.title()  # -> 'Left', 'Right'

    # Si hay varias filas por (video, hand), agregamos.
    # Puedes cambiar a .median() o .max() si prefieres.
    agg = (r.groupby(["video_id_norm", "hand"], as_index=False)
             .agg(p95=("p95", "mean")))

    # Pivot ahora ya no tiene duplicados
    pv = agg.pivot(index="video_id_norm", columns="hand", values="p95").reset_index()
    pv.columns.name = None
    pv = pv.rename(columns={"Left": "p95_left", "Right": "p95_right"})

    # Manifest con miss_% y meta
    man = pd.read_csv(manifest_csv)
    if "video" not in man.columns:
        raise RuntimeError("_manifest.csv debe tener columna 'video'.")
    man["video_id_norm"] = man["video"].apply(normalize_id)

    keep = ["video_id_norm", "miss_left_pct", "miss_right_pct", "frames", "fps"]
    man = man[keep]

    df = pv.merge(man, on="video_id_norm", how="left")

    # Features compuestas (usa skipna)
    df["max_p95"]  = df[["p95_left", "p95_right"]].max(axis=1, skipna=True)
    df["min_p95"]  = df[["p95_left", "p95_right"]].min(axis=1, skipna=True)
    df["mean_p95"] = df[["p95_left", "p95_right"]].mean(axis=1, skipna=True)

    df["max_miss"]  = df[["miss_left_pct", "miss_right_pct"]].max(axis=1, skipna=True)
    df["min_miss"]  = df[["miss_left_pct", "miss_right_pct"]].min(axis=1, skipna=True)
    df["mean_miss"] = df[["miss_left_pct", "miss_right_pct"]].mean(axis=1, skipna=True)

    return df


def main():
    # 1) Etiquetas
    y_df = load_labels_from_folders(LABELS_DIR)   # video_id_norm, label

    # 2) Features
    sig = load_signal_features(RECON_CSV, MANIFEST)   # tiene video_id_norm
    clu = build_cluster_features(EMB_WIN_CSV)         # tiene video_id_norm
    X = sig.merge(clu, on="video_id_norm", how="left")

    # 3) Unir con labels
    data = y_df.merge(X, on="video_id_norm", how="inner")
    data = data.dropna().reset_index(drop=True)

    # Diagnóstico si quedó vacío o muy chico
    print(f"[INFO] Total videos con label y features: {len(data)}")
    if len(data) == 0:
        # mostrar por qué no unió
        left_only = set(y_df["video_id_norm"]) - set(X["video_id_norm"])
        right_only= set(X["video_id_norm"]) - set(y_df["video_id_norm"])
        print("\n[DIAGNÓSTICO] Algunos IDs en labels que NO están en features (ejemplos):")
        print(list(sorted(left_only))[:20])
        print("\n[DIAGNÓSTICO] Algunos IDs en features que NO están en labels (ejemplos):")
        print(list(sorted(right_only))[:20])
        print("\nSugerencias:")
        print("- Revisa que tus carpetas de labels contengan los NOMBRES que corresponden a los videos procesados (mismo stem).")
        print("- El normalizador actual quita extensión, _15fps, espacios, (2), cambia guiones por underscores y pasa a minúsculas.")
        print("- Si aún no calzan, arma un labels.csv con columnas [video_id_norm,label] y cárgalo directo.")
        return

    print(f"[INFO] Labels: {Counter(data['label'])}")

    feat_cols = [
        "p95_left","p95_right","max_p95","min_p95","mean_p95",
        "miss_left_pct","miss_right_pct","max_miss","min_miss","mean_miss",
        "frames","fps","prop_c0","prop_c1","prop_c2"
    ]
    feat_cols = [c for c in feat_cols if c in data.columns]
    print(f"[INFO] N features: {len(feat_cols)} -> {feat_cols}")

    # 4) Split estratificado
    train_df, val_df = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["label"]
    )
    X_tr = train_df[feat_cols].values
    y_tr = train_df["label"].values
    X_va = val_df[feat_cols].values
    y_va = val_df["label"].values

    # 5) Modelo
    clf = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        random_state=42
    )
    clf.fit(X_tr, y_tr)

    # 6) Eval
    y_va_pred = clf.predict(X_va)
    print("\n=== Classification report (VAL) ===")
    print(classification_report(y_va, y_va_pred, digits=3))
    print("=== Confusion matrix (VAL) ===")
    print(confusion_matrix(y_va, y_va_pred, labels=["rojo","amarillo","verde"]))

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_va)
        classes = list(clf.classes_)
        y_true_bin = pd.get_dummies(pd.Categorical(y_va, categories=classes)).values
        try:
            auc = roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr")
            print(f"ROC-AUC macro: {auc:.3f}")
        except Exception:
            pass

    # 7) Guardar
    joblib.dump({"model": clf, "feat_cols": feat_cols, "classes": clf.classes_}, MODEL_OUT)
    data.to_csv(FEATURES_OUT, index=False)
    print(f"\n[OK] Modelo guardado -> {os.path.relpath(MODEL_OUT, BASE_PATH)}")
    print(f"[OK] Features usadas -> {os.path.relpath(FEATURES_OUT, BASE_PATH)}")

if __name__ == "__main__":
    main()
