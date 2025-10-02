# train_supervised_quality.py
import os
import re
import joblib
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ==== Rutas base ====
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR   = os.path.join(BASE_PATH, "artifacts")
DATA_DIR  = os.path.join(BASE_PATH, "datasets", "kp_v1")

RECON_CSV   = os.path.join(ART_DIR, "reconstruction_by_video_hand.csv")   # p95 por video/hand
EMB_WIN_CSV = os.path.join(ART_DIR, "kmeans_window_assignments.csv")      # cluster por ventana
MANIFEST    = os.path.join(DATA_DIR, "_manifest_clean.csv")                     # miss_% y paths

LABELS_DIR  = os.path.join(BASE_PATH, "labels")  # subcarpetas rojo/amarillo/verde
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
    b = re.sub(r'\.[A-Za-z0-9]+$', '', b)           # quitar extensión
    b = re.sub(r'(?i)[_-]?15fps$', '', b)           # quitar _15fps o -15fps al final
    b = b.replace('-', '_')
    b = b.replace(' ', '')
    b = re.sub(r'[\(\[]\s*\d+\s*[\)\]]$', '', b)    # quitar (2), [2], etc. al final
    return b.strip().lower()


# ---------- Carga de etiquetas desde carpetas ----------
def load_labels_from_folders(labels_dir: str) -> pd.DataFrame:
    rows = []
    for lbl in ("rojo", "amarillo", "verde"):
        d = os.path.join(labels_dir, lbl)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                vid_norm = normalize_id(f)
                rows.append({"video_id_norm": vid_norm, "label": lbl, "orig_file": f})
    df = pd.DataFrame(rows).drop_duplicates("video_id_norm")
    if df.empty:
        raise RuntimeError(f"No se encontraron etiquetas en {labels_dir}. Revisa rutas y extensiones.")
    print(f"[INFO] Etiquetas cargadas: {Counter(df['label'])}")
    return df


# ---------- Features de clusters (proporciones por clúster) ----------
def build_cluster_features(emb_win_csv: str) -> pd.DataFrame:
    if not os.path.exists(emb_win_csv):
        raise FileNotFoundError(f"No existe {emb_win_csv}")
    emb = pd.read_csv(emb_win_csv)
    need = {"video_id", "cluster"}
    if not need.issubset(set(emb.columns)):
        raise RuntimeError("kmeans_window_assignments.csv debe tener columnas (video_id, cluster).")

    emb["video_id_norm"] = emb["video_id"].apply(normalize_id)
    grp = emb.groupby(["video_id_norm", "cluster"]).size().reset_index(name="n")
    tot = grp.groupby("video_id_norm")["n"].sum().reset_index(name="N")
    m = grp.merge(tot, on="video_id_norm", how="left")
    m["prop"] = m["n"] / m["N"].replace(0, np.nan)

    pv = m.pivot(index="video_id_norm", columns="cluster", values="prop").fillna(0.0).reset_index()
    pv.columns.name = None

    # renombrar columnas numéricas -> prop_c0, prop_c1, ...
    rename = {}
    for c in pv.columns:
        if isinstance(c, (int, np.integer)):
            rename[c] = f"prop_c{c}"
    pv = pv.rename(columns=rename)

    # por si el KMeans no generó exactamente 3 labels (o están ausentes)
    for k in ("prop_c0", "prop_c1", "prop_c2"):
        if k not in pv.columns:
            pv[k] = 0.0

    return pv


# ---------- Features de señal (p95 por mano + miss% + frames/fps) ----------
def load_signal_features(recon_csv: str, manifest_csv: str) -> pd.DataFrame:
    if not os.path.exists(recon_csv):
        raise FileNotFoundError(f"No existe {recon_csv}")
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"No existe {manifest_csv}")

    r = pd.read_csv(recon_csv)  # columnas esperadas: video_id, hand, p95 (y quizá mean/median)
    need_r = {"video_id", "hand", "p95"}
    if not need_r.issubset(set(r.columns)):
        raise RuntimeError("reconstruction_by_video_hand.csv debe tener columnas (video_id, hand, p95).")

    r["video_id_norm"] = r["video_id"].apply(normalize_id)
    r["hand"] = r["hand"].str.title()  # Left / Right

    # si hay duplicados por (video_id_norm, hand), promediamos
    agg = (r.groupby(["video_id_norm", "hand"], as_index=False)
             .agg(p95=("p95", "mean")))

    # pivot a columnas p95_left / p95_right
    pv = agg.pivot(index="video_id_norm", columns="hand", values="p95").reset_index()
    pv.columns.name = None
    pv = pv.rename(columns={"Left": "p95_left", "Right": "p95_right"})

    # manifest para miss_% y meta
    man = pd.read_csv(manifest_csv)
    if "video" not in man.columns:
        raise RuntimeError("_manifest_clean.csv debe tener columna 'video'.")
    man["video_id_norm"] = man["video"].apply(normalize_id)

    keep = ["video_id_norm", "miss_left_pct", "miss_right_pct", "frames", "fps"]
    keep = [c for c in keep if c in man.columns]
    man = man[keep].copy()

    df = pv.merge(man, on="video_id_norm", how="left")

    # derivadas
    df["max_p95"]  = df[["p95_left", "p95_right"]].max(axis=1, skipna=True)
    df["min_p95"]  = df[["p95_left", "p95_right"]].min(axis=1, skipna=True)
    df["mean_p95"] = df[["p95_left", "p95_right"]].mean(axis=1, skipna=True)

    if "miss_left_pct" in df.columns and "miss_right_pct" in df.columns:
        df["max_miss"]  = df[["miss_left_pct", "miss_right_pct"]].max(axis=1, skipna=True)
        df["min_miss"]  = df[["miss_left_pct", "miss_right_pct"]].min(axis=1, skipna=True)
        df["mean_miss"] = df[["miss_left_pct", "miss_right_pct"]].mean(axis=1, skipna=True)
    else:
        # si por alguna razón no están, crea columnas a 0
        for c in ("miss_left_pct", "miss_right_pct", "max_miss", "min_miss", "mean_miss"):
            df[c] = 0.0

    return df


def main():
    # 1) Etiquetas desde carpetas
    y_df = load_labels_from_folders(LABELS_DIR)   # video_id_norm, label

    # 2) Features
    sig = load_signal_features(RECON_CSV, MANIFEST)   # p95_*, miss_*, frames, fps
    clu = build_cluster_features(EMB_WIN_CSV)         # prop_c*
    X = sig.merge(clu, on="video_id_norm", how="left")

    # Rellenar posibles NaN
    X = X.fillna(0.0)

    # 3) Unir con labels
    data = y_df.merge(X, on="video_id_norm", how="inner").dropna().reset_index(drop=True)

    print(f"[INFO] Total videos con label y features: {len(data)}")
    if len(data) == 0:
        left_only  = set(y_df["video_id_norm"]) - set(X["video_id_norm"])
        right_only = set(X["video_id_norm"]) - set(y_df["video_id_norm"])
        print("\n[DIAGNÓSTICO] IDs en labels que NO están en features (ejemplos):", list(sorted(left_only))[:20])
        print("[DIAGNÓSTICO] IDs en features que NO están en labels (ejemplos):", list(sorted(right_only))[:20])
        print("\nSugerencias:")
        print("- Revisa que los nombres en labels/ coincidan con los stem procesados (_15fps).")
        print("- Si no calzan, genera un labels.csv [video_id_norm,label] y únelo manualmente.")
        return

    print(f"[INFO] Labels: {Counter(data['label'])}")

    feat_cols = [
        "p95_left","p95_right","max_p95","min_p95","mean_p95",
        "miss_left_pct","miss_right_pct","max_miss","min_miss","mean_miss",
        "frames","fps","prop_c0","prop_c1","prop_c2"
    ]
    feat_cols = [c for c in feat_cols if c in data.columns]
    print(f"[INFO] N features: {len(feat_cols)} -> {feat_cols}")

    # Escalado simple (algunos modelos se benefician)
    scaler = StandardScaler()
    data[feat_cols] = scaler.fit_transform(data[feat_cols])

    # 4) Split estratificado con fallback
    label_counts = Counter(data["label"])
    can_stratify = all(v >= 2 for v in label_counts.values())
    test_size = 0.2 if len(data) >= 10 else 0.34  # si dataset chico, valida con 1/3

    if can_stratify:
        train_df, val_df = train_test_split(
            data, test_size=test_size, random_state=42, stratify=data["label"]
        )
    else:
        print("[WARN] Alguna clase tiene <2 muestras; usando split simple sin estratificar.")
        train_df, val_df = train_test_split(
            data, test_size=test_size, random_state=42
        )

    X_tr = train_df[feat_cols].values
    y_tr = train_df["label"].values
    X_va = val_df[feat_cols].values
    y_va = val_df["label"].values

    # 5) Modelo (GBDT estable para tabulares)
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
    print(classification_report(y_va, y_va_pred, digits=3, zero_division=0))
    print("=== Confusion matrix (VAL) ===")
    print(confusion_matrix(y_va, y_va_pred, labels=["rojo","amarillo","verde"]))

    if hasattr(clf, "predict_proba"):
        try:
            proba = clf.predict_proba(X_va)
            classes = list(clf.classes_)
            y_true_bin = pd.get_dummies(pd.Categorical(y_va, categories=classes)).values
            auc = roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr")
            print(f"ROC-AUC macro: {auc:.3f}")
        except Exception:
            pass

    # 7) Guardar modelo + columnas + scaler
    joblib.dump(
        {"model": clf, "feat_cols": feat_cols, "classes": clf.classes_, "scaler": scaler},
        MODEL_OUT
    )
    data.to_csv(FEATURES_OUT, index=False)
    print(f"\n[OK] Modelo guardado -> {os.path.relpath(MODEL_OUT, BASE_PATH)}")
    print(f"[OK] Features usadas -> {os.path.relpath(FEATURES_OUT, BASE_PATH)}")


if __name__ == "__main__":
    main()
