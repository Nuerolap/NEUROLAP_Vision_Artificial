import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_CSV   = os.path.join(BASE_PATH, "artifacts", "window_embeddings.csv")
MANIFEST  = os.path.join(BASE_PATH, "datasets", "kp_v1", "_manifest_clean.csv")
OUT_DIR   = os.path.join(BASE_PATH, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- carga -------------
emb = pd.read_csv(EMB_CSV)
z_cols = [c for c in emb.columns if c.startswith("z")]
Z = emb[z_cols].astype(np.float32).values  # <- forzar float32

# ------------- KMeans -------------
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10)
emb["cluster"] = km.fit_predict(Z)

sil = silhouette_score(Z, emb["cluster"])
print(f"[KMEANS] k={k} | silhouette={sil:.3f}")

# ------------- asignar nombres R/A/V (heurístico por tamaño)
sizes = emb.groupby("cluster").size().sort_values(ascending=False).index.tolist()
cluster_to_rag = {sizes[0]:"verde", sizes[1]:"amarillo", sizes[2]:"rojo"}
emb["rag_cluster"] = emb["cluster"].map(cluster_to_rag)

# ------------- resumen por video/hand (moda de R/A/V por mano)
agg = emb.groupby(["video_id","hand"])["rag_cluster"] \
         .agg(lambda s: s.value_counts().idxmax()).reset_index()
pivot = agg.pivot(index="video_id", columns="hand", values="rag_cluster").reset_index()
pivot.columns.name = None

# ------------- proporciones por cluster por video (robusto, sin choque de 'video_id')
props_series = emb.groupby(["video_id", "cluster"]).size()
props_norm   = props_series.groupby(level=0).apply(lambda s: s / s.sum())
props_w = props_norm.unstack(fill_value=0.0)  # index=video_id, cols=cluster
props_w.columns = [f"prop_c{int(c)}" for c in props_w.columns]
props_w = props_w.copy()
props_w["video_id"] = props_w.index      # <- NO reset_index(names="video_id")
props_w = props_w.reset_index(drop=True) # <- no intenta insertar otra 'video_id'

# ------------- combina con miss% del manifest -------------
man = pd.read_csv(MANIFEST)
# Manifest trae 'video' como nombre de archivo con _15fps.mp4
man["video_id"] = man["video"].str.replace("_15fps.mp4","", regex=False)

miss = man[["video","miss_left_pct","miss_right_pct","csv_path","video_id"]].copy()

# Merge de todo: moda por mano + proporciones de clusters + métricas de miss
res = pivot.merge(props_w, on="video_id", how="left") \
           .merge(miss, on="video_id", how="left")

out_csv = os.path.join(OUT_DIR, "kmeans_video_summary.csv")
res.to_csv(out_csv, index=False)
print(f"[OK] resumen por video -> {os.path.relpath(out_csv, BASE_PATH)}")

# ------------- export también predicción por ventana -------------
win_out = os.path.join(OUT_DIR, "kmeans_window_assignments.csv")
emb.to_csv(win_out, index=False)
print(f"[OK] asignaciones por ventana -> {os.path.relpath(win_out, BASE_PATH)}")
# Guardar el modelo KMeans para usar en inferencia
import joblib
joblib.dump(km, os.path.join(OUT_DIR, "kmeans_model.joblib"))
print(f"[OK] kmeans model -> {os.path.join(OUT_DIR, 'kmeans_model.joblib')}")
