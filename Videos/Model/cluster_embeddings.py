import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EMB_CSV   = os.path.join(BASE_PATH, "artifacts", "window_embeddings.csv")
MANIFEST  = os.path.join(BASE_PATH, "datasets", "kp_v1", "_manifest.csv")
OUT_DIR   = os.path.join(BASE_PATH, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------- carga -------------
emb = pd.read_csv(EMB_CSV)  # cols: split, video_id, hand, z0..z63
z_cols = [c for c in emb.columns if c.startswith("z")]
Z = emb[z_cols].values

# ------------- KMeans -------------
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10)
emb["cluster"] = km.fit_predict(Z)

sil = silhouette_score(Z, emb["cluster"])
print(f"[KMEANS] k={k} | silhouette={sil:.3f}")

# ------------- asignar nombres R/A/V (heurístico)
# Ordena clusters por reconstr. error promedio si lo tuvieras; como no está,
# usa densidad (inercia por cluster) o tamaño. Aquí usaremos tamaño como ejemplo:
sizes = emb.groupby("cluster").size().sort_values(ascending=False).index.tolist()
# mayor tamaño -> asumimos "verde" (más común), luego "amarillo", luego "rojo"
cluster_to_rag = {sizes[0]:"verde", sizes[1]:"amarillo", sizes[2]:"rojo"}
emb["rag_cluster"] = emb["cluster"].map(cluster_to_rag)

# ------------- resumen por video/hand -------------
agg = emb.groupby(["video_id","hand"])["rag_cluster"].agg(lambda s: s.value_counts().idxmax()).reset_index()
pivot = agg.pivot(index="video_id", columns="hand", values="rag_cluster").reset_index()
pivot.columns.name = None

# ------------- combina con miss% del manifest -------------
man = pd.read_csv(MANIFEST)
man["video_id"] = man["video"].str.replace("_15fps.mp4","", regex=False)
man["video_csv"] = man["csv_path"]

# mapea miss% por mano a df de resumen
miss = man[["video","miss_left_pct","miss_right_pct","csv_path"]].copy()
miss["video_id"] = miss["video"].str.replace("_15fps.mp4","", regex=False)

res = pivot.merge(miss, on="video_id", how="left")
out_csv = os.path.join(OUT_DIR, "kmeans_video_summary.csv")
res.to_csv(out_csv, index=False)
print(f"[OK] resumen por video -> {os.path.relpath(out_csv, BASE_PATH)}")

# ------------- export también predicción por ventana -------------
win_out = os.path.join(OUT_DIR, "kmeans_window_assignments.csv")
emb.to_csv(win_out, index=False)
print(f"[OK] asignaciones por ventana -> {os.path.relpath(win_out, BASE_PATH)}")
