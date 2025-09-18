import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(BASE_PATH, "artifacts")
EMB_CSV = os.path.join(ART_DIR, "window_embeddings.csv")
CKPT    = os.path.join(ART_DIR, "autoencoder_temporal.pt")
MANIFEST= os.path.join(BASE_PATH, "datasets", "kp_v1", "_manifest.csv")

FEATS_PER_FRAME = 63
WINDOW = 90

# ====== modelo del AE (igual que en entrenamiento) ======
class TemporalAutoencoder(nn.Module):
    def __init__(self, in_ch=FEATS_PER_FRAME, bottleneck=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_mu = nn.Linear(128, bottleneck)
        self.dec_in = nn.Linear(bottleneck, 128)
        self.dec = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, in_ch, 3, padding=1)
        )

    def forward(self, x):
        h = self.enc(x).squeeze(-1)
        z = self.fc_mu(h)
        d = self.dec_in(z).unsqueeze(-1).repeat(1,1,x.size(-1))
        xr = self.dec(d)
        return xr, z

# ====== carga datos originales por ventana ======
# Volvemos a construir las ventanas para poder evaluar (mismo pipeline que train)
# Para ahorrar tiempo aquí asumimos que ya tienes MANIFEST+CSVs. Simplificamos:
def load_all_windows(manifest_csv):
    dfm = pd.read_csv(manifest_csv)
    dfm["csv_path"] = dfm["csv_path"].str.replace("\\\\","/", regex=False)
    rows = []
    for _, r in dfm.iterrows():
        csv_path = os.path.join(BASE_PATH, r["csv_path"]).replace("\\","/")
        if not os.path.exists(csv_path):
            continue
        vdf = pd.read_csv(csv_path)
        # columnas coord:
        coord_cols = [f"{a}{i}" for i in range(21) for a in ("x","y","z")]
        mask_cols  = [f"m{i}" for i in range(21)]
        for hand, g in vdf.groupby("hand"):
            g = g.sort_values("frame_idx")
            X = g[coord_cols].values.astype(np.float32)
            M = g[mask_cols].values.astype(np.float32)
            T = len(g)
            if T < WINDOW: 
                continue
            for s in range(0, T-WINDOW+1, 15):
                e = s + WINDOW
                x = X[s:e].T             # [63,90]
                m = M[s:e].T             # [21,90]
                rows.append({
                    "video_id": g["video_id"].iloc[0],
                    "hand": hand,
                    "X": x,
                    "M": m
                })
    return rows

def main():
    device = "cpu"
    # Carga AE
    ck = torch.load(CKPT, map_location=device)
    model = TemporalAutoencoder(in_ch=FEATS_PER_FRAME, bottleneck=64)
    model.load_state_dict(ck["model"])
    model.eval()

    # Reconstrucción por ventana
    windows = load_all_windows(MANIFEST)
    print(f"[INFO] ventanas a puntuar: {len(windows)}")

    mse = nn.MSELoss(reduction="none")
    out_rows = []
    for w in windows:
        x = torch.tensor(w["X"]).unsqueeze(0)  # [1,63,90]
        xr, _ = model(x)
        # máscara 21->63
        m = torch.tensor(w["M"]).unsqueeze(0)  # [1,21,90]
        m63 = m.repeat_interleave(3, dim=1)
        loss_mat = mse(xr, x) * m63
        denom = m63.sum().clamp(min=1.0)
        recon = (loss_mat.sum() / denom).item()
        out_rows.append({
            "video_id": w["video_id"],
            "hand": w["hand"],
            "recon_error": recon
        })
    dfw = pd.DataFrame(out_rows)

    # Agrega percentiles por video/hand
    agg = dfw.groupby(["video_id","hand"])["recon_error"].agg(["mean","median",lambda s: np.percentile(s,95)]).reset_index()
    agg = agg.rename(columns={"<lambda_0>":"p95"})
    agg.to_csv(os.path.join(ART_DIR, "reconstruction_by_video_hand.csv"), index=False)
    print("[OK] reconstruction_by_video_hand.csv listo")

    # Mezcla con miss% del manifest para semáforo simple
    man = pd.read_csv(MANIFEST)
    man["video_id"] = man["video"].str.replace("_15fps.mp4","", regex=False)
    miss = man[["video","video_id","miss_left_pct","miss_right_pct"]]

    # score por video: max de p95 entre manos + ponderación por miss
    # (reglas simples, calibra luego con datos etiquetados)
    pv = agg.pivot(index="video_id", columns="hand", values="p95").reset_index()
    pv.columns.name = None
    pv = pv.rename(columns={"Left":"p95_left", "Right":"p95_right"})
    pv = pv.merge(miss, on="video_id", how="left")
    # score: (max p95)*0.8 + (max miss pct/100)*0.2
    pv["max_p95"] = pv[["p95_left","p95_right"]].max(axis=1)
    pv["max_miss"] = (pv[["miss_left_pct","miss_right_pct"]].max(axis=1))/100.0
    pv["score"] = pv["max_p95"]*0.8 + pv["max_miss"]*0.2

    # umbrales heurísticos (ajústalos)
    # bajo score => VERDE, medio => AMARILLO, alto => ROJO
    t1, t2 = pv["score"].quantile([0.33, 0.66]).values
    def to_rag(s):
        if s <= t1: return "verde"
        if s <= t2: return "amarillo"
        return "rojo"
    pv["rag_heuristic"] = pv["score"].apply(to_rag)

    out2 = os.path.join(ART_DIR, "video_quality_scores.csv")
    pv.to_csv(out2, index=False)
    print(f"[OK] video_quality_scores.csv -> {os.path.relpath(out2, BASE_PATH)}")

if __name__ == "__main__":
    main()
