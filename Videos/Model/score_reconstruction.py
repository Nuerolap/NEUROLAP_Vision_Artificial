import os
import math
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR = os.path.join(BASE_PATH, "artifacts")
CKPT    = os.path.join(ART_DIR, "autoencoder_temporal.pt")
MANIFEST= os.path.join(BASE_PATH, "datasets", "kp_v1", "_manifest_clean.csv")

# Deben coincidir con train_video_model.py
FEATS_PER_FRAME = 63
N_LM = 21
WINDOW_DEFAULT = 90
STRIDE_DEFAULT = 15
MIN_COVERAGE_DEFAULT = 0.60

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

# ====== helpers para alinear con train ======
def per_video_normalize(vdf: pd.DataFrame) -> pd.DataFrame:
    coord_cols = [f"{a}{i}" for i in range(N_LM) for a in ("x","y","z")]
    mask_cols  = [f"m{i}" for i in range(N_LM)]
    out = []
    for hand, g in vdf.groupby("hand"):
        M = g[mask_cols].values  # [T,21]
        X = g[coord_cols].values # [T,63]
        mask63 = np.repeat(M, 3, axis=1).astype(bool)
        if mask63.sum() > 0:
            mu = X[mask63].mean().astype(np.float32)
            std = X[mask63].std().astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            Xn = (X - mu) / std
        else:
            Xn = X
        gg = g.copy()
        gg[coord_cols] = Xn
        out.append(gg)
    return pd.concat(out, axis=0).sort_values(["hand","frame_idx"]).reset_index(drop=True)

def build_windows(vdf: pd.DataFrame, window: int, stride: int, min_cov: float):
    coord_cols = [f"{a}{i}" for i in range(N_LM) for a in ("x","y","z")]
    mask_cols  = [f"m{i}" for i in range(N_LM)]
    rows = []
    for hand, g in vdf.groupby("hand"):
        g = g.sort_values("frame_idx")
        X = g[coord_cols].values.astype(np.float32)
        M = g[mask_cols].values.astype(np.float32)
        T = len(g)
        if T < window:
            continue
        for s in range(0, T - window + 1, stride):
            e = s + window
            m_win = M[s:e]                # [W,21]
            coverage = float(m_win.mean())
            if coverage < min_cov:
                continue
            rows.append({
                "video_id": g["video_id"].iloc[0],
                "hand": hand,
                "X": X[s:e].T,            # [63,W]
                "M": M[s:e].T,            # [21,W]
            })
    return rows

def load_all_windows(manifest_csv, window, stride, min_cov):
    dfm = pd.read_csv(manifest_csv)
    dfm["csv_path"] = dfm["csv_path"].str.replace("\\\\","/", regex=False)
    all_rows = []
    for _, r in dfm.iterrows():
        csv_path = os.path.join(BASE_PATH, r["csv_path"]).replace("\\","/")
        if not os.path.exists(csv_path):
            continue
        vdf = pd.read_csv(csv_path)
        # normalización por video/mano (igual que train)
        vdf = per_video_normalize(vdf)
        # construir ventanas con misma lógica de cobertura y stride
        all_rows.extend(build_windows(vdf, window, stride, min_cov))
    return all_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    ap.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    ap.add_argument("--min_coverage", type=float, default=MIN_COVERAGE_DEFAULT)
    args = ap.parse_args()

    device = "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = TemporalAutoencoder(in_ch=FEATS_PER_FRAME, bottleneck=64)
    model.load_state_dict(ck["model"])
    model.eval()

    windows = load_all_windows(args.manifest, args.window, args.stride, args.min_coverage)
    print(f"[INFO] ventanas a puntuar: {len(windows)}")

    mse = nn.MSELoss(reduction="none")
    out_rows = []
    for w in windows:
        x = torch.tensor(w["X"]).unsqueeze(0)  # [1,63,W]
        xr, _ = model(x)
        m = torch.tensor(w["M"]).unsqueeze(0)  # [1,21,W]
        m63 = m.repeat_interleave(3, dim=1)
        loss_mat = mse(xr, x) * m63
        denom = m63.sum().clamp(min=1.0)
        recon = (loss_mat.sum() / denom).item()
        out_rows.append({"video_id": w["video_id"], "hand": w["hand"], "recon_error": recon})
    dfw = pd.DataFrame(out_rows)

    # Agregados p95/mean/median por video/hand
    agg = dfw.groupby(["video_id","hand"])["recon_error"].agg(
        mean="mean", median="median", p95=lambda s: np.percentile(s,95)
    ).reset_index()
    agg.to_csv(os.path.join(ART_DIR, "reconstruction_by_video_hand.csv"), index=False)
    print("[OK] reconstruction_by_video_hand.csv listo")

    # Semáforo simple con miss% del manifest
    man = pd.read_csv(args.manifest)
    man["video_id"] = man["video"].str.replace("_15fps.mp4","", regex=False)
    miss = man[["video","video_id","miss_left_pct","miss_right_pct"]]

    pv = agg.pivot(index="video_id", columns="hand", values="p95").reset_index()
    pv.columns.name = None
    pv = pv.rename(columns={"Left":"p95_left", "Right":"p95_right"})
    pv = pv.merge(miss, on="video_id", how="left")

    # features auxiliares
    for col in ["p95_left","p95_right"]:
        if col not in pv: pv[col] = np.nan
    pv["max_p95"] = pv[["p95_left","p95_right"]].max(axis=1, skipna=True)
    pv["max_miss"] = (pv[["miss_left_pct","miss_right_pct"]].max(axis=1, skipna=True))/100.0
    pv["score"] = pv["max_p95"]*0.8 + pv["max_miss"]*0.2

    # umbrales heurísticos
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
