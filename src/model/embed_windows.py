# embed_windows.py
import os, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART_DIR   = os.path.join(BASE_PATH, "artifacts")
os.makedirs(ART_DIR, exist_ok=True)

FEATS_PER_FRAME = 63
N_LM = 21

# ====== Modelo idéntico al de entrenamiento (enc + dec) ======
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

def per_video_normalize(vdf: pd.DataFrame) -> pd.DataFrame:
    coord_cols = [f"{a}{i}" for i in range(N_LM) for a in ("x","y","z")]
    mask_cols  = [f"m{i}" for i in range(N_LM)]
    out = []
    for hand, g in vdf.groupby("hand"):
        M = g[mask_cols].values
        X = g[coord_cols].values
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

def build_windows(vdf, window, stride, min_cov):
    coord_cols = [f"{a}{i}" for i in range(N_LM) for a in ("x","y","z")]
    mask_cols  = [f"m{i}" for i in range(N_LM)]
    rows = []
    for hand, g in vdf.groupby("hand"):
        g = g.sort_values("frame_idx")
        X = g[coord_cols].values.astype(np.float32)
        M = g[mask_cols].values.astype(np.float32)
        T = len(g)
        if T < window: continue
        for s in range(0, T - window + 1, stride):
            e = s + window
            m_win = M[s:e]
            coverage = float(m_win.mean())
            if coverage < min_cov: continue
            rows.append({
                "video_id": g["video_id"].iloc[0],
                "hand": hand,
                "X": X[s:e].T  # [63, W]
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window", type=int, default=90)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--min_coverage", type=float, default=0.6)
    args = ap.parse_args()

    # cargar modelo completo (enc+dec) EXACTO al train
    ck = torch.load(args.ckpt, map_location="cpu")
    ae = TemporalAutoencoder(in_ch=FEATS_PER_FRAME, bottleneck=64)
    ae.load_state_dict(ck["model"])  # coincide 1:1 con el checkpoint
    ae.eval()

    dfm = pd.read_csv(args.manifest)
    rows_all = []
    with torch.no_grad():
        for _, r in dfm.iterrows():
            csv_path = os.path.join(BASE_PATH, str(r["csv_path"]).replace("\\","/"))
            if not os.path.exists(csv_path):
                print(f"[WARN] no existe: {csv_path}")
                continue
            vdf = pd.read_csv(csv_path)
            vdf = per_video_normalize(vdf)
            wins = build_windows(vdf, args.window, args.stride, args.min_coverage)
            for w in wins:
                xt = torch.tensor(w["X"], dtype=torch.float32).unsqueeze(0)  # [1,63,W]
                xr, z = ae(xt)
                z_np = z.squeeze(0).cpu().numpy()
                row = {"video_id": w["video_id"], "hand": w["hand"]}
                for i, val in enumerate(z_np):
                    row[f"z{i}"] = float(val)
                rows_all.append(row)

    emb = pd.DataFrame(rows_all)
    out_csv = os.path.join(ART_DIR, "window_embeddings.csv")
    emb.to_csv(out_csv, index=False)
    print(f"[OK] embeddings -> {out_csv}")

if __name__ == "__main__":
    main()
