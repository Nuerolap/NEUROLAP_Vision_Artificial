import os
import argparse
import glob
import math
import json
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict

# =========================
# Config por defecto
# =========================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(BASE_PATH, "datasets", "kp_v1")
MANIFEST  = os.path.join(DATA_DIR, "_manifest.csv")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")   # opcional: video_id,label
OUT_DIR   = os.path.join(BASE_PATH, "artifacts")
os.makedirs(OUT_DIR, exist_ok=True)

# Señales
FEATS_PER_LM = 3        # x,y,z
N_LM = 21               # 21 landmarks por mano
FEATS_PER_FRAME = FEATS_PER_LM * N_LM  # 63
WINDOW = 90             # frames por ventana (6s @ 15fps)
STRIDE = 15             # solape (1s paso)
MIN_COVERAGE = 0.6      # % de puntos presentes por ventana
OUTLIER_MISS_PCT = 50.0 # filtra videos con miss% por mano > 50

RANDOM_SEED = 42

# =========================
# Utilidades
# =========================
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    # normaliza separadores
    df["csv_path"] = df["csv_path"].str.replace("\\\\", "/")
    return df

def filter_outliers(df: pd.DataFrame, outlier_thresh=OUTLIER_MISS_PCT) -> pd.DataFrame:
    # Mantener videos donde al menos una mano es razonable (<= outlier_thresh)
    keep = (df["miss_left_pct"] <= outlier_thresh) | (df["miss_right_pct"] <= outlier_thresh)
    return df[keep].reset_index(drop=True)

def load_video_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # columnas esperadas:
    # video_id, hand, frame_idx, t_sec, x0,y0,z0,...,x20,y20,z20, m0..m20
    return df

def per_video_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """ Normaliza x,y,z por video y mano, usando solo puntos presentes (mask=1). """
    coord_cols = [f"{a}{i}" for i in range(N_LM) for a in ("x","y","z")]
    mask_cols  = [f"m{i}" for i in range(N_LM)]
    # Para cada mano por video
    out = []
    for (hand), g in df.groupby("hand"):
        M = g[mask_cols].values  # [T,21]
        X = g[coord_cols].values # [T,63]
        # mascarilla expandida a 63 (x,y,z por landmark)
        mask63 = np.repeat(M, FEATS_PER_LM, axis=1).astype(bool)
        if mask63.sum() > 0:
            mu = (X[mask63].mean()).astype(np.float32)
            # media por feature -> vector length 63
            # para std, evitamos dividir por cero
            std = X[mask63].std().astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std)
            Xn = (X - mu) / std
        else:
            # sin puntos presentes: deja tal cual
            Xn = X
        g2 = g.copy()
        g2[coord_cols] = Xn
        out.append(g2)
    return pd.concat(out, axis=0).sort_values(["hand","frame_idx"]).reset_index(drop=True)

def build_windows_from_df(df: pd.DataFrame, window=WINDOW, stride=STRIDE, min_coverage=MIN_COVERAGE):
    coord_cols = [f"{a}{i}" for i in range(N_LM) for a in ("x","y","z")]
    mask_cols  = [f"m{i}" for i in range(N_LM)]

    windows = []
    # por mano
    for hand, g in df.groupby("hand"):
        g = g.sort_values("frame_idx").reset_index(drop=True)
        T = len(g)
        if T < window:
            continue
        X = g[coord_cols].values.astype(np.float32)  # [T,63]
        M = g[mask_cols].values.astype(np.float32)   # [T,21]

        # imputación simple: deja 0 (ya viene 0 cuando falta); la máscara lo indica
        # opcional: podrías hacer forward-fill limitado aquí

        for start in range(0, T - window + 1, stride):
            end = start + window
            x_win = X[start:end]        # [W,63]
            m_win = M[start:end]        # [W,21]

            # cobertura = porcentaje de puntos con mask=1 en la ventana
            coverage = m_win.mean()
            if coverage < min_coverage:
                continue

            windows.append({
                "hand": hand,
                "start": int(start),
                "end": int(end),
                "X": x_win,              # [W,63]
                "M": m_win,              # [W,21]
            })
    return windows

def attach_labels(windows, video_id, labels_map):
    for w in windows:
        w["video_id"] = video_id
        w["y"] = labels_map.get(video_id, None)
    return windows

# =========================
# Dataset
# =========================
class WindowDataset(Dataset):
    def __init__(self, items, supervised=False, label_to_idx=None):
        self.items = items
        self.supervised = supervised
        self.label_to_idx = label_to_idx or {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        X = it["X"].transpose(1,0)   # [63, W] para Conv1d
        M = it["M"].transpose(1,0)   # [21, W]
        sample = {
            "X": torch.tensor(X, dtype=torch.float32),
            "M": torch.tensor(M, dtype=torch.float32),
            "video_id": it.get("video_id",""),
            "hand": it["hand"],
        }
        if self.supervised:
            y = it.get("y", None)
            if y is None:
                # etiqueta faltante -> ignora (puedes filtrar antes)
                y_idx = -1
            else:
                y_idx = self.label_to_idx[y]
            sample["y"] = torch.tensor(y_idx, dtype=torch.long)
        return sample

# =========================
# Modelos
# =========================
class TemporalCNN(nn.Module):
    """ Clasificador temporal 1D sencillo. """
    def __init__(self, in_ch=FEATS_PER_FRAME, n_classes=3, seq_len=WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),    # global average pooling
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x, m=None):
        # x: [B,63,W]
        h = self.net(x).squeeze(-1)  # [B,128]
        return self.head(h)

class TemporalAutoencoder(nn.Module):
    """ Autoencoder temporal para aprender embeddings. Reconstruye x con MSE en puntos presentes. """
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
        # x: [B,63,W]
        h = self.enc(x).squeeze(-1)      # [B,128]
        z = self.fc_mu(h)                # [B,BOT]
        d = self.dec_in(z).unsqueeze(-1) # [B,128,1]
        # repeat a lo largo de W para reconstruir serie
        W = x.size(-1)
        d = d.repeat(1,1,W)              # [B,128,W]
        xr = self.dec(d)                 # [B,63,W]
        return xr, z

# =========================
# Entrenamiento
# =========================
def train_supervised(train_items, val_items, labels_map):
    classes = sorted(list(set([y for y in labels_map.values()])))
    label_to_idx = {c:i for i,c in enumerate(classes)}
    print(f"[INFO] Clases: {label_to_idx}")

    train_items = [it for it in train_items if it.get("y") is not None]
    val_items   = [it for it in val_items   if it.get("y") is not None]

    tr_ds = WindowDataset(train_items, supervised=True, label_to_idx=label_to_idx)
    va_ds = WindowDataset(val_items, supervised=True, label_to_idx=label_to_idx)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    model = TemporalCNN(in_ch=FEATS_PER_FRAME, n_classes=len(classes), seq_len=WINDOW)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(20):
        model.train()
        tr_loss, tr_ok, tr_tot = 0.0, 0, 0
        for batch in tr_loader:
            x = batch["X"]   # [B,63,W]
            y = batch["y"]   # [B]
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss) * x.size(0)
            pred = logits.argmax(dim=1)
            tr_ok += (pred == y).sum().item()
            tr_tot += x.size(0)

        model.eval()
        va_ok, va_tot = 0, 0
        with torch.no_grad():
            for batch in va_loader:
                x = batch["X"]; y = batch["y"]
                logits = model(x)
                pred = logits.argmax(dim=1)
                va_ok += (pred == y).sum().item()
                va_tot += x.size(0)
        tr_acc = tr_ok / max(1,tr_tot)
        va_acc = va_ok / max(1,va_tot)
        print(f"[SUP] epoch {epoch+1:02d} | loss {tr_loss/tr_tot:.4f} | acc_tr {tr_acc:.3f} | acc_val {va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(),
                        "label_to_idx": label_to_idx},
                       os.path.join(OUT_DIR, "supervised_cnn1d.pt"))
    print(f"[SUP] best val acc: {best_val:.3f}")

def train_autoencoder(train_items, val_items):
    tr_ds = WindowDataset(train_items, supervised=False)
    va_ds = WindowDataset(val_items, supervised=False)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

    model = TemporalAutoencoder(in_ch=FEATS_PER_FRAME, bottleneck=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss(reduction="none")

    best_val = math.inf
    for epoch in range(20):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        for batch in tr_loader:
            x = batch["X"]   # [B,63,W]
            xr, z = model(x)
            # reconstrucción solo en puntos presentes:
            # expandir máscara 21->63
            m = batch["M"]                       # [B,21,W]
            m63 = m.repeat_interleave(3, dim=1)  # [B,63,W]
            loss_mat = mse(xr, x) * m63
            denom = m63.sum().clamp(min=1.0)
            loss = (loss_mat.sum() / denom)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss_sum += float(loss) * x.size(0)
            tr_n += x.size(0)

        model.eval()
        va_loss_sum, va_n = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                x = batch["X"]
                xr, z = model(x)
                m = batch["M"]
                m63 = m.repeat_interleave(3, dim=1)
                loss_mat = mse(xr, x) * m63
                denom = m63.sum().clamp(min=1.0)
                loss = (loss_mat.sum() / denom)
                va_loss_sum += float(loss) * x.size(0)
                va_n += x.size(0)
        print(f"[AE ] epoch {epoch+1:02d} | recon_tr {tr_loss_sum/max(1,tr_n):.4f} | recon_val {va_loss_sum/max(1,va_n):.4f}")

        if (va_loss_sum/max(1,va_n)) < best_val:
            best_val = (va_loss_sum/max(1,va_n))
            torch.save({"model": model.state_dict()},
                       os.path.join(OUT_DIR, "autoencoder_temporal.pt"))
    print(f"[AE ] best val recon: {best_val:.4f}")

    # Exporta embeddings por ventana (para clustering o un clasificador ligero downstream)
    emb_csv = os.path.join(OUT_DIR, "window_embeddings.csv")
    model.eval()
    rows = []
    with torch.no_grad():
        for split_name, ds in [("train", tr_ds), ("val", va_ds)]:
            loader = DataLoader(ds, batch_size=128, shuffle=False)
            for batch in loader:
                x = batch["X"]
                _, z = model(x)
                z = z.cpu().numpy()  # [B,64]
                for i in range(z.shape[0]):
                    rows.append({
                        "split": split_name,
                        "video_id": batch["video_id"][i],
                        "hand": batch["hand"][i],
                        **{f"z{j}": float(z[i,j]) for j in range(z.shape[1])}
                    })
    pd.DataFrame(rows).to_csv(emb_csv, index=False)
    print(f"[AE ] embeddings -> {os.path.relpath(emb_csv, BASE_PATH)}")

# =========================
# Main
# =========================
def main(args):
    set_seed()

    dfm = load_manifest(args.manifest)
    dfm_filtered = filter_outliers(dfm, outlier_thresh=args.outlier_thresh)
    print(f"[INFO] Videos totales: {len(dfm)} | tras filtro outliers: {len(dfm_filtered)}")

    # Cargar labels si hay
    labels_map = {}
    supervised = False
    if args.labels and os.path.exists(args.labels):
        lab = pd.read_csv(args.labels)
        # columnas: video_id,label
        labels_map = dict(zip(lab["video_id"], lab["label"]))
        supervised = True
        print(f"[INFO] Modo SUPERVISADO con {len(labels_map)} etiquetas.")
    else:
        print("[INFO] Modo AUTOENCODER (sin labels.csv).")

    # Construcción de ventanas de TODOS los videos
    all_windows = []
    for _, row in dfm_filtered.iterrows():
        csv_path = os.path.join(BASE_PATH, row["csv_path"]).replace("\\","/")
        if not os.path.exists(csv_path):
            print(f"[WARN] No existe {csv_path}")
            continue
        vdf = load_video_csv(csv_path)
        # normalización por video (y por mano)
        vdf = per_video_normalize(vdf)
        # ventanas
        wins = build_windows_from_df(vdf, window=args.window, stride=args.stride, min_coverage=args.min_coverage)
        # adjuntar labels si hay
        wins = attach_labels(wins, vdf["video_id"].iloc[0], labels_map)
        all_windows.extend(wins)

    print(f"[INFO] Ventanas totales construidas: {len(all_windows)}")
    if supervised:
        # descarta ventanas sin etiqueta (si algún video carece de label)
        all_windows = [w for w in all_windows if w.get("y") is not None]
        print(f"[INFO] Ventanas con etiqueta: {len(all_windows)}")

    # Split train/val por video (no mezclar ventanas del mismo video en splits distintos)
    vids = sorted(list(set([w["video_id"] for w in all_windows])))
    tr_vids, va_vids = train_test_split(vids, test_size=0.2, random_state=RANDOM_SEED)
    train_items = [w for w in all_windows if w["video_id"] in tr_vids]
    val_items   = [w for w in all_windows if w["video_id"] in va_vids]
    print(f"[INFO] train windows: {len(train_items)} | val windows: {len(val_items)}")

    if supervised:
        train_supervised(train_items, val_items, labels_map)
    else:
        train_autoencoder(train_items, val_items)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=MANIFEST, help="Ruta a _manifest.csv")
    parser.add_argument("--labels",   type=str, default=LABELS_CSV, help="Ruta a labels.csv (opcional)")
    parser.add_argument("--outlier_thresh", type=float, default=OUTLIER_MISS_PCT,
                        help="Umbral %miss por mano para filtrar videos (>= se considera outlier)")
    parser.add_argument("--window", type=int, default=WINDOW, help="Frames por ventana")
    parser.add_argument("--stride", type=int, default=STRIDE, help="Stride de ventanas")
    parser.add_argument("--min_coverage", type=float, default=MIN_COVERAGE,
                        help="Cobertura mínima promedio de landmarks en la ventana (0-1)")
    args = parser.parse_args()
    main(args)
