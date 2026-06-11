import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .common import artifacts_v2_dir, ensure_dir, resolve_source_path, utc_now_iso, videos_base, write_json


SELECTED_LANDMARKS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
PALM_LANDMARKS = [5, 9, 13, 17]
HAND_NAMES = ("left", "right")
META_COLUMNS = [
    "sample_id",
    "group_id",
    "content_group_id",
    "label",
    "source_set",
    "original_name",
    "normalized_stem",
    "duplicate_status",
    "content_duplicate_status",
    "source_path",
]


def safe_scale(xy: np.ndarray) -> np.ndarray:
    wrist = xy[:, 0, :]
    palm = xy[:, PALM_LANDMARKS, :]
    dist = np.linalg.norm(palm - wrist[:, None, :], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = np.nanmedian(dist, axis=1)
    return np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)


def interp_channel(values: np.ndarray, present: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    present = np.asarray(present, dtype=bool) & np.isfinite(values)
    if values.size == 0 or not np.any(present):
        return np.zeros(len(target_x), dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return np.interp(target_x, src_x[present], values[present]).astype(np.float32)


def hand_channels(coords: np.ndarray, present: np.ndarray) -> tuple[list[str], np.ndarray]:
    present = np.asarray(present, dtype=bool)
    if coords.size == 0:
        return [], np.empty((0, 0), dtype=np.float32)
    finite = np.isfinite(coords[:, 0, 0])
    present = present & finite
    xy = coords[:, :, :2].astype(np.float32)
    z = coords[:, :, 2].astype(np.float32)

    wrist_xy = np.where(present[:, None], xy[:, 0, :], 0.0)
    wrist_z = np.where(present, z[:, 0], 0.0)
    scale = safe_scale(xy)
    scale = np.where(present, scale, 1.0)

    names: list[str] = []
    channels: list[np.ndarray] = []
    for landmark in SELECTED_LANDMARKS:
        local_xy = (xy[:, landmark, :] - wrist_xy) / scale[:, None]
        local_z = (z[:, landmark] - wrist_z) / scale
        for axis_idx, axis in enumerate(("x", "y")):
            vals = np.where(present, local_xy[:, axis_idx], np.nan)
            names.append(f"lm{landmark}_n{axis}")
            channels.append(vals.astype(np.float32))
        vals_z = np.where(present, local_z, np.nan)
        names.append(f"lm{landmark}_nz")
        channels.append(vals_z.astype(np.float32))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroid = np.nanmean(xy, axis=1)
        x_min = np.nanmin(xy[:, :, 0], axis=1)
        x_max = np.nanmax(xy[:, :, 0], axis=1)
        y_min = np.nanmin(xy[:, :, 1], axis=1)
        y_max = np.nanmax(xy[:, :, 1], axis=1)
    global_channels = {
        "centroid_x": centroid[:, 0],
        "centroid_y": centroid[:, 1],
        "bbox_w": x_max - x_min,
        "bbox_h": y_max - y_min,
        "bbox_area": (x_max - x_min) * (y_max - y_min),
        "present": present.astype(np.float32),
    }
    for name, vals in global_channels.items():
        names.append(name)
        channels.append(np.where(present | (name == "present"), vals, np.nan).astype(np.float32))
    return names, np.vstack(channels)


def resample_hand(prefix: str, coords: np.ndarray, present: np.ndarray, steps: int) -> dict[str, float]:
    names, channels = hand_channels(coords, present)
    target_x = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    row: dict[str, float] = {}
    if channels.size == 0:
        return row
    present = np.asarray(present, dtype=bool)
    for name, values in zip(names, channels):
        interp = interp_channel(values, present, target_x)
        for step, value in enumerate(interp):
            row[f"seq{steps}_{prefix}_{name}_t{step:02d}"] = float(value)
        if name.startswith("lm"):
            velocity = np.diff(interp, prepend=interp[0])
            for step, value in enumerate(velocity):
                row[f"seq{steps}_{prefix}_{name}_d_t{step:02d}"] = float(value)
    return row


def build_row(row: pd.Series, base: Path, steps: int) -> dict:
    out = {col: row[col] for col in META_COLUMNS if col in row}
    out["sequence_status"] = "error"
    keypoints_path = str(row.get("keypoints_path", ""))
    if not keypoints_path:
        out["sequence_error"] = "missing_keypoints_path"
        return out
    path = resolve_source_path(keypoints_path, base)
    if not path.exists():
        out["sequence_error"] = f"missing_keypoints_file: {path}"
        return out
    try:
        data = np.load(path)
        for prefix in HAND_NAMES:
            coords = data[f"coords_{prefix}"].astype(np.float32)
            present = data[f"present_{prefix}"].astype(bool)
            out.update(resample_hand(prefix, coords, present, steps))
    except Exception as exc:
        out["sequence_error"] = str(exc)
        return out
    out["sequence_status"] = "ok"
    out["sequence_error"] = ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-length temporal sequence features from cached v2 keypoints.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=32)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v2_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "features_enriched.csv"
    out_csv = Path(args.out) if args.out else out_dir / f"sequence_features_{args.steps}.csv"
    df = pd.read_csv(features_csv)
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"].astype(str).str.lower() == "ok"].copy()

    rows = [build_row(row, base, args.steps) for _, row in df.iterrows()]
    seq = pd.DataFrame(rows)
    numeric_cols = seq.select_dtypes(include=[np.number, "bool"]).columns
    seq[numeric_cols] = seq[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ensure_dir(out_csv.parent)
    seq.to_csv(out_csv, index=False)
    write_json(
        out_dir / f"sequence_features_{args.steps}_metadata.json",
        {
            "generated_at": utc_now_iso(),
            "input": str(features_csv),
            "output": str(out_csv),
            "rows": int(len(seq)),
            "columns": int(len(seq.columns)),
            "steps": int(args.steps),
            "status": seq["sequence_status"].value_counts(dropna=False).to_dict(),
        },
    )
    print("=== V2 SEQUENCE FEATURES ===", flush=True)
    print(f"input: {features_csv}", flush=True)
    print(f"output: {out_csv}", flush=True)
    print(f"rows: {len(seq)} columns: {len(seq.columns)}", flush=True)
    print(seq["sequence_status"].value_counts(dropna=False).to_string(), flush=True)


if __name__ == "__main__":
    main()
