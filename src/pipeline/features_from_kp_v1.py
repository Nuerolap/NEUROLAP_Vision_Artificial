import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    artifacts_v2_dir,
    ensure_dir,
    normalize_stem,
    rel_to,
    strip_video_extension,
    utc_now_iso,
    videos_base,
)
from .extract_features import BASE_METADATA_COLUMNS, N_LANDMARKS, hand_feature_row, save_keypoints, write_feature_output


COORD_COLS = [f"{axis}{idx}" for idx in range(N_LANDMARKS) for axis in ("x", "y", "z")]
MASK_COLS = [f"m{idx}" for idx in range(N_LANDMARKS)]


def processed_video_id(original_name: str) -> str:
    return f"{strip_video_extension(original_name)}_15fps"


def manifest_lookup(manifest: pd.DataFrame) -> dict[tuple[str, str], dict]:
    rows = {}
    for _, row in manifest.iterrows():
        label = str(row["label"]).strip().casefold()
        video_id = str(row["video_id"])
        rows[(label, video_id)] = row.to_dict()
    return rows


def hand_arrays(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[float]]:
    df = pd.read_csv(csv_path)
    needed = {"hand", "frame_idx", "t_sec", *COORD_COLS, *MASK_COLS}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise RuntimeError(f"{csv_path} missing columns: {missing[:8]}")

    frames = sorted(pd.to_numeric(df["frame_idx"], errors="coerce").dropna().astype(int).unique().tolist())
    frame_to_pos = {frame: idx for idx, frame in enumerate(frames)}
    n = len(frames)
    left = np.full((n, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    right = np.full((n, N_LANDMARKS, 3), np.nan, dtype=np.float32)
    present_left = np.zeros(n, dtype=bool)
    present_right = np.zeros(n, dtype=bool)
    t_seconds = np.zeros(n, dtype=np.float32)

    for _, row in df.iterrows():
        frame = int(row["frame_idx"])
        pos = frame_to_pos[frame]
        t_seconds[pos] = float(row["t_sec"])
        masks = row[MASK_COLS].astype(float).values.reshape(N_LANDMARKS)
        present = bool(np.nanmean(masks) > 0.5)
        if not present:
            continue
        coords = row[COORD_COLS].astype(float).values.reshape(N_LANDMARKS, 3).astype(np.float32)
        hand = str(row["hand"]).strip().title()
        if hand == "Left":
            left[pos] = coords
            present_left[pos] = True
        elif hand == "Right":
            right[pos] = coords
            present_right[pos] = True

    return left, right, present_left, present_right, frames, t_seconds.astype(float).tolist()


def build_feature_row(index_row: pd.Series, manifest_row: dict, base: Path, keypoints_dir: Path, no_keypoint_cache: bool) -> dict:
    csv_path = base / str(manifest_row["csv_path"]).replace("\\", "/")
    left, right, present_left, present_right, frames, t_seconds = hand_arrays(csv_path)
    sample_id = str(index_row["sample_id"])
    keypoints_path = keypoints_dir / f"{sample_id}.npz"

    if not no_keypoint_cache:
        save_keypoints(keypoints_path, left, right, present_left, present_right, frames, t_seconds)

    any_hand = present_left | present_right
    both_hands = present_left & present_right
    miss_left = float(100.0 * (1.0 - present_left.mean())) if len(present_left) else 100.0
    miss_right = float(100.0 * (1.0 - present_right.mean())) if len(present_right) else 100.0

    row = {col: index_row[col] for col in BASE_METADATA_COLUMNS if col in index_row}
    row.update(
        {
            "processed_at": utc_now_iso(),
            "feature_source": "kp_v1_import",
            "target_fps": 15.0,
            "target_width": 640,
            "target_height": 360,
            "model_complexity": "",
            "keypoints_path": "" if no_keypoint_cache else rel_to(keypoints_path, base),
            "extraction_status": "ok",
            "error": "",
            "source_fps": float(manifest_row.get("fps", 15.0)),
            "source_frames": int(manifest_row.get("frames", len(frames))),
            "source_width": 640,
            "source_height": 360,
            "source_duration_s": float(manifest_row.get("frames", len(frames))) / max(float(manifest_row.get("fps", 15.0)), 1e-6),
            "frames_read": int(manifest_row.get("frames", len(frames))),
            "frames_used": int(len(frames)),
            "effective_sample_fps": float(manifest_row.get("fps", 15.0)),
            "miss_left_pct": miss_left,
            "miss_right_pct": miss_right,
            "max_miss": max(miss_left, miss_right),
            "min_miss": min(miss_left, miss_right),
            "mean_miss": (miss_left + miss_right) / 2.0,
            "any_hand_coverage": float(any_hand.mean()) if len(any_hand) else 0.0,
            "both_hands_coverage": float(both_hands.mean()) if len(both_hands) else 0.0,
            "hand_coverage_gap": float(abs(present_left.mean() - present_right.mean())) if len(present_left) else 0.0,
            "processing_seconds": 0.0,
        }
    )
    row.update(hand_feature_row("left", left, present_left))
    row.update(hand_feature_row("right", right, present_right))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Import legacy v2 features from existing datasets/kp_v1 keypoint CSVs.")
    parser.add_argument("--base", default=str(videos_base()), help="Videos directory.")
    parser.add_argument("--index", default=None, help="v2 video_index.csv path.")
    parser.add_argument("--manifest", default=None, help="datasets/kp_v1/_manifest_clean.csv path.")
    parser.add_argument("--out", default=None, help="features CSV output path.")
    parser.add_argument("--keypoints-dir", default=None, help="Compressed keypoint cache directory.")
    parser.add_argument("--source-set", default="legacy_raw", help="source_set to import from the v2 index.")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--no-keypoint-cache", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = artifacts_v2_dir(base)
    index_csv = Path(args.index) if args.index else out_dir / "video_index.csv"
    manifest_csv = Path(args.manifest) if args.manifest else base / "datasets" / "kp_v1" / "_manifest_clean.csv"
    out_csv = Path(args.out) if args.out else out_dir / "features.csv"
    keypoints_dir = Path(args.keypoints_dir) if args.keypoints_dir else out_dir / "keypoints"

    index = pd.read_csv(index_csv)
    selected = index[index["source_set"].astype(str) == args.source_set].copy()
    manifest = pd.read_csv(manifest_csv)
    lookup = manifest_lookup(manifest)

    rows = []
    missing = []
    for _, index_row in selected.iterrows():
        video_id = processed_video_id(str(index_row["original_name"]))
        key = (str(index_row["label"]).strip().casefold(), video_id)
        manifest_row = lookup.get(key)
        if manifest_row is None:
            alt_matches = manifest[
                (manifest["label"].astype(str).str.strip().str.casefold() == key[0])
                & (manifest["video"].map(normalize_stem) == str(index_row["normalized_stem"]))
            ]
            if not alt_matches.empty:
                manifest_row = alt_matches.iloc[0].to_dict()
        if manifest_row is None:
            missing.append({"sample_id": index_row["sample_id"], "label": index_row["label"], "original_name": index_row["original_name"]})
            continue
        try:
            rows.append(build_feature_row(index_row, manifest_row, base, keypoints_dir, args.no_keypoint_cache))
        except Exception as exc:
            failed = {col: index_row[col] for col in BASE_METADATA_COLUMNS if col in index_row}
            failed.update({"processed_at": utc_now_iso(), "feature_source": "kp_v1_import", "extraction_status": "error", "error": str(exc)})
            rows.append(failed)

    existing = pd.DataFrame()
    if out_csv.exists() and not args.overwrite_output:
        existing = pd.read_csv(out_csv)
    write_feature_output(out_csv, existing, rows, args.overwrite_output)

    missing_csv = out_dir / "legacy_kp_v1_missing.csv"
    pd.DataFrame(missing).to_csv(missing_csv, index=False)
    ok_count = sum(1 for row in rows if row.get("extraction_status") == "ok")
    print("=== LEGACY KP_V1 IMPORT ===")
    print(f"selected: {len(selected)}")
    print(f"imported rows: {len(rows)}")
    print(f"ok rows: {ok_count}")
    print(f"missing mappings: {len(missing)} -> {missing_csv}")
    print(f"[OK] features -> {out_csv}")


if __name__ == "__main__":
    main()

