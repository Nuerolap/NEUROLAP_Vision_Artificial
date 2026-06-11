import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from .common import (
    artifacts_v2_dir,
    ensure_dir,
    parse_csv_list,
    rel_to,
    resolve_source_path,
    utc_now_iso,
    videos_base,
)


N_LANDMARKS = 21
HAND_LABELS = ("Left", "Right")
BASE_METADATA_COLUMNS = {
    "sample_id",
    "group_id",
    "label",
    "source_set",
    "source_path",
    "original_name",
    "normalized_stem",
    "duplicate_status",
}


def resize_letterbox(frame: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    target_w, target_h = target
    h, w = frame.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def select_rows(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    work = df.copy()
    source_sets = set(parse_csv_list(args.source_set))
    if source_sets:
        work = work[work["source_set"].isin(source_sets)].copy()

    selected_indexes: set[int] = set()
    sample_ids = set(parse_csv_list(args.sample_ids))
    if sample_ids:
        selected_indexes.update(df.index[df["sample_id"].isin(sample_ids)].tolist())

    contains = parse_csv_list(args.include_name_contains)
    for needle in contains:
        mask = (
            work["original_name"].astype(str).str.contains(needle, case=False, regex=False)
            | work["normalized_stem"].astype(str).str.contains(needle, case=False, regex=False)
            | work["sample_id"].astype(str).str.contains(needle, case=False, regex=False)
        )
        selected_indexes.update(work.index[mask].tolist())

    exact_stems = set(parse_csv_list(args.include_normalized_stems))
    if exact_stems:
        selected_indexes.update(work.index[work["normalized_stem"].astype(str).isin(exact_stems)].tolist())

    if args.sample_per_label:
        sampled = work.sort_values(["label", "source_set", "normalized_stem"]).groupby("label", group_keys=False).head(args.sample_per_label)
        selected_indexes.update(sampled.index.tolist())

    if not selected_indexes:
        selected = work
    else:
        selected = df.loc[sorted(selected_indexes)].copy()

    selected = selected.sort_values(["source_set", "label", "normalized_stem", "sample_id"]).reset_index(drop=True)
    if args.limit:
        selected = selected.head(args.limit)
    return selected


def detected_hands(results: Any) -> dict[str, np.ndarray]:
    out: dict[str, tuple[float, np.ndarray]] = {}
    if not results or not results.multi_hand_landmarks:
        return {}

    handedness = results.multi_handedness or []
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        label = None
        score = 0.0
        if idx < len(handedness) and handedness[idx].classification:
            cls = handedness[idx].classification[0]
            label = cls.label
            score = float(cls.score or 0.0)
        if label not in HAND_LABELS:
            label = f"Unknown{idx}"
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        if label in HAND_LABELS and (label not in out or score >= out[label][0]):
            out[label] = (score, coords)
    return {label: coords for label, (_, coords) in out.items()}


def _safe_float(value: float) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def _summary(values: np.ndarray, fn: str, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    if fn == "mean":
        return _safe_float(float(np.mean(values)))
    if fn == "std":
        return _safe_float(float(np.std(values)))
    if fn == "p95":
        return _safe_float(float(np.percentile(values, 95)))
    if fn == "max":
        return _safe_float(float(np.max(values)))
    raise ValueError(fn)


def hand_feature_row(prefix: str, coords: np.ndarray, present: np.ndarray) -> dict[str, float]:
    total = int(len(present))
    present_count = int(present.sum())
    coverage = present_count / max(total, 1)
    row: dict[str, float] = {
        f"{prefix}_coverage": float(coverage),
        f"{prefix}_present_frames": float(present_count),
        f"{prefix}_miss_pct": float(100.0 * (1.0 - coverage)),
    }

    if present_count == 0:
        for name in (
            "centroid_x_mean",
            "centroid_y_mean",
            "centroid_x_std",
            "centroid_y_std",
            "speed_mean",
            "speed_std",
            "speed_p95",
            "speed_max",
            "jitter_mean",
            "jitter_p95",
            "bbox_w_mean",
            "bbox_h_mean",
            "bbox_area_mean",
            "bbox_area_std",
            "bbox_area_p95",
            "spread_mean",
            "spread_std",
            "x_std_mean",
            "y_std_mean",
            "z_std_mean",
            "landmark_motion_mean",
            "landmark_motion_p95",
        ):
            row[f"{prefix}_{name}"] = 0.0
        return row

    valid = coords[present]
    xy = valid[:, :, :2]
    centroids = xy.mean(axis=1)
    x_min, y_min = xy.min(axis=1).T
    x_max, y_max = xy.max(axis=1).T
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_area = bbox_w * bbox_h
    centered = xy - centroids[:, None, :]
    spread = np.linalg.norm(centered, axis=2).mean(axis=1)

    row[f"{prefix}_centroid_x_mean"] = _summary(centroids[:, 0], "mean")
    row[f"{prefix}_centroid_y_mean"] = _summary(centroids[:, 1], "mean")
    row[f"{prefix}_centroid_x_std"] = _summary(centroids[:, 0], "std")
    row[f"{prefix}_centroid_y_std"] = _summary(centroids[:, 1], "std")
    row[f"{prefix}_bbox_w_mean"] = _summary(bbox_w, "mean")
    row[f"{prefix}_bbox_h_mean"] = _summary(bbox_h, "mean")
    row[f"{prefix}_bbox_area_mean"] = _summary(bbox_area, "mean")
    row[f"{prefix}_bbox_area_std"] = _summary(bbox_area, "std")
    row[f"{prefix}_bbox_area_p95"] = _summary(bbox_area, "p95")
    row[f"{prefix}_spread_mean"] = _summary(spread, "mean")
    row[f"{prefix}_spread_std"] = _summary(spread, "std")
    row[f"{prefix}_x_std_mean"] = _summary(valid[:, :, 0].std(axis=1), "mean")
    row[f"{prefix}_y_std_mean"] = _summary(valid[:, :, 1].std(axis=1), "mean")
    row[f"{prefix}_z_std_mean"] = _summary(valid[:, :, 2].std(axis=1), "mean")

    if present_count >= 2:
        deltas = np.diff(centroids, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)
        row[f"{prefix}_speed_mean"] = _summary(speeds, "mean")
        row[f"{prefix}_speed_std"] = _summary(speeds, "std")
        row[f"{prefix}_speed_p95"] = _summary(speeds, "p95")
        row[f"{prefix}_speed_max"] = _summary(speeds, "max")
        landmark_motion = np.linalg.norm(np.diff(valid[:, :, :2], axis=0), axis=2).mean(axis=1)
        row[f"{prefix}_landmark_motion_mean"] = _summary(landmark_motion, "mean")
        row[f"{prefix}_landmark_motion_p95"] = _summary(landmark_motion, "p95")
    else:
        for name in ("speed_mean", "speed_std", "speed_p95", "speed_max", "landmark_motion_mean", "landmark_motion_p95"):
            row[f"{prefix}_{name}"] = 0.0

    if present_count >= 3:
        accel = np.diff(np.diff(centroids, axis=0), axis=0)
        jitter = np.linalg.norm(accel, axis=1)
        row[f"{prefix}_jitter_mean"] = _summary(jitter, "mean")
        row[f"{prefix}_jitter_p95"] = _summary(jitter, "p95")
    else:
        row[f"{prefix}_jitter_mean"] = 0.0
        row[f"{prefix}_jitter_p95"] = 0.0

    return row


def save_keypoints(
    path: Path,
    left: np.ndarray,
    right: np.ndarray,
    present_left: np.ndarray,
    present_right: np.ndarray,
    frame_indices: list[int],
    t_seconds: list[float],
) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        coords_left=left.astype(np.float32),
        coords_right=right.astype(np.float32),
        present_left=present_left.astype(np.bool_),
        present_right=present_right.astype(np.bool_),
        frame_indices=np.array(frame_indices, dtype=np.int32),
        t_seconds=np.array(t_seconds, dtype=np.float32),
    )


def write_feature_output(out_csv: Path, existing: pd.DataFrame, rows: list[dict], overwrite_output: bool) -> None:
    if not rows and existing.empty:
        return
    new_df = pd.DataFrame(rows)
    combined = new_df if overwrite_output or existing.empty else pd.concat([existing, new_df], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("sample_id", keep="last") if "sample_id" in combined.columns else combined
    ensure_dir(out_csv.parent)
    combined.to_csv(out_csv, index=False)


def extract_one(row: pd.Series, hands: Any, args: argparse.Namespace, base: Path, keypoints_dir: Path) -> dict:
    started = time.perf_counter()
    metadata = {col: row[col] for col in BASE_METADATA_COLUMNS if col in row}
    sample_id = str(row["sample_id"])
    source_path = resolve_source_path(row["source_path"], base)
    keypoints_path = keypoints_dir / f"{sample_id}.npz"

    result = {
        **metadata,
        "processed_at": utc_now_iso(),
        "target_fps": float(args.target_fps),
        "target_width": int(args.width),
        "target_height": int(args.height),
        "model_complexity": int(args.model_complexity),
        "keypoints_path": rel_to(keypoints_path, base),
        "extraction_status": "error",
        "error": "",
    }

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        result["error"] = f"could not open video: {source_path}"
        return result

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0 or not math.isfinite(source_fps):
        source_fps = 30.0
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_s = float(source_frames / source_fps) if source_fps > 0 else 0.0

    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    sampled_frame_indices: list[int] = []
    sampled_t_seconds: list[float] = []

    sample_interval = max(source_fps / max(float(args.target_fps), 1e-6), 1.0)
    next_sample_frame = 0.0
    frame_idx = 0
    frames_read = 0
    frames_used = 0

    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            frames_read += 1
            if frame_idx + 1e-9 < next_sample_frame:
                frame_idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok:
                break
            next_sample_frame += sample_interval

            frame_resized = resize_letterbox(frame, (int(args.width), int(args.height)))
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            detections = detected_hands(hands.process(rgb))

            missing = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
            left_rows.append(detections.get("Left", missing).astype(np.float32))
            right_rows.append(detections.get("Right", missing).astype(np.float32))
            sampled_frame_indices.append(frame_idx)
            sampled_t_seconds.append(frame_idx / source_fps)
            frames_used += 1

            if args.max_sampled_frames and frames_used >= args.max_sampled_frames:
                break
            frame_idx += 1
    finally:
        cap.release()

    if frames_used == 0:
        result["error"] = "no sampled frames"
        return result

    left = np.stack(left_rows, axis=0)
    right = np.stack(right_rows, axis=0)
    present_left = ~np.isnan(left[:, 0, 0])
    present_right = ~np.isnan(right[:, 0, 0])

    if not args.no_keypoint_cache:
        save_keypoints(keypoints_path, left, right, present_left, present_right, sampled_frame_indices, sampled_t_seconds)
    else:
        result["keypoints_path"] = ""

    left_features = hand_feature_row("left", left, present_left)
    right_features = hand_feature_row("right", right, present_right)
    any_hand = present_left | present_right
    both_hands = present_left & present_right

    result.update(
        {
            "source_fps": source_fps,
            "source_frames": source_frames,
            "source_width": source_width,
            "source_height": source_height,
            "source_duration_s": duration_s,
            "frames_read": frames_read,
            "frames_used": frames_used,
            "effective_sample_fps": float(frames_used / max(sampled_t_seconds[-1] - sampled_t_seconds[0], 1e-6))
            if len(sampled_t_seconds) > 1
            else float(args.target_fps),
            "miss_left_pct": float(100.0 * (1.0 - present_left.mean())),
            "miss_right_pct": float(100.0 * (1.0 - present_right.mean())),
            "max_miss": float(max(100.0 * (1.0 - present_left.mean()), 100.0 * (1.0 - present_right.mean()))),
            "min_miss": float(min(100.0 * (1.0 - present_left.mean()), 100.0 * (1.0 - present_right.mean()))),
            "mean_miss": float((100.0 * (1.0 - present_left.mean()) + 100.0 * (1.0 - present_right.mean())) / 2.0),
            "any_hand_coverage": float(any_hand.mean()),
            "both_hands_coverage": float(both_hands.mean()),
            "hand_coverage_gap": float(abs(present_left.mean() - present_right.mean())),
            "processing_seconds": float(time.perf_counter() - started),
            "extraction_status": "ok",
            "error": "",
        }
    )
    result.update(left_features)
    result.update(right_features)
    return result


def extract_one_worker(row_dict: dict, args_dict: dict, base_str: str, keypoints_dir_str: str) -> dict:
    args = argparse.Namespace(**args_dict)
    row = pd.Series(row_dict)
    base = Path(base_str)
    keypoints_dir = Path(keypoints_dir_str)
    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=int(args.model_complexity),
            min_detection_confidence=float(args.min_detection_confidence),
            min_tracking_confidence=float(args.min_tracking_confidence),
        ) as hands:
            return extract_one(row, hands, args, base, keypoints_dir)
    except Exception as exc:
        failed = {col: row[col] for col in BASE_METADATA_COLUMNS if col in row}
        failed.update(
            {
                "processed_at": utc_now_iso(),
                "extraction_status": "error",
                "error": str(exc),
            }
        )
        return failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract compact v2 keypoint features directly from source videos.")
    parser.add_argument("--base", default=str(videos_base()), help="Videos directory.")
    parser.add_argument("--index", default=None, help="video_index.csv path.")
    parser.add_argument("--out", default=None, help="features CSV output path.")
    parser.add_argument("--keypoints-dir", default=None, help="Compressed keypoint cache directory.")
    parser.add_argument("--source-set", nargs="*", default=None, help="Optional source_set filter, comma-separated or repeated.")
    parser.add_argument("--sample-ids", nargs="*", default=None, help="Optional sample_id filter, comma-separated or repeated.")
    parser.add_argument("--include-name-contains", nargs="*", default=None, help="Include matching original_name/stem/sample_id.")
    parser.add_argument("--include-normalized-stems", nargs="*", default=None, help="Include exact normalized_stem values.")
    parser.add_argument("--sample-per-label", type=int, default=0, help="Smoke mode: N videos per label.")
    parser.add_argument("--limit", type=int, default=0, help="Limit selected rows after filtering.")
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--model-complexity", type=int, choices=[0, 1], default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.55)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.80)
    parser.add_argument("--max-sampled-frames", type=int, default=0, help="Debug/smoke limit per video.")
    parser.add_argument("--flush-every", type=int, default=1, help="Write partial CSV every N processed videos.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes. Use 1 for deterministic serial logs.")
    parser.add_argument("--overwrite-output", action="store_true", help="Ignore existing features output.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-extract rows already present in output.")
    parser.add_argument("--no-keypoint-cache", action="store_true", help="Do not write .npz keypoint caches.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = artifacts_v2_dir(base)
    index_csv = Path(args.index) if args.index else out_dir / "video_index.csv"
    out_csv = Path(args.out) if args.out else out_dir / "features.csv"
    keypoints_dir = Path(args.keypoints_dir) if args.keypoints_dir else out_dir / "keypoints"

    if not index_csv.exists():
        raise FileNotFoundError(f"Missing index CSV: {index_csv}")

    index = pd.read_csv(index_csv)
    selected = select_rows(index, args)
    if selected.empty:
        print("No rows selected.")
        return

    existing = pd.DataFrame()
    done_ids: set[str] = set()
    if out_csv.exists() and not args.overwrite_output:
        existing = pd.read_csv(out_csv)
        if not args.no_skip_existing and "sample_id" in existing.columns:
            done_ids = set(existing["sample_id"].astype(str))
            selected = selected[~selected["sample_id"].astype(str).isin(done_ids)].copy()

    print("=== V2 FEATURE EXTRACTION ===", flush=True)
    print(f"index: {index_csv}", flush=True)
    print(f"output: {out_csv}", flush=True)
    print(f"selected: {len(selected)}", flush=True)
    print(f"already_done: {len(done_ids)}", flush=True)
    print(f"model_complexity: {args.model_complexity}", flush=True)

    rows: list[dict] = []
    if args.workers > 1:
        print(f"workers: {args.workers}", flush=True)
        args_dict = vars(args).copy()
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            futures = {
                executor.submit(extract_one_worker, row.to_dict(), args_dict, str(base), str(keypoints_dir)): row.to_dict()
                for _, row in selected.iterrows()
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                original = futures[future]
                result = future.result()
                rows.append(result)
                status = result.get("extraction_status", "unknown")
                print(
                    f"[{idx}/{len(selected)}] {status} {original['label']}/{original['original_name']} -> {original['sample_id']}",
                    flush=True,
                )
                if args.flush_every and idx % max(1, args.flush_every) == 0:
                    write_feature_output(out_csv, existing, rows, args.overwrite_output)
    else:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=int(args.model_complexity),
            min_detection_confidence=float(args.min_detection_confidence),
            min_tracking_confidence=float(args.min_tracking_confidence),
        ) as hands:
            for idx, (_, row) in enumerate(selected.iterrows(), start=1):
                print(f"[{idx}/{len(selected)}] {row['label']}/{row['original_name']} -> {row['sample_id']}", flush=True)
                try:
                    rows.append(extract_one(row, hands, args, base, keypoints_dir))
                except Exception as exc:
                    failed = {col: row[col] for col in BASE_METADATA_COLUMNS if col in row}
                    failed.update(
                        {
                            "processed_at": utc_now_iso(),
                            "extraction_status": "error",
                            "error": str(exc),
                        }
                    )
                    rows.append(failed)
                if args.flush_every and idx % max(1, args.flush_every) == 0:
                    write_feature_output(out_csv, existing, rows, args.overwrite_output)

    write_feature_output(out_csv, existing, rows, args.overwrite_output)
    combined = pd.read_csv(out_csv)
    new_df = pd.DataFrame(rows)
    ok_count = int((new_df.get("extraction_status") == "ok").sum()) if not new_df.empty else 0
    print(f"[OK] wrote {len(combined)} total rows -> {out_csv}", flush=True)
    print(f"[OK] new ok rows: {ok_count}/{len(new_df)}", flush=True)


if __name__ == "__main__":
    main()
