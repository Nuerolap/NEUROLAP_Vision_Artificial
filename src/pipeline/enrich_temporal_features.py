import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .common import artifacts_v2_dir, ensure_dir, rel_to, resolve_source_path, utc_now_iso, videos_base, write_json


SELECTED_LANDMARKS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
TIP_LANDMARKS = [4, 8, 12, 16, 20]
PALM_LANDMARKS = [0, 5, 9, 13, 17]
BIN_COUNT = 5
EDGE_THRESHOLDS = (0.02, 0.03, 0.05, 0.08, 0.10)
SUMMARY_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


def safe_float(value: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    if not np.isfinite(value):
        return default
    return value


def threshold_name(threshold: float) -> str:
    return f"{int(round(threshold * 100)):02d}"


def longest_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def add_summary(row: dict, prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        row[f"{prefix}_mean"] = 0.0
        row[f"{prefix}_std"] = 0.0
        row[f"{prefix}_min"] = 0.0
        row[f"{prefix}_max"] = 0.0
        row[f"{prefix}_range"] = 0.0
        for pct in SUMMARY_PERCENTILES:
            row[f"{prefix}_p{pct:02d}"] = 0.0
        return
    row[f"{prefix}_mean"] = safe_float(np.mean(values))
    row[f"{prefix}_std"] = safe_float(np.std(values))
    row[f"{prefix}_min"] = safe_float(np.min(values))
    row[f"{prefix}_max"] = safe_float(np.max(values))
    row[f"{prefix}_range"] = safe_float(np.max(values) - np.min(values))
    for pct in SUMMARY_PERCENTILES:
        row[f"{prefix}_p{pct:02d}"] = safe_float(np.percentile(values, pct))


def valid_present(coords: np.ndarray, present: np.ndarray) -> np.ndarray:
    present = np.asarray(present, dtype=bool)
    finite = np.isfinite(coords[:, 0, 0]) if coords.size else np.zeros_like(present, dtype=bool)
    return present & finite


def palm_scale(xy: np.ndarray) -> np.ndarray:
    wrist = xy[:, 0, :]
    palm = xy[:, PALM_LANDMARKS, :]
    dist = np.linalg.norm(palm - wrist[:, None, :], axis=2)
    scale = np.nanmean(dist, axis=1)
    return np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)


def add_edge_features(row: dict, prefix: str, xy: np.ndarray) -> None:
    if xy.size == 0:
        add_summary(row, f"{prefix}_edge_margin_all", np.array([]))
        add_summary(row, f"{prefix}_edge_margin_tip", np.array([]))
        row[f"{prefix}_oob_landmark_ratio"] = 0.0
        row[f"{prefix}_oob_frame_any_ratio"] = 0.0
        for threshold in EDGE_THRESHOLDS:
            t = threshold_name(threshold)
            for name in (
                "edge_landmark_ratio",
                "edge_tip_ratio",
                "edge_frame_any_ratio",
                "edge_bbox_touch_ratio",
                "edge_left_ratio",
                "edge_right_ratio",
                "edge_top_ratio",
                "edge_bottom_ratio",
            ):
                row[f"{prefix}_{name}_{t}"] = 0.0
        return

    x = xy[:, :, 0]
    y = xy[:, :, 1]
    margin = np.minimum.reduce([x, 1.0 - x, y, 1.0 - y])
    tip_margin = margin[:, TIP_LANDMARKS]
    frame_min_margin = np.nanmin(margin, axis=1)
    frame_tip_min_margin = np.nanmin(tip_margin, axis=1)
    out_of_bounds = (x < 0.0) | (x > 1.0) | (y < 0.0) | (y > 1.0)
    add_summary(row, f"{prefix}_edge_margin_all", margin.reshape(-1))
    add_summary(row, f"{prefix}_edge_margin_tip", tip_margin.reshape(-1))
    add_summary(row, f"{prefix}_edge_frame_min_margin", frame_min_margin)
    add_summary(row, f"{prefix}_edge_tip_frame_min_margin", frame_tip_min_margin)
    row[f"{prefix}_oob_landmark_ratio"] = safe_float(np.mean(out_of_bounds))
    row[f"{prefix}_oob_frame_any_ratio"] = safe_float(np.mean(np.any(out_of_bounds, axis=1)))

    x_min = np.nanmin(x, axis=1)
    x_max = np.nanmax(x, axis=1)
    y_min = np.nanmin(y, axis=1)
    y_max = np.nanmax(y, axis=1)
    for threshold in EDGE_THRESHOLDS:
        t = threshold_name(threshold)
        row[f"{prefix}_edge_landmark_ratio_{t}"] = safe_float(np.mean(margin <= threshold))
        row[f"{prefix}_edge_tip_ratio_{t}"] = safe_float(np.mean(tip_margin <= threshold))
        row[f"{prefix}_edge_frame_any_ratio_{t}"] = safe_float(np.mean(frame_min_margin <= threshold))
        row[f"{prefix}_edge_tip_frame_any_ratio_{t}"] = safe_float(np.mean(frame_tip_min_margin <= threshold))
        row[f"{prefix}_edge_bbox_touch_ratio_{t}"] = safe_float(
            np.mean((x_min <= threshold) | (x_max >= 1.0 - threshold) | (y_min <= threshold) | (y_max >= 1.0 - threshold))
        )
        row[f"{prefix}_edge_left_ratio_{t}"] = safe_float(np.mean(x <= threshold))
        row[f"{prefix}_edge_right_ratio_{t}"] = safe_float(np.mean(x >= 1.0 - threshold))
        row[f"{prefix}_edge_top_ratio_{t}"] = safe_float(np.mean(y <= threshold))
        row[f"{prefix}_edge_bottom_ratio_{t}"] = safe_float(np.mean(y >= 1.0 - threshold))


def add_bin_features(row: dict, prefix: str, coords: np.ndarray, present: np.ndarray, times: np.ndarray) -> None:
    total = len(present)
    edges = np.linspace(0, total, BIN_COUNT + 1, dtype=int)
    for bin_idx in range(BIN_COUNT):
        start, end = int(edges[bin_idx]), int(edges[bin_idx + 1])
        if end <= start:
            end = min(total, start + 1)
        bin_present = present[start:end]
        row[f"{prefix}_bin{bin_idx}_coverage"] = safe_float(np.mean(bin_present)) if len(bin_present) else 0.0
        if not np.any(bin_present):
            row[f"{prefix}_bin{bin_idx}_centroid_x"] = 0.0
            row[f"{prefix}_bin{bin_idx}_centroid_y"] = 0.0
            row[f"{prefix}_bin{bin_idx}_bbox_area"] = 0.0
            row[f"{prefix}_bin{bin_idx}_edge_any_05"] = 0.0
            row[f"{prefix}_bin{bin_idx}_motion_mean"] = 0.0
            continue
        xy = coords[start:end][bin_present, :, :2]
        centroid = np.nanmean(xy, axis=1)
        x_min = np.nanmin(xy[:, :, 0], axis=1)
        x_max = np.nanmax(xy[:, :, 0], axis=1)
        y_min = np.nanmin(xy[:, :, 1], axis=1)
        y_max = np.nanmax(xy[:, :, 1], axis=1)
        margin = np.minimum.reduce([xy[:, :, 0], 1.0 - xy[:, :, 0], xy[:, :, 1], 1.0 - xy[:, :, 1]])
        row[f"{prefix}_bin{bin_idx}_centroid_x"] = safe_float(np.mean(centroid[:, 0]))
        row[f"{prefix}_bin{bin_idx}_centroid_y"] = safe_float(np.mean(centroid[:, 1]))
        row[f"{prefix}_bin{bin_idx}_bbox_area"] = safe_float(np.mean((x_max - x_min) * (y_max - y_min)))
        row[f"{prefix}_bin{bin_idx}_edge_any_05"] = safe_float(np.mean(np.nanmin(margin, axis=1) <= 0.05))
        valid_times = times[start:end][bin_present] if len(times) == total else np.arange(len(xy), dtype=float)
        if len(xy) >= 2:
            dt = np.diff(valid_times)
            dt = np.where(dt > 1e-6, dt, 1.0)
            motion = np.linalg.norm(np.diff(centroid, axis=0), axis=1) / dt
            row[f"{prefix}_bin{bin_idx}_motion_mean"] = safe_float(np.mean(motion))
        else:
            row[f"{prefix}_bin{bin_idx}_motion_mean"] = 0.0


def add_hand_features(row: dict, prefix: str, coords: np.ndarray, present: np.ndarray, times: np.ndarray) -> None:
    present = valid_present(coords, present)
    total = int(len(present))
    count = int(np.sum(present))
    row[f"{prefix}_seq_total_frames"] = float(total)
    row[f"{prefix}_seq_present_frames"] = float(count)
    row[f"{prefix}_seq_coverage"] = safe_float(count / max(total, 1))
    row[f"{prefix}_seq_miss_pct"] = safe_float(100.0 * (1.0 - count / max(total, 1)))
    row[f"{prefix}_seq_longest_miss_run"] = float(longest_run(~present))
    row[f"{prefix}_seq_longest_present_run"] = float(longest_run(present))
    add_bin_features(row, prefix, coords, present, times)

    if count == 0:
        add_edge_features(row, prefix, np.empty((0, 21, 2), dtype=np.float32))
        return

    valid = coords[present]
    xy = valid[:, :, :2]
    z = valid[:, :, 2]
    valid_times = times[present] if len(times) == total else np.arange(count, dtype=float)
    centroids = np.nanmean(xy, axis=1)
    scale = palm_scale(xy)
    normalized_xy = (xy - centroids[:, None, :]) / scale[:, None, None]
    x_min = np.nanmin(xy[:, :, 0], axis=1)
    x_max = np.nanmax(xy[:, :, 0], axis=1)
    y_min = np.nanmin(xy[:, :, 1], axis=1)
    y_max = np.nanmax(xy[:, :, 1], axis=1)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_area = bbox_w * bbox_h
    spread = np.linalg.norm(xy - centroids[:, None, :], axis=2).mean(axis=1)

    add_summary(row, f"{prefix}_centroid_x", centroids[:, 0])
    add_summary(row, f"{prefix}_centroid_y", centroids[:, 1])
    add_summary(row, f"{prefix}_bbox_w", bbox_w)
    add_summary(row, f"{prefix}_bbox_h", bbox_h)
    add_summary(row, f"{prefix}_bbox_area", bbox_area)
    add_summary(row, f"{prefix}_spread", spread)
    add_summary(row, f"{prefix}_palm_scale", scale)
    add_edge_features(row, prefix, xy)

    for landmark in SELECTED_LANDMARKS:
        add_summary(row, f"{prefix}_lm{landmark}_x", xy[:, landmark, 0])
        add_summary(row, f"{prefix}_lm{landmark}_y", xy[:, landmark, 1])
        add_summary(row, f"{prefix}_lm{landmark}_z", z[:, landmark])
        add_summary(row, f"{prefix}_lm{landmark}_nx", normalized_xy[:, landmark, 0])
        add_summary(row, f"{prefix}_lm{landmark}_ny", normalized_xy[:, landmark, 1])

    if count >= 2:
        dt = np.diff(valid_times)
        dt = np.where(dt > 1e-6, dt, 1.0)
        centroid_motion = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        centroid_speed = centroid_motion / dt
        landmark_motion = np.linalg.norm(np.diff(xy[:, SELECTED_LANDMARKS, :], axis=0), axis=2).mean(axis=1)
        landmark_speed = landmark_motion / dt
        add_summary(row, f"{prefix}_motion_centroid", centroid_motion)
        add_summary(row, f"{prefix}_speed_centroid", centroid_speed)
        add_summary(row, f"{prefix}_motion_landmark", landmark_motion)
        add_summary(row, f"{prefix}_speed_landmark", landmark_speed)
    else:
        add_summary(row, f"{prefix}_motion_centroid", np.array([]))
        add_summary(row, f"{prefix}_speed_centroid", np.array([]))
        add_summary(row, f"{prefix}_motion_landmark", np.array([]))
        add_summary(row, f"{prefix}_speed_landmark", np.array([]))

    if count >= 3:
        accel = np.diff(np.diff(centroids, axis=0), axis=0)
        jitter = np.linalg.norm(accel, axis=1)
        add_summary(row, f"{prefix}_jitter_centroid", jitter)
    else:
        add_summary(row, f"{prefix}_jitter_centroid", np.array([]))


def add_cross_hand_features(row: dict, left: np.ndarray, right: np.ndarray, present_left: np.ndarray, present_right: np.ndarray) -> None:
    present_left = valid_present(left, present_left)
    present_right = valid_present(right, present_right)
    any_hand = present_left | present_right
    both_hand = present_left & present_right
    row["seq_any_hand_coverage"] = safe_float(np.mean(any_hand)) if len(any_hand) else 0.0
    row["seq_both_hands_coverage"] = safe_float(np.mean(both_hand)) if len(both_hand) else 0.0
    row["seq_hand_coverage_gap"] = safe_float(abs(np.mean(present_left) - np.mean(present_right))) if len(any_hand) else 0.0
    row["seq_longest_no_hand_run"] = float(longest_run(~any_hand))
    row["seq_longest_two_hand_run"] = float(longest_run(both_hand))
    if not np.any(both_hand):
        add_summary(row, "cross_hand_distance", np.array([]))
        add_summary(row, "cross_hand_x_gap", np.array([]))
        add_summary(row, "cross_hand_y_gap", np.array([]))
        return
    left_centroid = np.nanmean(left[both_hand, :, :2], axis=1)
    right_centroid = np.nanmean(right[both_hand, :, :2], axis=1)
    delta = left_centroid - right_centroid
    add_summary(row, "cross_hand_distance", np.linalg.norm(delta, axis=1))
    add_summary(row, "cross_hand_x_gap", delta[:, 0])
    add_summary(row, "cross_hand_y_gap", delta[:, 1])


def keypoint_features(row: pd.Series, base: Path) -> dict:
    result: dict = {}
    keypoints_path = str(row.get("keypoints_path", ""))
    if not keypoints_path:
        result["enrichment_status"] = "missing_keypoints_path"
        return result
    path = resolve_source_path(keypoints_path, base)
    if not path.exists():
        result["enrichment_status"] = "missing_keypoints_file"
        return result
    try:
        data = np.load(path)
        left = data["coords_left"].astype(np.float32)
        right = data["coords_right"].astype(np.float32)
        present_left = data["present_left"].astype(bool)
        present_right = data["present_right"].astype(bool)
        times = data["t_seconds"].astype(float) if "t_seconds" in data.files else np.arange(len(present_left), dtype=float)
    except Exception as exc:
        result["enrichment_status"] = "keypoints_error"
        result["enrichment_error"] = str(exc)
        return result

    add_hand_features(result, "left", left, present_left, times)
    add_hand_features(result, "right", right, present_right, times)
    add_cross_hand_features(result, left, right, present_left, present_right)
    result["enrichment_status"] = "ok"
    result["enrichment_error"] = ""
    return result


def file_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if not path.exists() or not path.is_file():
        return ""
    stat = path.stat()
    hasher = hashlib.md5()
    hasher.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as handle:
        first = handle.read(chunk_size)
        hasher.update(first)
        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            hasher.update(handle.read(chunk_size))
    return f"{stat.st_size}_{hasher.hexdigest()[:16]}"


def add_content_groups(df: pd.DataFrame, base: Path) -> pd.DataFrame:
    work = df.copy()
    if "content_fingerprint" not in work.columns:
        work["content_fingerprint"] = ""
    fingerprints = []
    for _, row in work.iterrows():
        source = str(row.get("source_path", ""))
        fingerprints.append(file_fingerprint(resolve_source_path(source, base)) if source else "")
    work["content_fingerprint"] = fingerprints
    work["content_group_id"] = work.get("group_id", work["sample_id"]).astype(str)
    work["content_duplicate_status"] = "not_checked"

    for fingerprint, idx in work.groupby("content_fingerprint").groups.items():
        if not fingerprint or len(idx) <= 1:
            continue
        idx_list = list(idx)
        work.loc[idx_list, "content_group_id"] = f"content_{fingerprint}"
        work.loc[idx_list, "content_duplicate_status"] = "content_duplicate"

    if "normalized_stem" in work.columns:
        for stem, idx in work.groupby(work["normalized_stem"].astype(str)).groups.items():
            idx_list = list(idx)
            if not stem or len(idx_list) <= 1:
                continue
            mask = work.index.isin(idx_list) & (work["content_duplicate_status"] == "not_checked")
            work.loc[mask, "content_group_id"] = f"name_{stem}"
            work.loc[mask, "content_duplicate_status"] = "name_only_duplicate"
    return work


def framing_level(score: float, edge_ratio: float, max_miss: float) -> str:
    if edge_ratio >= 0.40 or score >= 0.55 or max_miss >= 85.0:
        return "high_risk"
    if edge_ratio >= 0.20 or score >= 0.30 or max_miss >= 65.0:
        return "review"
    return "ok"


def build_framing_report(df: pd.DataFrame) -> pd.DataFrame:
    left_edge = pd.to_numeric(df.get("left_edge_frame_any_ratio_05", 0.0), errors="coerce").fillna(0.0)
    right_edge = pd.to_numeric(df.get("right_edge_frame_any_ratio_05", 0.0), errors="coerce").fillna(0.0)
    max_edge = np.maximum(left_edge, right_edge)
    any_cov = pd.to_numeric(df.get("seq_any_hand_coverage", df.get("any_hand_coverage", 0.0)), errors="coerce").fillna(0.0)
    both_cov = pd.to_numeric(df.get("seq_both_hands_coverage", df.get("both_hands_coverage", 0.0)), errors="coerce").fillna(0.0)
    max_miss = pd.to_numeric(df.get("max_miss", 0.0), errors="coerce").fillna(0.0)
    oob = np.maximum(
        pd.to_numeric(df.get("left_oob_frame_any_ratio", 0.0), errors="coerce").fillna(0.0),
        pd.to_numeric(df.get("right_oob_frame_any_ratio", 0.0), errors="coerce").fillna(0.0),
    )
    score = (0.45 * max_edge) + (0.20 * (1.0 - any_cov)) + (0.20 * (max_miss / 100.0)) + (0.10 * (1.0 - both_cov)) + (0.05 * oob)
    report_cols = [
        "sample_id",
        "label",
        "source_set",
        "original_name",
        "normalized_stem",
        "source_path",
        "miss_left_pct",
        "miss_right_pct",
        "max_miss",
        "any_hand_coverage",
        "both_hands_coverage",
    ]
    out = df[[col for col in report_cols if col in df.columns]].copy()
    out["left_edge_frame_any_ratio_05"] = left_edge
    out["right_edge_frame_any_ratio_05"] = right_edge
    out["max_edge_frame_any_ratio_05"] = max_edge
    out["seq_any_hand_coverage"] = any_cov
    out["seq_both_hands_coverage"] = both_cov
    out["oob_frame_any_ratio"] = oob
    out["framing_risk_score"] = score
    out["framing_level"] = [framing_level(safe_float(s), safe_float(e), safe_float(m)) for s, e, m in zip(score, max_edge, max_miss)]
    out["framing_note"] = np.where(
        out["framing_level"] == "high_risk",
        "hands often near/outside frame or detection is very poor",
        np.where(out["framing_level"] == "review", "review capture; hands may be close to frame edge", "capture looks usable"),
    )
    return out.sort_values(["framing_risk_score", "max_edge_frame_any_ratio_05"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add temporal, duplicate, and hand-framing features to v2 extraction output.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--framing-report", default=None)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v2_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "features.csv"
    out_csv = Path(args.out) if args.out else out_dir / "features_enriched.csv"
    report_csv = Path(args.framing_report) if args.framing_report else out_dir / "hand_framing_report.csv"

    df = pd.read_csv(features_csv)
    rows = []
    for _, row in df.iterrows():
        base_row = row.to_dict()
        if str(base_row.get("extraction_status", "")).lower() == "ok":
            base_row.update(keypoint_features(row, base))
        else:
            base_row["enrichment_status"] = "skipped_extraction_not_ok"
        rows.append(base_row)

    enriched = pd.DataFrame(rows)
    enriched = add_content_groups(enriched, base)
    numeric_cols = enriched.select_dtypes(include=[np.number, "bool"]).columns
    enriched[numeric_cols] = enriched[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ensure_dir(out_csv.parent)
    enriched.to_csv(out_csv, index=False)

    framing = build_framing_report(enriched)
    ensure_dir(report_csv.parent)
    framing.to_csv(report_csv, index=False)

    status_counts = enriched["enrichment_status"].value_counts(dropna=False).to_dict() if "enrichment_status" in enriched.columns else {}
    framing_counts = framing["framing_level"].value_counts(dropna=False).to_dict()
    metadata = {
        "generated_at": utc_now_iso(),
        "input": rel_to(features_csv, base),
        "output": rel_to(out_csv, base),
        "framing_report": rel_to(report_csv, base),
        "rows": int(len(enriched)),
        "columns": int(len(enriched.columns)),
        "enrichment_status": status_counts,
        "content_duplicate_status": enriched["content_duplicate_status"].value_counts(dropna=False).to_dict(),
        "framing_level": framing_counts,
    }
    write_json(out_dir / "features_enriched_metadata.json", metadata)
    print("=== V2 FEATURE ENRICHMENT ===", flush=True)
    print(f"input: {features_csv}", flush=True)
    print(f"output: {out_csv}", flush=True)
    print(f"rows: {len(enriched)} columns: {len(enriched.columns)}", flush=True)
    print(f"enrichment_status: {status_counts}", flush=True)
    print(f"framing_level: {framing_counts}", flush=True)
    print(f"framing_report: {report_csv}", flush=True)


if __name__ == "__main__":
    main()
