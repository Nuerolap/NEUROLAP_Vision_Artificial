from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ensure_dir, resolve_source_path, utc_now_iso, videos_base, write_json
from .edm_common_v3 import (
    EXERCISE_TIMING_COLUMNS,
    REP_COLUMNS,
    artifacts_v3_dir,
    load_v3_source_table,
    quality_from_row,
    score_from_color_proxy,
    truthy,
)
from .edm_hand_order_v3 import HAND_ORDER_COLUMNS


TIP_LANDMARKS = np.array([4, 8, 12, 16, 20], dtype=int)
MCP_LANDMARKS = np.array([2, 5, 9, 13, 17], dtype=int)
PALM_LANDMARKS = np.array([0, 5, 9, 13, 17], dtype=int)
SUMMARY_PERCENTILES = (10, 25, 50, 75, 90)


def safe_float(value: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    if not np.isfinite(value):
        return default
    return value


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
    if coords.size == 0:
        return np.zeros_like(present, dtype=bool)
    finite = np.isfinite(coords[:, 0, 0])
    return present & finite


def palm_scale(xy: np.ndarray) -> np.ndarray:
    wrist = xy[:, 0, :]
    palm = xy[:, PALM_LANDMARKS, :]
    dist = np.linalg.norm(palm - wrist[:, None, :], axis=2)
    scale = np.nanmean(dist, axis=1)
    return np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)


def hand_series(coords: np.ndarray, present: np.ndarray) -> dict[str, np.ndarray]:
    present = valid_present(coords, present)
    n = int(len(present))
    signals = {
        "present": present.astype(float),
        "centroid_x": np.zeros(n, dtype=np.float32),
        "centroid_y": np.zeros(n, dtype=np.float32),
        "motion": np.zeros(n, dtype=np.float32),
        "finger_extension": np.zeros(n, dtype=np.float32),
        "finger_spread": np.zeros(n, dtype=np.float32),
        "finger_curl": np.zeros(n, dtype=np.float32),
        "bbox_area": np.zeros(n, dtype=np.float32),
        "bbox_aspect": np.zeros(n, dtype=np.float32),
        "palm_proxy": np.zeros(n, dtype=np.float32),
        "fist_proxy": np.zeros(n, dtype=np.float32),
        "side_proxy": np.zeros(n, dtype=np.float32),
    }
    if not np.any(present):
        return signals

    valid_idx = np.where(present)[0]
    valid = coords[present]
    xy = valid[:, :, :2]
    wrist = xy[:, 0, :]
    centroids = np.nanmean(xy, axis=1)
    scale = palm_scale(xy)
    tips = xy[:, TIP_LANDMARKS, :]
    mcps = xy[:, MCP_LANDMARKS, :]
    tip_dist = np.linalg.norm(tips - wrist[:, None, :], axis=2)
    mcp_dist = np.linalg.norm(mcps - wrist[:, None, :], axis=2)
    extension = np.nanmean(tip_dist, axis=1) / scale
    mcp_extension = np.nanmean(mcp_dist, axis=1) / scale
    spread = np.linalg.norm(tips - np.nanmean(tips, axis=1, keepdims=True), axis=2).mean(axis=1) / scale
    curl = (mcp_extension - extension) / np.maximum(mcp_extension, 1e-6)
    x_min = np.nanmin(xy[:, :, 0], axis=1)
    x_max = np.nanmax(xy[:, :, 0], axis=1)
    y_min = np.nanmin(xy[:, :, 1], axis=1)
    y_max = np.nanmax(xy[:, :, 1], axis=1)
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    bbox_area = bbox_w * bbox_h
    bbox_aspect = bbox_w / np.maximum(bbox_h, 1e-6)

    signals["centroid_x"][valid_idx] = centroids[:, 0]
    signals["centroid_y"][valid_idx] = centroids[:, 1]
    signals["finger_extension"][valid_idx] = extension
    signals["finger_spread"][valid_idx] = spread
    signals["finger_curl"][valid_idx] = curl
    signals["bbox_area"][valid_idx] = bbox_area
    signals["bbox_aspect"][valid_idx] = bbox_aspect
    signals["palm_proxy"][valid_idx] = extension + spread
    signals["fist_proxy"][valid_idx] = curl - spread
    signals["side_proxy"][valid_idx] = -np.abs(bbox_aspect - 0.55) + (0.15 * extension)

    if len(valid_idx) >= 2:
        diffs = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        signals["motion"][valid_idx[1:]] = diffs
    return signals


def transition_count(values: np.ndarray, present: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    present = np.asarray(present, dtype=bool)
    valid = values[present & np.isfinite(values)]
    if valid.size < 4:
        return 0.0
    centered = valid - np.nanmedian(valid)
    sign = np.sign(centered)
    sign[np.abs(centered) < max(np.nanstd(valid) * 0.15, 1e-6)] = 0
    sign = sign[sign != 0]
    if sign.size < 2:
        return 0.0
    return float(np.sum(sign[1:] != sign[:-1]))


def add_hand_block(row: dict, prefix: str, signals: dict[str, np.ndarray], start: int, end: int) -> None:
    present = signals["present"][start:end] > 0.5
    length = max(int(end - start), 1)
    row[f"{prefix}_frames"] = float(length)
    row[f"{prefix}_coverage"] = safe_float(np.mean(present)) if length else 0.0
    for name in (
        "motion",
        "finger_extension",
        "finger_spread",
        "finger_curl",
        "bbox_area",
        "bbox_aspect",
        "palm_proxy",
        "fist_proxy",
        "side_proxy",
        "centroid_x",
        "centroid_y",
    ):
        values = signals[name][start:end]
        add_summary(row, f"{prefix}_{name}", values[present])
    row[f"{prefix}_motion_peaks"] = transition_count(signals["motion"][start:end], present)
    row[f"{prefix}_posture_transitions"] = transition_count(signals["palm_proxy"][start:end], present) + transition_count(signals["fist_proxy"][start:end], present)


def segment_bounds(total: int, segments: int = 6) -> list[tuple[int, int]]:
    edges = np.linspace(0, max(total, 1), segments + 1, dtype=int)
    bounds = []
    for idx in range(segments):
        start, end = int(edges[idx]), int(edges[idx + 1])
        if end <= start:
            end = min(max(total, 1), start + 1)
        bounds.append((start, end))
    return bounds


def load_keypoints(row: pd.Series, base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keypoints_value = str(row.get("keypoints_path", "")).strip()
    if not keypoints_value:
        raise FileNotFoundError("missing keypoints_path")
    path = resolve_source_path(keypoints_value, base)
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = np.load(path)
    return (
        data["coords_left"].astype(np.float32),
        data["coords_right"].astype(np.float32),
        data["present_left"].astype(bool),
        data["present_right"].astype(bool),
        data["t_seconds"].astype(np.float32) if "t_seconds" in data.files else np.arange(len(data["present_left"]), dtype=np.float32),
    )


def mirror_keypoints(
    left: np.ndarray,
    right: np.ndarray,
    present_left: np.ndarray,
    present_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mirrored_left = right.copy()
    mirrored_right = left.copy()
    if mirrored_left.size:
        mirrored_left[:, :, 0] = 1.0 - mirrored_left[:, :, 0]
    if mirrored_right.size:
        mirrored_right[:, :, 0] = 1.0 - mirrored_right[:, :, 0]
    return mirrored_left, mirrored_right, present_right.copy(), present_left.copy()


def mirror_decision(row: pd.Series) -> tuple[bool, str]:
    override = str(row.get("hand_order_override", "")).strip().casefold()
    if override == "mirror_on":
        return True, "human_override"
    if override == "mirror_off":
        return False, "human_override"
    return truthy(row.get("hand_order_mirror_suggested", "")), "auto_audit"


def parse_optional_time(value: object) -> float | None:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if not np.isfinite(out) or out < 0:
        return None
    return out


def apply_exercise_timing(
    left: np.ndarray,
    right: np.ndarray,
    present_left: np.ndarray,
    present_right: np.ndarray,
    times: np.ndarray,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    start_s = parse_optional_time(row.get("exercise_start_s", ""))
    end_s = parse_optional_time(row.get("exercise_end_s", ""))
    total = int(min(len(left), len(right), len(present_left), len(present_right), len(times)))
    if total <= 0:
        return left, right, present_left, present_right, times, {"exercise_timing_applied": False, "exercise_timing_source": "empty"}

    left = left[:total]
    right = right[:total]
    present_left = present_left[:total]
    present_right = present_right[:total]
    times = times[:total]
    if start_s is None and end_s is None:
        return left, right, present_left, present_right, times, {
            "exercise_timing_applied": False,
            "exercise_timing_source": "full_video",
            "exercise_start_s_used": "",
            "exercise_end_s_used": "",
        }

    mask = np.ones(total, dtype=bool)
    if start_s is not None:
        mask &= times >= start_s
    if end_s is not None and (start_s is None or end_s > start_s):
        mask &= times <= end_s
    idx = np.where(mask)[0]
    if len(idx) < 24:
        return left, right, present_left, present_right, times, {
            "exercise_timing_applied": False,
            "exercise_timing_source": "invalid_or_too_short",
            "exercise_start_s_used": "" if start_s is None else start_s,
            "exercise_end_s_used": "" if end_s is None else end_s,
        }
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    return left[lo:hi], right[lo:hi], present_left[lo:hi], present_right[lo:hi], times[lo:hi], {
        "exercise_timing_applied": True,
        "exercise_timing_source": "manual_annotation",
        "exercise_start_s_used": "" if start_s is None else round(float(start_s), 3),
        "exercise_end_s_used": "" if end_s is None else round(float(end_s), 3),
        "exercise_timing_frames": int(hi - lo),
    }


def feature_row(row: pd.Series, base: Path) -> dict:
    out = {
        "sample_id": row["sample_id"],
        "source_path": row["source_path"],
        "label_actual": row.get("label_actual", row.get("label", "")),
        "weak_score_from_folder": score_from_color_proxy(str(row.get("label_actual", row.get("label", "")))),
        "weak_color_from_folder": row.get("label_actual", row.get("label", "")),
        "source_set": row.get("source_set", ""),
        "original_name": row.get("original_name", ""),
        "normalized_stem": row.get("normalized_stem", ""),
        "group_id": row.get("group_id", ""),
        "content_group_id": row.get("content_group_id", row.get("group_id", "")),
        "duplicate_status": row.get("duplicate_status", ""),
        "content_duplicate_status": row.get("content_duplicate_status", ""),
        "calidad_video": quality_from_row(row),
        "framing_level": row.get("framing_level", ""),
        "keypoints_path": row.get("keypoints_path", ""),
        "source_duration_s": row.get("source_duration_s", ""),
        "source_frames": row.get("source_frames", ""),
        "source_fps": row.get("source_fps", ""),
        "any_hand_coverage": row.get("any_hand_coverage", ""),
        "both_hands_coverage": row.get("both_hands_coverage", ""),
        "left_coverage": row.get("left_coverage", ""),
        "right_coverage": row.get("right_coverage", ""),
        "max_miss": row.get("max_miss", ""),
        "hand_order_status": row.get("hand_order_status", ""),
        "hand_order_confidence": row.get("hand_order_confidence", ""),
        "hand_order_mirror_suggested": row.get("hand_order_mirror_suggested", ""),
        "hand_order_override": row.get("hand_order_override", ""),
        "hand_order_reason": row.get("hand_order_reason", ""),
        "exercise_start_s": row.get("exercise_start_s", ""),
        "exercise_end_s": row.get("exercise_end_s", ""),
        "exercise_timing_note": row.get("exercise_timing_note", ""),
    }
    try:
        left, right, present_left, present_right, times = load_keypoints(row, base)
    except Exception as exc:
        out["edm_feature_status"] = "error"
        out["edm_feature_error"] = str(exc)
        return out

    left, right, present_left, present_right, times, timing_meta = apply_exercise_timing(left, right, present_left, present_right, times, row)
    out.update(timing_meta)

    mirror_applied, mirror_source = mirror_decision(row)
    if mirror_applied:
        left, right, present_left, present_right = mirror_keypoints(left, right, present_left, present_right)
    out["hand_order_mirror_applied"] = bool(mirror_applied)
    out["hand_order_standardization_source"] = mirror_source

    left_s = hand_series(left, present_left)
    right_s = hand_series(right, present_right)
    total = int(min(len(left_s["present"]), len(right_s["present"])))
    if total <= 0:
        out["edm_feature_status"] = "error"
        out["edm_feature_error"] = "empty keypoints"
        return out

    times = times[:total] if len(times) >= total else np.arange(total, dtype=np.float32)
    out["edm_total_frames"] = float(total)
    out["edm_duration_s"] = safe_float(times[-1] - times[0]) if total >= 2 else safe_float(row.get("source_duration_s", 0.0))
    out["edm_left_coverage"] = safe_float(np.mean(left_s["present"][:total] > 0.5))
    out["edm_right_coverage"] = safe_float(np.mean(right_s["present"][:total] > 0.5))
    out["edm_both_coverage"] = safe_float(np.mean((left_s["present"][:total] > 0.5) & (right_s["present"][:total] > 0.5)))
    out["edm_any_coverage"] = safe_float(np.mean((left_s["present"][:total] > 0.5) | (right_s["present"][:total] > 0.5)))

    for name, signals in (("left", left_s), ("right", right_s)):
        add_hand_block(out, f"{name}_global", signals, 0, total)

    for rep_name, (start, end) in zip(REP_COLUMNS, segment_bounds(total, 6)):
        hand = "left" if rep_name.startswith("left") else "right"
        signals = left_s if hand == "left" else right_s
        other = right_s if hand == "left" else left_s
        add_hand_block(out, rep_name, signals, start, end)
        out[f"{rep_name}_other_hand_coverage"] = safe_float(np.mean(other["present"][start:end] > 0.5))
        out[f"{rep_name}_relative_start"] = safe_float(start / max(total, 1))
        out[f"{rep_name}_relative_end"] = safe_float(end / max(total, 1))

    out["left_right_motion_balance"] = safe_float(out.get("left_global_motion_mean", 0.0) - out.get("right_global_motion_mean", 0.0))
    out["left_right_coverage_balance"] = safe_float(out["edm_left_coverage"] - out["edm_right_coverage"])
    out["edm_feature_status"] = "ok"
    out["edm_feature_error"] = ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EDM-aware temporal features from v2 keypoints.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--index", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--hand-order", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--no-hand-order-standardization", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v3_dir(base))
    out_csv = Path(args.out) if args.out else out_dir / "edm_features.csv"
    source = load_v3_source_table(
        base,
        index_csv=Path(args.index) if args.index else None,
        features_csv=Path(args.features) if args.features else None,
    )
    hand_order_csv = Path(args.hand_order).resolve() if args.hand_order else out_dir / "edm_hand_order_audit.csv"
    if not args.no_hand_order_standardization and hand_order_csv.exists():
        audit = pd.read_csv(hand_order_csv, keep_default_na=False)
        keep = [col for col in ("sample_id", *HAND_ORDER_COLUMNS, "early_left_motion_rate", "early_right_motion_rate", "late_left_motion_rate", "late_right_motion_rate") if col in audit.columns]
        if "sample_id" in keep:
            source = source.merge(audit[keep].drop_duplicates(subset=["sample_id"]), on="sample_id", how="left")
        for col in HAND_ORDER_COLUMNS:
            if col not in source.columns:
                source[col] = ""
            source[col] = source[col].fillna("")
    else:
        for col in HAND_ORDER_COLUMNS:
            if col not in source.columns:
                source[col] = ""
    annotations_csv = Path(args.annotations).resolve() if args.annotations else out_dir / "edm_annotations_working.csv"
    if not args.no_hand_order_standardization and annotations_csv.exists():
        annotations = pd.read_csv(annotations_csv, keep_default_na=False)
        merge_cols = [col for col in ("sample_id", "hand_order_override", *EXERCISE_TIMING_COLUMNS) if col in annotations.columns]
        if "sample_id" in merge_cols:
            annotation_meta = annotations[merge_cols].drop_duplicates(subset=["sample_id"])
            source = source.drop(columns=[col for col in merge_cols if col != "sample_id"], errors="ignore").merge(annotation_meta, on="sample_id", how="left")
            if "hand_order_override" not in source.columns:
                source["hand_order_override"] = ""
            source["hand_order_override"] = source["hand_order_override"].fillna("")
        elif "hand_order_override" not in source.columns:
            source["hand_order_override"] = ""
    elif "hand_order_override" not in source.columns:
        source["hand_order_override"] = ""
    for col in EXERCISE_TIMING_COLUMNS:
        if col not in source.columns:
            source[col] = ""
        source[col] = source[col].fillna("")
    if args.sample_ids:
        wanted = {str(value) for value in args.sample_ids}
        source = source[source["sample_id"].astype(str).isin(wanted)].copy()
    source = source.sort_values(["source_set", "label_actual", "normalized_stem", "sample_id"]).reset_index(drop=True)
    if args.limit:
        source = source.head(args.limit).copy()

    rows = [feature_row(row, base) for _, row in source.iterrows()]
    out = pd.DataFrame(rows)
    ensure_dir(out_csv.parent)
    out.to_csv(out_csv, index=False)
    summary = {
        "generated_at": utc_now_iso(),
        "rows": int(len(out)),
        "status_counts": dict(Counter(out["edm_feature_status"].astype(str))) if "edm_feature_status" in out.columns else {},
        "labels": dict(Counter(out["label_actual"].astype(str))) if "label_actual" in out.columns else {},
        "quality": dict(Counter(out["calidad_video"].astype(str))) if "calidad_video" in out.columns else {},
        "out_csv": str(out_csv),
        "feature_columns": int(len([col for col in out.columns if pd.api.types.is_numeric_dtype(out[col])])),
        "hand_order_csv": str(hand_order_csv) if hand_order_csv.exists() else "",
        "hand_order_annotations_csv": str(annotations_csv) if annotations_csv.exists() else "",
        "hand_order_standardization": not args.no_hand_order_standardization and hand_order_csv.exists(),
        "mirror_applied": int(out["hand_order_mirror_applied"].astype(bool).sum()) if "hand_order_mirror_applied" in out.columns else 0,
        "mirror_source_counts": dict(Counter(out["hand_order_standardization_source"].astype(str))) if "hand_order_standardization_source" in out.columns else {},
        "exercise_timing_applied": int(out["exercise_timing_applied"].astype(bool).sum()) if "exercise_timing_applied" in out.columns else 0,
        "exercise_timing_source_counts": dict(Counter(out["exercise_timing_source"].astype(str))) if "exercise_timing_source" in out.columns else {},
    }
    write_json(out_csv.with_suffix(".metadata.json"), summary)

    print("=== EDM V3 FEATURES ===", flush=True)
    print(f"rows: {summary['rows']} status: {summary['status_counts']}", flush=True)
    print(f"labels: {summary['labels']}", flush=True)
    print(f"quality: {summary['quality']}", flush=True)
    print(f"[OK] features -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
