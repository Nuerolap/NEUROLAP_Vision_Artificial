from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ensure_dir, resolve_source_path, utc_now_iso, videos_base, write_json
from .edm_common_v3 import artifacts_v3_dir, load_v3_source_table


HAND_ORDER_COLUMNS = (
    "hand_order_status",
    "hand_order_confidence",
    "hand_order_first_active",
    "hand_order_mirror_suggested",
    "hand_order_reason",
    "early_left_activity",
    "early_right_activity",
    "late_left_activity",
    "late_right_activity",
    "early_left_coverage",
    "early_right_coverage",
    "late_left_coverage",
    "late_right_coverage",
    "hand_order_margin",
)


def safe_float(value: float, default: float = 0.0) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    if not np.isfinite(value):
        return default
    return value


def valid_present(coords: np.ndarray, present: np.ndarray) -> np.ndarray:
    present = np.asarray(present, dtype=bool)
    if coords.size == 0:
        return np.zeros_like(present, dtype=bool)
    return present & np.isfinite(coords[:, 0, 0])


def hand_motion(coords: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    present = valid_present(coords, present)
    motion = np.zeros(len(present), dtype=np.float32)
    idx = np.where(present)[0]
    if len(idx) >= 2:
        centroids = np.nanmean(coords[present, :, :2], axis=1)
        diffs = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        motion[idx[1:]] = diffs.astype(np.float32)
    return motion, present


def window_activity(motion: np.ndarray, present: np.ndarray, start: int, end: int) -> tuple[float, float, float]:
    start = max(0, int(start))
    end = min(len(present), int(end))
    length = max(end - start, 1)
    present_window = present[start:end]
    motion_window = motion[start:end]
    coverage = safe_float(np.mean(present_window)) if length else 0.0
    motion_rate = safe_float(np.nansum(motion_window) / length)
    # Coverage is a small tie-breaker; motion carries the actual order signal.
    activity = motion_rate + (0.03 * coverage)
    return safe_float(activity), safe_float(coverage), safe_float(motion_rate)


def audit_row(row: pd.Series, base: Path, *, margin_threshold: float, early_threshold: float) -> dict:
    out = {
        "sample_id": row["sample_id"],
        "label_actual": row.get("label_actual", row.get("label", "")),
        "source_set": row.get("source_set", ""),
        "source_path": row.get("source_path", ""),
        "original_name": row.get("original_name", ""),
        "calidad_video": row.get("framing_level", ""),
        "keypoints_path": row.get("keypoints_path", ""),
    }
    keypoints_value = str(row.get("keypoints_path", "")).strip()
    if not keypoints_value:
        out.update(error_result("missing_keypoints_path"))
        return out
    keypoints_path = resolve_source_path(keypoints_value, base)
    if not keypoints_path.exists():
        out.update(error_result(f"missing_keypoints_file:{keypoints_path}"))
        return out

    try:
        data = np.load(keypoints_path)
        left = data["coords_left"].astype(np.float32)
        right = data["coords_right"].astype(np.float32)
        present_left = data["present_left"].astype(bool)
        present_right = data["present_right"].astype(bool)
    except Exception as exc:
        out.update(error_result(f"keypoints_error:{exc}"))
        return out

    total = int(min(len(present_left), len(present_right), len(left), len(right)))
    if total < 24:
        out.update(error_result("too_few_keypoint_frames"))
        return out

    left_motion, left_present = hand_motion(left[:total], present_left[:total])
    right_motion, right_present = hand_motion(right[:total], present_right[:total])

    early_start = int(round(total * 0.08))
    early_end = int(round(total * 0.48))
    late_start = int(round(total * 0.52))
    late_end = int(round(total * 0.92))

    early_left, early_left_cov, early_left_motion = window_activity(left_motion, left_present, early_start, early_end)
    early_right, early_right_cov, early_right_motion = window_activity(right_motion, right_present, early_start, early_end)
    late_left, late_left_cov, late_left_motion = window_activity(left_motion, left_present, late_start, late_end)
    late_right, late_right_cov, late_right_motion = window_activity(right_motion, right_present, late_start, late_end)

    early_left_advantage = early_left - early_right
    late_right_advantage = late_right - late_left
    starts_left_score = early_left_advantage + late_right_advantage
    starts_right_score = -starts_left_score
    margin = abs(starts_left_score - starts_right_score)

    if starts_left_score > starts_right_score and margin >= margin_threshold and early_left_advantage >= early_threshold:
        status = "starts_left"
        first_active = "left"
        mirror = False
        reason = "early_left_and_late_right_activity"
    elif starts_right_score > starts_left_score and margin >= margin_threshold and -early_left_advantage >= early_threshold:
        status = "starts_right"
        first_active = "right"
        mirror = True
        reason = "early_right_and_late_left_activity"
    else:
        status = "uncertain"
        first_active = "right" if early_right > early_left else ("left" if early_left > early_right else "")
        mirror = False
        reason = "low_or_conflicting_hand_order_signal"

    confidence = min(0.99, max(0.0, margin / 0.18))
    out.update(
        {
            "hand_order_status": status,
            "hand_order_confidence": round(float(confidence), 4),
            "hand_order_first_active": first_active,
            "hand_order_mirror_suggested": "1" if mirror else "0",
            "hand_order_reason": reason,
            "early_left_activity": round(float(early_left), 6),
            "early_right_activity": round(float(early_right), 6),
            "late_left_activity": round(float(late_left), 6),
            "late_right_activity": round(float(late_right), 6),
            "early_left_coverage": round(float(early_left_cov), 6),
            "early_right_coverage": round(float(early_right_cov), 6),
            "late_left_coverage": round(float(late_left_cov), 6),
            "late_right_coverage": round(float(late_right_cov), 6),
            "early_left_motion_rate": round(float(early_left_motion), 6),
            "early_right_motion_rate": round(float(early_right_motion), 6),
            "late_left_motion_rate": round(float(late_left_motion), 6),
            "late_right_motion_rate": round(float(late_right_motion), 6),
            "hand_order_margin": round(float(margin), 6),
            "hand_order_status_detail": "ok",
        }
    )
    return out


def error_result(message: str) -> dict:
    return {
        "hand_order_status": "error",
        "hand_order_confidence": 0.0,
        "hand_order_first_active": "",
        "hand_order_mirror_suggested": "0",
        "hand_order_reason": message,
        "early_left_activity": 0.0,
        "early_right_activity": 0.0,
        "late_left_activity": 0.0,
        "late_right_activity": 0.0,
        "early_left_coverage": 0.0,
        "early_right_coverage": 0.0,
        "late_left_coverage": 0.0,
        "late_right_coverage": 0.0,
        "early_left_motion_rate": 0.0,
        "early_right_motion_rate": 0.0,
        "late_left_motion_rate": 0.0,
        "late_right_motion_rate": 0.0,
        "hand_order_margin": 0.0,
        "hand_order_status_detail": message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether EDM videos appear to start with left or right hand.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--index", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--margin-threshold", type=float, default=0.018)
    parser.add_argument("--early-threshold", type=float, default=0.003)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v3_dir(base))
    out_csv = Path(args.out).resolve() if args.out else out_dir / "edm_hand_order_audit.csv"
    source = load_v3_source_table(
        base,
        index_csv=Path(args.index) if args.index else None,
        features_csv=Path(args.features) if args.features else None,
    )
    if args.sample_ids:
        wanted = {str(value) for value in args.sample_ids}
        source = source[source["sample_id"].astype(str).isin(wanted)].copy()
    source = source.sort_values(["source_set", "label_actual", "normalized_stem", "sample_id"]).reset_index(drop=True)
    if args.limit:
        source = source.head(args.limit).copy()

    rows = [audit_row(row, base, margin_threshold=args.margin_threshold, early_threshold=args.early_threshold) for _, row in source.iterrows()]
    out = pd.DataFrame(rows)
    ensure_dir(out_csv.parent)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": utc_now_iso(),
        "out_csv": str(out_csv),
        "rows": int(len(out)),
        "status_counts": dict(Counter(out["hand_order_status"].astype(str))),
        "mirror_suggested": int(out["hand_order_mirror_suggested"].astype(str).eq("1").sum()),
        "by_label_status": {
            f"{label}|{status}": int(count)
            for (label, status), count in out.groupby(["label_actual", "hand_order_status"]).size().items()
        },
        "margin_threshold": float(args.margin_threshold),
        "early_threshold": float(args.early_threshold),
    }
    write_json(out_csv.with_suffix(".metadata.json"), summary)

    print("=== EDM V3 HAND ORDER AUDIT ===", flush=True)
    print(f"rows: {summary['rows']} status: {summary['status_counts']}", flush=True)
    print(f"mirror_suggested: {summary['mirror_suggested']}", flush=True)
    print(f"[OK] audit -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
