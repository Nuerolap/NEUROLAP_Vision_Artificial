from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import ensure_dir, rel_to, resolve_source_path, safe_slug, utc_now_iso, videos_base, write_json
from .edm_common_v3 import (
    ERROR_COLUMNS,
    EXERCISE_TIMING_COLUMNS,
    EXTRA_REP_COLUMNS,
    REP_COLUMNS,
    artifacts_v3_dir,
    load_v3_source_table,
    quality_from_row,
    score_from_color_proxy,
)


FRAME_LABELS = ("L1", "L2", "L3", "R1", "R2", "R3")
FRAME_FRACTIONS = (0.10, 0.24, 0.38, 0.55, 0.70, 0.85)
QUALITY_COLORS = {
    "ok": (80, 190, 100),
    "review": (0, 190, 230),
    "high_risk": (60, 70, 230),
}
LABEL_COLORS = {
    "amarillo": (0, 210, 255),
    "rojo": (40, 40, 230),
    "verde": (60, 170, 60),
}


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = frame.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 18, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def read_frame(cap: cv2.VideoCapture, fraction: float) -> np.ndarray | None:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        return None
    frame_idx = max(0, min(frame_count - 1, int(round(frame_count * fraction))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def load_presence(keypoints_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not keypoints_path.exists():
        return np.array([], dtype=bool), np.array([], dtype=bool)
    try:
        data = np.load(keypoints_path)
        return data["present_left"].astype(bool), data["present_right"].astype(bool)
    except Exception:
        return np.array([], dtype=bool), np.array([], dtype=bool)


def draw_timeline(canvas: np.ndarray, x: int, y: int, w: int, h: int, left: np.ndarray, right: np.ndarray) -> None:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 70, 70), 1)
    if len(left) == 0 or len(right) == 0:
        cv2.putText(canvas, "Sin keypoints", (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        return
    n = min(len(left), len(right))
    left = left[:n]
    right = right[:n]
    top_h = h // 2 - 2
    bot_y = y + h // 2 + 2
    for i in range(w):
        start = int(i * n / w)
        end = max(start + 1, int((i + 1) * n / w))
        l_cov = float(np.mean(left[start:end]))
        r_cov = float(np.mean(right[start:end]))
        l_col = (35, int(70 + 160 * l_cov), 60)
        r_col = (60, int(80 + 150 * r_cov), 220)
        cv2.line(canvas, (x + i, y + 14), (x + i, y + 14 + top_h), l_col, 1)
        cv2.line(canvas, (x + i, bot_y + 14), (x + i, bot_y + 14 + top_h), r_col, 1)
    for rep_idx in range(1, 6):
        sx = x + int(rep_idx * w / 6)
        cv2.line(canvas, (sx, y), (sx, y + h), (170, 170, 170), 1)
    cv2.putText(canvas, "Left", (x + 8, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 240, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Right", (x + 8, bot_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 220, 255), 1, cv2.LINE_AA)


def sheet_path_for(row: pd.Series, out_dir: Path) -> Path:
    label = safe_slug(str(row.get("label_actual", row.get("label", "unknown"))))
    return out_dir / "review_sheets" / label / f"{safe_slug(str(row['sample_id']), max_len=140)}.jpg"


def build_review_sheet(row: pd.Series, base: Path, out_path: Path) -> bool:
    source_path = resolve_source_path(str(row["source_path"]), base)
    keypoints_path = resolve_source_path(str(row.get("keypoints_path", "")), base) if str(row.get("keypoints_path", "")).strip() else Path("")
    label = str(row.get("label_actual", row.get("label", ""))).strip().casefold()
    quality = quality_from_row(row)
    label_color = LABEL_COLORS.get(label, (210, 210, 210))
    quality_color = QUALITY_COLORS.get(quality, (210, 210, 210))

    frame_w, frame_h = 260, 150
    cols = 3
    header_h = 116
    gap = 8
    grid_h = frame_h * 2 + gap
    timeline_h = 92
    footer_h = 48
    sheet_w = frame_w * cols
    sheet_h = header_h + grid_h + timeline_h + footer_h
    sheet = np.full((sheet_h, sheet_w, 3), 24, dtype=np.uint8)

    cv2.rectangle(sheet, (0, 0), (sheet_w - 1, sheet_h - 1), label_color, 2)
    cv2.putText(sheet, f"{label.upper()} | {quality.upper()} | {row['sample_id']}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_color, 1, cv2.LINE_AA)
    cv2.putText(sheet, f"{row.get('source_set', '')} | {row.get('original_name', '')}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    metrics = (
        f"dur={float(row.get('source_duration_s', 0) or 0):.1f}s "
        f"any={float(row.get('any_hand_coverage', 0) or 0):.2f} "
        f"both={float(row.get('both_hands_coverage', 0) or 0):.2f} "
        f"miss={float(row.get('max_miss', 0) or 0):.1f}"
    )
    cv2.putText(sheet, metrics, (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, quality_color, 1, cv2.LINE_AA)
    cv2.putText(sheet, rel_to(source_path, base)[-115:], (10, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

    cap = cv2.VideoCapture(str(source_path))
    for idx, (fraction, frame_label) in enumerate(zip(FRAME_FRACTIONS, FRAME_LABELS)):
        frame = read_frame(cap, fraction)
        x = (idx % cols) * frame_w
        y = header_h + (idx // cols) * (frame_h + gap)
        if frame is None:
            fitted = np.full((frame_h, frame_w, 3), 18, dtype=np.uint8)
            cv2.putText(fitted, "SIN FRAME", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        else:
            fitted = fit_frame(frame, (frame_w, frame_h))
        sheet[y : y + frame_h, x : x + frame_w] = fitted
        cv2.rectangle(sheet, (x, y), (x + frame_w - 1, y + frame_h - 1), label_color, 1)
        cv2.putText(sheet, f"{frame_label} {int(round(fraction * 100))}%", (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cap.release()

    left, right = load_presence(keypoints_path)
    timeline_y = header_h + grid_h + 8
    draw_timeline(sheet, 10, timeline_y, sheet_w - 20, timeline_h - 18, left, right)
    cv2.putText(
        sheet,
        "Anotar L1-L3 y R1-R3: correcta / parcial / incorrecta / no_visible",
        (10, sheet_h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    ensure_dir(out_path.parent)
    return bool(cv2.imwrite(str(out_path), sheet))


def build_annotation_template(df: pd.DataFrame, base: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        quality = quality_from_row(row)
        review_sheet = rel_to(sheet_path_for(row, out_dir), base)
        label = str(row.get("label_actual", row.get("label", ""))).strip().casefold()
        record = {
            "sample_id": row["sample_id"],
            "source_path": row["source_path"],
            "source_exists": bool(row.get("source_exists", False)),
            "label_actual": label,
            "suggested_score_from_folder": score_from_color_proxy(label),
            "suggested_color_from_folder": label,
            "score_0_6": "",
            "color_final": "",
            "confidence_human": "",
            "hand_order_override": "",
            "annotation_status": "",
            "reviewer": "",
            "review_sheet": review_sheet,
            "calidad_video": quality,
            "source_set": row.get("source_set", ""),
            "original_name": row.get("original_name", ""),
            "normalized_stem": row.get("normalized_stem", ""),
            "content_group_id": row.get("content_group_id", row.get("group_id", "")),
            "duplicate_status": row.get("duplicate_status", ""),
            "content_duplicate_status": row.get("content_duplicate_status", ""),
            "duration_s": row.get("source_duration_s", ""),
            "source_frames": row.get("source_frames", ""),
            "source_fps": row.get("source_fps", ""),
            "any_hand_coverage": row.get("any_hand_coverage", ""),
            "both_hands_coverage": row.get("both_hands_coverage", ""),
            "left_coverage": row.get("left_coverage", ""),
            "right_coverage": row.get("right_coverage", ""),
            "max_miss": row.get("max_miss", ""),
            "framing_risk_score": row.get("framing_risk_score", ""),
            "reviewer_notes": "",
        }
        for col in REP_COLUMNS:
            record[col] = ""
        for col in ERROR_COLUMNS:
            record[col] = ""
        for col in EXTRA_REP_COLUMNS:
            record[col] = ""
        for col in EXERCISE_TIMING_COLUMNS:
            record[col] = ""
        rows.append(record)
    ordered = [
        "sample_id",
        "source_path",
        "label_actual",
        "suggested_score_from_folder",
        "suggested_color_from_folder",
        "score_0_6",
        "color_final",
        "confidence_human",
        "hand_order_override",
        *REP_COLUMNS,
        *ERROR_COLUMNS,
        *EXTRA_REP_COLUMNS,
        *EXERCISE_TIMING_COLUMNS,
        "annotation_status",
        "reviewer",
        "reviewer_notes",
        "review_sheet",
        "calidad_video",
        "source_exists",
        "source_set",
        "original_name",
        "normalized_stem",
        "content_group_id",
        "duplicate_status",
        "content_duplicate_status",
        "duration_s",
        "source_frames",
        "source_fps",
        "any_hand_coverage",
        "both_hands_coverage",
        "left_coverage",
        "right_coverage",
        "max_miss",
        "framing_risk_score",
    ]
    return pd.DataFrame(rows).reindex(columns=ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create EDM-aware v3 annotation template and visual review sheets.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--index", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else artifacts_v3_dir(base))
    df = load_v3_source_table(
        base,
        index_csv=Path(args.index) if args.index else None,
        features_csv=Path(args.features) if args.features else None,
    )
    if args.sample_ids:
        wanted = {str(value) for value in args.sample_ids}
        df = df[df["sample_id"].astype(str).isin(wanted)].copy()
    df = df.sort_values(["source_set", "label_actual", "normalized_stem", "sample_id"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).copy()

    annotation = build_annotation_template(df, base, out_dir)
    annotation_csv = out_dir / "edm_annotation_template.csv"
    annotation.to_csv(annotation_csv, index=False)

    image_ok = 0
    image_skipped = 0
    image_failed = 0
    if not args.no_images:
        for _, row in df.iterrows():
            path = sheet_path_for(row, out_dir)
            if path.exists() and not args.overwrite_images:
                image_skipped += 1
                continue
            if build_review_sheet(row, base, path):
                image_ok += 1
            else:
                image_failed += 1

    summary = {
        "generated_at": utc_now_iso(),
        "out_dir": str(out_dir),
        "annotation_csv": str(annotation_csv),
        "rows": int(len(annotation)),
        "counts_by_label_actual": dict(Counter(annotation["label_actual"].astype(str))),
        "counts_by_calidad_video": dict(Counter(annotation["calidad_video"].astype(str))),
        "missing_source_paths": int((~annotation["source_exists"].astype(bool)).sum()),
        "images_written": int(image_ok),
        "images_skipped": int(image_skipped),
        "images_failed": int(image_failed),
    }
    write_json(out_dir / "edm_review_manifest.json", summary)

    print("=== EDM V3 REVIEW PACKAGE ===", flush=True)
    print(f"rows: {summary['rows']} labels: {summary['counts_by_label_actual']}", flush=True)
    print(f"quality: {summary['counts_by_calidad_video']}", flush=True)
    print(f"images written/skipped/failed: {image_ok}/{image_skipped}/{image_failed}", flush=True)
    print(f"[OK] annotation -> {annotation_csv}", flush=True)
    print(f"[OK] manifest -> {out_dir / 'edm_review_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
