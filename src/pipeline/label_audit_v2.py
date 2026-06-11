import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import artifacts_v2_dir, ensure_dir, rel_to, resolve_source_path, safe_slug, utc_now_iso, videos_base, write_json


LABEL_COLORS = {
    "amarillo": (0, 210, 255),
    "rojo": (40, 40, 230),
    "verde": (60, 170, 60),
}
TEXT_COLOR = (245, 245, 245)
BAR_COLOR = (28, 28, 28)


def truncate_text(value: str, max_chars: int) -> str:
    value = str(value)
    return value if len(value) <= max_chars else value[: max_chars - 3] + "..."


def read_frame_at(cap: cv2.VideoCapture, ratio: float) -> np.ndarray | None:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        target = int(max(0, min(frame_count - 1, round((frame_count - 1) * ratio))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if ok:
        return frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    return frame if ok else None


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    if frame is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def draw_text(img: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.52, color: tuple[int, int, int] = TEXT_COLOR) -> None:
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def video_thumbnails(video_path: Path, ratios: list[float], thumb_size: tuple[int, int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    thumbs = []
    if not cap.isOpened():
        return [letterbox(None, thumb_size) for _ in ratios]
    try:
        for ratio in ratios:
            frame = read_frame_at(cap, ratio)
            thumbs.append(letterbox(frame, thumb_size))
    finally:
        cap.release()
    return thumbs


def make_contact_sheet(group: pd.DataFrame, base: Path, output_path: Path, ratios: list[float], thumb_size: tuple[int, int]) -> bool:
    row_h = thumb_size[1] + 72
    sheet_w = thumb_size[0] * len(ratios)
    sheet_h = row_h * len(group)
    sheet = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)

    for row_idx, (_, row) in enumerate(group.reset_index(drop=True).iterrows()):
        y0 = row_idx * row_h
        label = str(row.get("label", ""))
        color = LABEL_COLORS.get(label, (90, 90, 90))
        cv2.rectangle(sheet, (0, y0), (sheet_w, y0 + 72), BAR_COLOR, -1)
        cv2.rectangle(sheet, (0, y0), (12, y0 + row_h), color, -1)
        title = f"{label.upper()} | {row.get('sample_id', '')}"
        subtitle = f"{row.get('source_set', '')} | {row.get('original_name', '')}"
        draw_text(sheet, truncate_text(title, 92), (20, y0 + 25), 0.58, color)
        draw_text(sheet, truncate_text(subtitle, 110), (20, y0 + 51), 0.48, TEXT_COLOR)

        source_path = resolve_source_path(str(row.get("source_path", "")), base)
        thumbs = video_thumbnails(source_path, ratios, thumb_size)
        for thumb_idx, thumb in enumerate(thumbs):
            x0 = thumb_idx * thumb_size[0]
            y_thumb = y0 + 72
            sheet[y_thumb : y_thumb + thumb_size[1], x0 : x0 + thumb_size[0]] = thumb
            cv2.rectangle(sheet, (x0, y_thumb), (x0 + thumb_size[0] - 1, y_thumb + thumb_size[1] - 1), color, 2)
            draw_text(sheet, f"{int(ratios[thumb_idx] * 100)}%", (x0 + 8, y_thumb + 22), 0.52, TEXT_COLOR)

    ensure_dir(output_path.parent)
    return bool(cv2.imwrite(str(output_path), sheet))


def conflict_type(group_id: str) -> str:
    if str(group_id).startswith("content_"):
        return "exact_content_cross_label"
    if str(group_id).startswith("name_"):
        return "name_only_cross_label"
    return "cross_label"


def find_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    conflict_ids = []
    for group_id, group in df.groupby("content_group_id", dropna=False):
        labels = sorted(group["label"].astype(str).unique().tolist())
        if len(labels) > 1:
            conflict_ids.append(str(group_id))
    conflicts = df[df["content_group_id"].astype(str).isin(conflict_ids)].copy()
    conflicts["conflict_type"] = conflicts["content_group_id"].astype(str).map(conflict_type)
    conflicts["labels_in_group"] = conflicts.groupby("content_group_id")["label"].transform(lambda s: ",".join(sorted(s.astype(str).unique())))
    conflicts["rows_in_group"] = conflicts.groupby("content_group_id")["sample_id"].transform("count")
    return conflicts.sort_values(["conflict_type", "content_group_id", "label", "sample_id"]).reset_index(drop=True)


def merge_framing(conflicts: pd.DataFrame, framing_csv: Path) -> pd.DataFrame:
    if not framing_csv.exists():
        return conflicts
    framing = pd.read_csv(framing_csv)
    keep = [
        "sample_id",
        "framing_level",
        "framing_risk_score",
        "max_edge_frame_any_ratio_05",
        "seq_any_hand_coverage",
        "seq_both_hands_coverage",
        "oob_frame_any_ratio",
    ]
    keep = [col for col in keep if col in framing.columns]
    return conflicts.merge(framing[keep], on="sample_id", how="left")


def build_review_template(audit: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "content_group_id",
        "conflict_type",
        "labels_in_group",
        "sample_id",
        "label",
        "source_set",
        "original_name",
        "normalized_stem",
        "source_path",
        "contact_sheet",
    ]
    out = audit[[col for col in cols if col in audit.columns]].copy()
    out.insert(0, "review_status", "")
    out.insert(1, "reviewed_label", "")
    out.insert(2, "review_action", "")
    out.insert(3, "sample_weight", "")
    out.insert(4, "reviewer_notes", "")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a visual audit pack for cross-label video conflicts.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--framing-report", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=180)
    parser.add_argument("--frame-ratios", default="0.2,0.5,0.8")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    artifacts_dir = artifacts_v2_dir(base)
    features_csv = Path(args.features) if args.features else artifacts_dir / "features_enriched.csv"
    framing_csv = Path(args.framing_report) if args.framing_report else artifacts_dir / "hand_framing_report.csv"
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else artifacts_dir / "label_audit")
    sheets_dir = ensure_dir(out_dir / "contact_sheets")
    ratios = [float(part.strip()) for part in str(args.frame_ratios).split(",") if part.strip()]

    df = pd.read_csv(features_csv)
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"].astype(str).str.lower() == "ok"].copy()
    conflicts = merge_framing(find_conflicts(df), framing_csv)
    if conflicts.empty:
        print("No cross-label conflicts found.", flush=True)
        return

    sheet_paths: dict[str, str] = {}
    for group_id, group in conflicts.groupby("content_group_id", sort=True):
        sheet_path = sheets_dir / f"{safe_slug(str(group_id), max_len=120)}.jpg"
        make_contact_sheet(group, base, sheet_path, ratios, (int(args.thumb_width), int(args.thumb_height)))
        sheet_paths[str(group_id)] = rel_to(sheet_path, base)
    conflicts["contact_sheet"] = conflicts["content_group_id"].astype(str).map(sheet_paths)
    conflicts["source_exists"] = conflicts["source_path"].astype(str).map(lambda p: resolve_source_path(p, base).exists())

    audit_cols = [
        "content_group_id",
        "conflict_type",
        "labels_in_group",
        "rows_in_group",
        "sample_id",
        "label",
        "source_set",
        "original_name",
        "normalized_stem",
        "duplicate_status",
        "content_duplicate_status",
        "source_path",
        "source_exists",
        "source_duration_s",
        "source_frames",
        "source_fps",
        "max_miss",
        "any_hand_coverage",
        "both_hands_coverage",
        "framing_level",
        "framing_risk_score",
        "max_edge_frame_any_ratio_05",
        "contact_sheet",
    ]
    audit = conflicts[[col for col in audit_cols if col in conflicts.columns]].copy()
    audit_csv = out_dir / "label_conflict_audit.csv"
    review_csv = out_dir / "label_review_template.csv"
    audit.to_csv(audit_csv, index=False)
    build_review_template(audit).to_csv(review_csv, index=False)

    summary = {
        "generated_at": utc_now_iso(),
        "features_csv": rel_to(features_csv, base),
        "framing_csv": rel_to(framing_csv, base) if framing_csv.exists() else "",
        "output_dir": rel_to(out_dir, base),
        "conflict_groups": int(audit["content_group_id"].nunique()),
        "conflict_rows": int(len(audit)),
        "labels": audit["label"].value_counts().to_dict(),
        "conflict_type": audit["conflict_type"].value_counts().to_dict(),
        "contact_sheets": len(sheet_paths),
    }
    write_json(out_dir / "label_audit_summary.json", summary)

    print("=== V2 LABEL AUDIT ===", flush=True)
    print(f"conflict_groups: {summary['conflict_groups']}", flush=True)
    print(f"conflict_rows: {summary['conflict_rows']}", flush=True)
    print(f"audit_csv: {audit_csv}", flush=True)
    print(f"review_template: {review_csv}", flush=True)
    print(f"contact_sheets: {sheets_dir}", flush=True)


if __name__ == "__main__":
    main()
