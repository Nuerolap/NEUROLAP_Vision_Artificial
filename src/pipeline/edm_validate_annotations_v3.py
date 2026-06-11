from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import ensure_dir, utc_now_iso, videos_base, write_json
from .edm_common_v3 import (
    COLOR_ORDER,
    ERROR_COLUMNS,
    EXERCISE_TIMING_COLUMNS,
    EXTRA_REP_COLUMNS,
    REP_COLUMNS,
    REP_STATUS_VALUES,
    artifacts_v3_dir,
    color_from_score,
    normalize_review_status,
    partial_count,
    score_from_repetitions,
    truthy,
)


def clean_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def is_annotated(row: pd.Series) -> bool:
    if clean_cell(row.get("score_0_6", "")) or clean_cell(row.get("color_final", "")):
        return True
    return any(clean_cell(row.get(col, "")) for col in REP_COLUMNS)


def parse_score(value: object) -> float | None:
    text = clean_cell(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_extra_reps(value: object) -> int:
    text = clean_cell(value)
    if not text:
        return 0
    text = text.replace("+", "")
    try:
        return max(0, int(float(text)))
    except ValueError:
        return -1


def parse_time_s(value: object) -> float | None:
    text = clean_cell(value)
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return float("nan")
    return out if out >= 0 else float("nan")


def validate_row(row: pd.Series) -> tuple[dict, list[str]]:
    errors = []
    out = row.to_dict()
    for col in REP_COLUMNS:
        value = normalize_review_status(row.get(col, ""))
        out[col] = value
        if value not in REP_STATUS_VALUES:
            errors.append(f"{col} invalid='{value}'")

    computed_score = score_from_repetitions(pd.Series(out))
    partials = partial_count(pd.Series(out))
    no_response = truthy(row.get("no_responde", ""))
    computed_color = color_from_score(computed_score, no_response=no_response)
    provided_score = parse_score(row.get("score_0_6", ""))
    provided_color = clean_cell(row.get("color_final", "")).casefold()
    if provided_color and provided_color not in COLOR_ORDER:
        errors.append(f"color_final invalid='{provided_color}'")
    if provided_score is not None and int(round(provided_score)) != computed_score:
        errors.append(f"score_0_6={provided_score} does not match repetitions={computed_score}")
    if provided_color and provided_color != computed_color:
        errors.append(f"color_final={provided_color} does not match computed_color={computed_color}")

    out["computed_score_0_6"] = int(computed_score)
    out["computed_color"] = computed_color
    out["partial_repetitions"] = int(partials)
    out["score_0_6_final"] = int(computed_score if provided_score is None else round(provided_score))
    out["color_final_validated"] = provided_color or computed_color
    out["annotation_complete"] = all(normalize_review_status(row.get(col, "")) in REP_STATUS_VALUES[1:] for col in REP_COLUMNS)
    for col in ERROR_COLUMNS:
        out[col] = truthy(row.get(col, ""))
    for col in EXTRA_REP_COLUMNS:
        if col == "extra_reps_note":
            out[col] = clean_cell(row.get(col, ""))
            continue
        value = parse_extra_reps(row.get(col, ""))
        if value < 0:
            errors.append(f"{col} invalid='{row.get(col, '')}'")
            value = 0
        out[col] = int(value)
    out["extra_reps_total"] = int(out.get("left_extra_reps", 0) + out.get("right_extra_reps", 0))
    start_s = parse_time_s(row.get("exercise_start_s", ""))
    end_s = parse_time_s(row.get("exercise_end_s", ""))
    if start_s is not None and (pd.isna(start_s)):
        errors.append(f"exercise_start_s invalid='{row.get('exercise_start_s', '')}'")
        start_s = None
    if end_s is not None and (pd.isna(end_s)):
        errors.append(f"exercise_end_s invalid='{row.get('exercise_end_s', '')}'")
        end_s = None
    if start_s is not None and end_s is not None and end_s <= start_s:
        errors.append(f"exercise_end_s={end_s} must be greater than exercise_start_s={start_s}")
    out["exercise_start_s"] = "" if start_s is None else round(float(start_s), 3)
    out["exercise_end_s"] = "" if end_s is None else round(float(end_s), 3)
    out["exercise_timing_note"] = clean_cell(row.get("exercise_timing_note", ""))
    out["annotation_error"] = "; ".join(errors)
    out["annotation_valid"] = not errors
    return out, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EDM v3 annotations against the 6-point rubric.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v3_dir(base))
    annotations_csv = Path(args.annotations) if args.annotations else out_dir / "edm_annotation_template.csv"
    out_csv = Path(args.out) if args.out else out_dir / "edm_annotations_validated.csv"
    df = pd.read_csv(annotations_csv, keep_default_na=False)
    missing = [col for col in ("sample_id", "label_actual", *REP_COLUMNS) if col not in df.columns]
    if missing:
        raise RuntimeError(f"Annotation CSV is missing required columns: {missing}")
    if df["sample_id"].duplicated().any():
        duplicated = sorted(df.loc[df["sample_id"].duplicated(), "sample_id"].astype(str).unique())
        raise RuntimeError(f"Duplicate sample_id values in annotations: {duplicated[:10]}")

    rows = []
    row_errors = []
    for _, row in df.iterrows():
        validated, errors = validate_row(row)
        if errors:
            row_errors.append({"sample_id": row["sample_id"], "errors": "; ".join(errors)})
        rows.append(validated)
    out = pd.DataFrame(rows)
    ensure_dir(out_csv.parent)
    out.to_csv(out_csv, index=False)

    annotated = out[out.apply(is_annotated, axis=1)].copy()
    complete = out[out["annotation_complete"].astype(bool)].copy()
    label_conflicts = complete[
        complete["color_final_validated"].astype(str).str.len().gt(0)
        & complete["label_actual"].astype(str).str.casefold().ne(complete["color_final_validated"].astype(str).str.casefold())
    ].copy()
    label_conflicts.to_csv(out_dir / "edm_annotation_label_conflicts.csv", index=False)
    pd.DataFrame(row_errors).to_csv(out_dir / "edm_annotation_errors.csv", index=False)
    summary = {
        "generated_at": utc_now_iso(),
        "annotations_csv": str(annotations_csv),
        "out_csv": str(out_csv),
        "rows": int(len(out)),
        "annotated_rows": int(len(annotated)),
        "complete_rows": int(len(complete)),
        "valid_rows": int(out["annotation_valid"].astype(bool).sum()),
        "error_rows": int((~out["annotation_valid"].astype(bool)).sum()),
        "label_conflicts": int(len(label_conflicts)),
        "computed_color_counts": dict(Counter(complete["computed_color"].astype(str))) if not complete.empty else {},
        "validated_color_counts": dict(Counter(complete["color_final_validated"].astype(str))) if not complete.empty else {},
    }
    write_json(out_dir / "edm_annotation_validation_report.json", summary)

    print("=== EDM V3 ANNOTATION VALIDATION ===", flush=True)
    print(f"rows: {summary['rows']} annotated: {summary['annotated_rows']} complete: {summary['complete_rows']}", flush=True)
    print(f"errors: {summary['error_rows']} label_conflicts: {summary['label_conflicts']}", flush=True)
    print(f"[OK] validated -> {out_csv}", flush=True)
    print(f"[OK] report -> {out_dir / 'edm_annotation_validation_report.json'}", flush=True)
    if summary["error_rows"] and not args.allow_incomplete:
        raise SystemExit("Annotation validation found errors. Review edm_annotation_errors.csv.")


if __name__ == "__main__":
    main()
