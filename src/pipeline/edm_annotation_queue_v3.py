from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json
from .edm_common_v3 import REP_COLUMNS, artifacts_v3_dir, normalize_review_status, truthy


LABEL_ORDER = ("amarillo", "rojo", "verde")
QUALITY_POINTS = {"high_risk": 28.0, "review": 20.0, "ok": 0.0}
SOURCE_POINTS = {"new_raw": 8.0, "legacy_raw": 4.0}
CONFLICT_POINTS = {
    "exact_content_cross_label": 65.0,
    "name_cross_label": 48.0,
    "cross_label_duplicate": 42.0,
}
PILOT_COLUMNS = (
    "pilot_selected",
    "pilot_rank",
    "pilot_within_label_rank",
    "pilot_priority_score",
    "pilot_reason",
    "pilot_generated_at",
)


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def to_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(clean_text(value))
    except Exception:
        return default
    if pd.isna(out):
        return default
    return out


def stable_jitter(sample_id: str) -> float:
    digest = hashlib.sha1(sample_id.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def is_complete(row: pd.Series) -> bool:
    status = clean_text(row.get("annotation_status", "")).casefold()
    if status == "complete":
        return True
    return all(normalize_review_status(row.get(col, "")) for col in REP_COLUMNS)


def load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Annotation CSV not found: {path}")
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = [col for col in ("sample_id", "label_actual", "calidad_video", "source_set") if col not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    for col in PILOT_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def load_conflict_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["sample_id", "conflict_type", "labels_in_group", "rows_in_group"])
    conflicts = pd.read_csv(path, keep_default_na=False, dtype=str)
    keep = [col for col in ("sample_id", "conflict_type", "labels_in_group", "rows_in_group") if col in conflicts.columns]
    if "sample_id" not in keep:
        return pd.DataFrame(columns=["sample_id", "conflict_type", "labels_in_group", "rows_in_group"])
    conflicts = conflicts[keep].drop_duplicates(subset=["sample_id"]).copy()
    for col in ("conflict_type", "labels_in_group", "rows_in_group"):
        if col not in conflicts.columns:
            conflicts[col] = ""
    return conflicts


def annotate_priority(row: pd.Series) -> tuple[float, str]:
    reasons: list[str] = []
    score = 0.0

    conflict_type = clean_text(row.get("conflict_type", "")).casefold()
    if conflict_type:
        score += CONFLICT_POINTS.get(conflict_type, 36.0)
        labels = clean_text(row.get("labels_in_group", ""))
        reasons.append(f"conflicto_etiqueta:{conflict_type}{'(' + labels + ')' if labels else ''}")

    quality = clean_text(row.get("calidad_video", "")).casefold()
    if quality in QUALITY_POINTS:
        score += QUALITY_POINTS[quality]
        if quality != "ok":
            reasons.append(f"calidad:{quality}")

    source_set = clean_text(row.get("source_set", "")).casefold()
    score += SOURCE_POINTS.get(source_set, 0.0)
    if source_set:
        reasons.append(f"fuente:{source_set}")

    duplicate_status = clean_text(row.get("duplicate_status", "")).casefold()
    content_duplicate_status = clean_text(row.get("content_duplicate_status", "")).casefold()
    if "cross" in duplicate_status or "cross" in content_duplicate_status:
        score += 14.0
        reasons.append("duplicado_cross_label")
    elif "duplicate" in duplicate_status or "duplicate" in content_duplicate_status:
        score += 4.0
        reasons.append("duplicado")

    risk = to_float(row.get("framing_risk_score", ""), 0.0)
    both_cov = to_float(row.get("both_hands_coverage", ""), 1.0)
    any_cov = to_float(row.get("any_hand_coverage", ""), 1.0)
    max_miss = to_float(row.get("max_miss", ""), 0.0)
    score += max(0.0, min(12.0, risk * 12.0))
    if both_cov < 0.35:
        score += 8.0
        reasons.append("baja_cobertura_ambas_manos")
    if any_cov < 0.90:
        score += 5.0
        reasons.append("mano_poco_visible")
    if max_miss >= 65.0:
        score += 5.0
        reasons.append("missing_alto")

    duration = to_float(row.get("duration_s", ""), 0.0)
    if duration and (duration < 12.0 or duration > 55.0):
        score += 3.0
        reasons.append("duracion_atipica")

    if truthy(row.get("source_exists", "1")) is False:
        score -= 100.0
        reasons.append("source_missing")

    score += stable_jitter(clean_text(row.get("sample_id", ""))) * 0.01
    if not reasons:
        reasons.append("representativo")
    return score, "; ".join(reasons)


def quotas_for(target_total: int, per_label: int | None) -> dict[str, int]:
    if per_label and per_label > 0:
        return {label: per_label for label in LABEL_ORDER}
    base = max(1, target_total // len(LABEL_ORDER))
    quotas = {label: base for label in LABEL_ORDER}
    remainder = max(0, target_total - base * len(LABEL_ORDER))
    for label in LABEL_ORDER[:remainder]:
        quotas[label] += 1
    return quotas


def quality_targets_for(label_quota: int) -> dict[str, int]:
    ok_target = max(1, int(round(label_quota * 0.45)))
    review_target = max(1, int(round(label_quota * 0.25)))
    high_risk_target = max(1, label_quota - ok_target - review_target)
    return {"ok": ok_target, "review": review_target, "high_risk": high_risk_target}


def select_label_group(group: pd.DataFrame, quota: int, *, balance_quality: bool) -> pd.DataFrame:
    sorted_group = group.sort_values(
        by=["pilot_priority_score", "quality_order", "source_set", "sample_id"],
        ascending=[False, True, True, True],
    )
    if not balance_quality:
        out = sorted_group.head(quota).copy()
        out["pilot_within_label_rank"] = range(1, len(out) + 1)
        return out

    selected_indices: list[int] = []

    def add_candidates(candidates: pd.DataFrame, amount: int) -> None:
        if amount <= 0:
            return
        for idx in candidates.index:
            if idx in selected_indices:
                continue
            selected_indices.append(idx)
            if len(selected_indices) >= quota:
                break
            amount -= 1
            if amount <= 0:
                break

    conflict_candidates = sorted_group[sorted_group["conflict_type"].astype(str).str.len().gt(0)]
    add_candidates(conflict_candidates, min(8, quota))

    quality_targets = quality_targets_for(quota)
    for quality in ("ok", "review", "high_risk"):
        selected_quality = sorted_group.loc[selected_indices, "calidad_video"].astype(str).str.casefold().eq(quality).sum() if selected_indices else 0
        need = max(0, quality_targets[quality] - int(selected_quality))
        quality_candidates = sorted_group[sorted_group["calidad_video"].astype(str).str.casefold().eq(quality)]
        add_candidates(quality_candidates, need)

    add_candidates(sorted_group, quota - len(selected_indices))
    out = sorted_group.loc[selected_indices].copy()
    out["pilot_within_label_rank"] = range(1, len(out) + 1)
    return out


def build_queue(
    df: pd.DataFrame,
    conflicts: pd.DataFrame,
    target_total: int,
    per_label: int | None,
    include_complete: bool,
    balance_quality: bool,
) -> pd.DataFrame:
    data = df.merge(conflicts, on="sample_id", how="left", suffixes=("", "_conflict"))
    for col in ("conflict_type", "labels_in_group", "rows_in_group"):
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str)
    data["label_actual"] = data["label_actual"].astype(str).str.strip().str.casefold()
    data["annotation_complete_for_queue"] = data.apply(is_complete, axis=1)
    if not include_complete:
        data = data[~data["annotation_complete_for_queue"].astype(bool)].copy()

    priority = data.apply(annotate_priority, axis=1)
    data["pilot_priority_score"] = [round(item[0], 4) for item in priority]
    data["pilot_reason"] = [item[1] for item in priority]
    data["label_order"] = data["label_actual"].map({label: idx for idx, label in enumerate(LABEL_ORDER)}).fillna(99).astype(int)
    data["quality_order"] = data["calidad_video"].astype(str).str.casefold().map({"high_risk": 0, "review": 1, "ok": 2}).fillna(9).astype(int)

    selected_parts = []
    quotas = quotas_for(target_total, per_label)
    generated_at = utc_now_iso()
    for label in LABEL_ORDER:
        group = data[data["label_actual"] == label].copy()
        group = select_label_group(group, quotas[label], balance_quality=balance_quality)
        selected_parts.append(group)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else data.head(0).copy()
    selected = selected.sort_values(
        by=["label_order", "pilot_priority_score", "quality_order", "sample_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    selected["pilot_selected"] = "1"
    selected["pilot_rank"] = range(1, len(selected) + 1)
    selected["pilot_generated_at"] = generated_at
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced pilot annotation queue for EDM v3.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--conflicts", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--target-total", type=int, default=120)
    parser.add_argument("--per-label", type=int, default=0)
    parser.add_argument("--include-complete", action="store_true")
    parser.add_argument("--no-quality-balance", action="store_true", help="Use pure priority ranking without quality quotas.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v3_dir(base))
    annotations_csv = Path(args.annotations).resolve() if args.annotations else out_dir / "edm_annotations_working.csv"
    if not annotations_csv.exists():
        annotations_csv = out_dir / "edm_annotation_template.csv"
    conflict_csv = Path(args.conflicts).resolve() if args.conflicts else artifacts_v2_dir(base) / "label_audit" / "label_conflict_audit.csv"
    out_csv = Path(args.out).resolve() if args.out else out_dir / "edm_annotation_pilot_queue.csv"

    df = load_annotations(annotations_csv)
    conflicts = load_conflict_table(conflict_csv)
    selected = build_queue(df, conflicts, args.target_total, args.per_label or None, args.include_complete, not args.no_quality_balance)
    ensure_dir(out_csv.parent)
    selected.to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "generated_at": utc_now_iso(),
        "annotations_csv": str(annotations_csv),
        "conflict_csv": str(conflict_csv),
        "out_csv": str(out_csv),
        "target_total": int(args.target_total),
        "selected_rows": int(len(selected)),
        "counts_by_label": dict(Counter(selected["label_actual"].astype(str))),
        "counts_by_quality": dict(Counter(selected["calidad_video"].astype(str))),
        "counts_by_source_set": dict(Counter(selected["source_set"].astype(str))),
        "conflict_rows": int(selected["conflict_type"].fillna("").astype(str).str.len().gt(0).sum()) if "conflict_type" in selected.columns else 0,
        "include_complete": bool(args.include_complete),
        "quality_balanced": not args.no_quality_balance,
    }
    write_json(out_dir / "edm_annotation_pilot_queue_summary.json", summary)

    print("=== EDM V3 PILOT QUEUE ===", flush=True)
    print(f"selected: {summary['selected_rows']} -> {out_csv}", flush=True)
    print(f"labels: {summary['counts_by_label']}", flush=True)
    print(f"quality: {summary['counts_by_quality']}", flush=True)
    print(f"source: {summary['counts_by_source_set']}", flush=True)
    print(f"conflict_rows: {summary['conflict_rows']}", flush=True)


if __name__ == "__main__":
    main()
