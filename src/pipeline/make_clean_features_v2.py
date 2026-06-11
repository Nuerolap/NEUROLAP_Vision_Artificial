import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json


def parse_csv_values(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in str(value).split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a v2 feature CSV excluding unresolved cross-label conflicts.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--conflict-types",
        default="exact_content_cross_label,name_only_cross_label,cross_label",
        help="Comma-separated conflict_type values to exclude.",
    )
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v2_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "features_enriched.csv"
    audit_csv = Path(args.audit) if args.audit else out_dir / "label_audit" / "label_conflict_audit.csv"
    out_csv = Path(args.out) if args.out else out_dir / "features_enriched_no_label_conflicts.csv"
    conflict_types = parse_csv_values(args.conflict_types)

    df = pd.read_csv(features_csv)
    audit = pd.read_csv(audit_csv)
    required_features = {"sample_id", "label", "content_group_id"}
    missing_features = required_features - set(df.columns)
    if missing_features:
        raise RuntimeError(f"Features CSV is missing required columns: {sorted(missing_features)}")
    required_audit = {"content_group_id", "conflict_type"}
    missing_audit = required_audit - set(audit.columns)
    if missing_audit:
        raise RuntimeError(f"Audit CSV is missing required columns: {sorted(missing_audit)}")

    selected_audit = audit[audit["conflict_type"].astype(str).isin(conflict_types)].copy()
    conflict_ids = set(selected_audit["content_group_id"].astype(str))
    clean = df[~df["content_group_id"].astype(str).isin(conflict_ids)].copy()

    ensure_dir(out_csv.parent)
    clean.to_csv(out_csv, index=False)
    summary = {
        "generated_at": utc_now_iso(),
        "features_csv": str(features_csv),
        "audit_csv": str(audit_csv),
        "out_csv": str(out_csv),
        "input_rows": int(len(df)),
        "output_rows": int(len(clean)),
        "excluded_rows": int(len(df) - len(clean)),
        "excluded_groups": int(len(conflict_ids)),
        "excluded_conflict_types": sorted(conflict_types),
        "labels_before": dict(Counter(df["label"].astype(str))),
        "labels_after": dict(Counter(clean["label"].astype(str))),
    }
    write_json(out_csv.with_suffix(".metadata.json"), summary)

    print("=== V2 CLEAN FEATURES ===", flush=True)
    print(f"input_rows: {summary['input_rows']} labels: {summary['labels_before']}", flush=True)
    print(f"excluded_rows: {summary['excluded_rows']} groups: {summary['excluded_groups']}", flush=True)
    print(f"output_rows: {summary['output_rows']} labels: {summary['labels_after']}", flush=True)
    print(f"[OK] clean_features -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
