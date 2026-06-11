import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .common import (
    LABELS,
    artifacts_v2_dir,
    clean_label,
    ensure_dir,
    is_video_file,
    label_from_path,
    normalize_stem,
    rel_to,
    safe_slug,
    utc_now_iso,
    videos_base,
    write_json,
)


DEFAULT_COLUMNS = [
    "sample_id",
    "group_id",
    "label",
    "source_set",
    "source_path",
    "original_name",
    "normalized_stem",
    "duplicate_status",
]


def collect_sources(base: Path) -> list[dict]:
    roots = [
        ("legacy_raw", base / "training-videos"),
        ("new_raw", base / "nuevos videos"),
    ]
    rows: list[dict] = []
    for source_set, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not is_video_file(path):
                continue
            label = label_from_path(path)
            if label not in LABELS:
                continue
            normalized = normalize_stem(path.name)
            rows.append(
                {
                    "label": clean_label(label),
                    "source_set": source_set,
                    "source_path": rel_to(path, base),
                    "original_name": path.name,
                    "normalized_stem": normalized,
                    "group_id": safe_slug(normalized),
                }
            )
    return rows


def assign_ids_and_duplicates(rows: list[dict]) -> list[dict]:
    stem_counts = Counter(row["normalized_stem"] for row in rows)
    labels_by_stem: dict[str, set[str]] = defaultdict(set)
    sources_by_stem: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_stem[row["normalized_stem"]].add(row["label"])
        sources_by_stem[row["normalized_stem"]].add(row["source_set"])

    base_id_counts: Counter[str] = Counter()
    for row in rows:
        id_base = safe_slug(f"{row['source_set']}_{row['label']}_{row['normalized_stem']}")
        base_id_counts[id_base] += 1
        suffix = "" if base_id_counts[id_base] == 1 else f"__dup{base_id_counts[id_base]:02d}"
        row["sample_id"] = f"{id_base}{suffix}"

        stem = row["normalized_stem"]
        if len(labels_by_stem[stem]) > 1:
            status = "cross_label_duplicate"
        elif stem_counts[stem] > 1:
            status = "same_label_duplicate"
        else:
            status = "unique"
        if len(sources_by_stem[stem]) > 1:
            status = f"{status}|cross_source"
        row["duplicate_status"] = status

    return rows


def build_conflict_report(rows: list[dict]) -> pd.DataFrame:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["normalized_stem"]].append(row)

    report_rows = []
    for stem, items in sorted(grouped.items()):
        if len(items) == 1:
            continue
        labels = sorted({item["label"] for item in items})
        sources = sorted({item["source_set"] for item in items})
        statuses = sorted({item["duplicate_status"] for item in items})
        report_rows.append(
            {
                "normalized_stem": stem,
                "count": len(items),
                "labels": ",".join(labels),
                "source_sets": ",".join(sources),
                "duplicate_status": "|".join(statuses),
                "sample_ids": "|".join(item["sample_id"] for item in items),
                "source_paths": "|".join(item["source_path"] for item in items),
            }
        )
    return pd.DataFrame(report_rows)


def build_index(base: Path) -> pd.DataFrame:
    rows = assign_ids_and_duplicates(collect_sources(base))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    return df[DEFAULT_COLUMNS].sort_values(["source_set", "label", "normalized_stem", "source_path"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the NEUROLAP v2 source video index.")
    parser.add_argument("--base", default=str(videos_base()), help="Videos directory.")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    parser.add_argument("--conflicts-out", default=None, help="Duplicate/conflict report CSV path.")
    parser.add_argument("--metadata-out", default=None, help="Index metadata JSON path.")
    parser.add_argument("--expect-count", type=int, default=None, help="Fail if the indexed count differs.")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without writing files.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = artifacts_v2_dir(base)
    out_csv = Path(args.out) if args.out else out_dir / "video_index.csv"
    conflicts_csv = Path(args.conflicts_out) if args.conflicts_out else out_dir / "conflict_report.csv"
    metadata_json = Path(args.metadata_out) if args.metadata_out else out_dir / "index_metadata.json"

    df = build_index(base)
    conflicts = build_conflict_report(df.to_dict("records"))

    summary = {
        "generated_at": utc_now_iso(),
        "base": str(base),
        "total_videos": int(len(df)),
        "counts_by_source_set": df["source_set"].value_counts().to_dict() if not df.empty else {},
        "counts_by_label": df["label"].value_counts().to_dict() if not df.empty else {},
        "duplicate_status_counts": df["duplicate_status"].value_counts().to_dict() if not df.empty else {},
        "conflict_rows": int(len(conflicts)),
        "sample_id_duplicates": int(df["sample_id"].duplicated().sum()) if not df.empty else 0,
        "missing_paths": int(sum(not (base / p).exists() for p in df["source_path"])) if not df.empty else 0,
    }

    print("=== V2 VIDEO INDEX ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if args.expect_count is not None and len(df) != args.expect_count:
        raise SystemExit(f"Expected {args.expect_count} videos, found {len(df)}")
    if summary["sample_id_duplicates"]:
        raise SystemExit("sample_id duplicates detected")
    if summary["missing_paths"]:
        raise SystemExit("missing source paths detected")

    if args.dry_run:
        return

    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    conflicts.to_csv(conflicts_csv, index=False)
    write_json(metadata_json, summary)
    print(f"[OK] index -> {out_csv}")
    print(f"[OK] conflicts -> {conflicts_csv}")
    print(f"[OK] metadata -> {metadata_json}")


if __name__ == "__main__":
    main()

