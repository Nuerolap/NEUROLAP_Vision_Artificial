import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import LABELS, artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json


VALID_ACTIONS = {"keep", "relabel", "exclude", "downweight"}
ACTION_ALIASES = {
    "ok": "keep",
    "same": "keep",
    "mantener": "keep",
    "re_etiquetar": "relabel",
    "reetiquetar": "relabel",
    "corregir": "relabel",
    "drop": "exclude",
    "remove": "exclude",
    "descartar": "exclude",
    "excluir": "exclude",
    "bajar_peso": "downweight",
}
EXCLUDE_STATUSES = {"exclude", "excluded", "descartar", "descartado", "excluir", "excluido"}


def clean_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_token(value: object) -> str:
    return clean_cell(value).casefold().replace(" ", "_").replace("-", "_")


def clean_label(value: object) -> str:
    return clean_cell(value).casefold()


def parse_weight(value: object) -> float | None:
    text = clean_cell(value)
    if not text:
        return None
    try:
        weight = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid sample_weight '{text}'. Use a number such as 1.0, 0.5, or 0.0.") from exc
    if weight < 0:
        raise ValueError(f"Invalid sample_weight '{text}'. Weight must be >= 0.")
    return weight


def infer_action(row: pd.Series, original_label: str) -> str:
    action = clean_token(row.get("review_action", ""))
    action = ACTION_ALIASES.get(action, action)
    if action:
        return action

    status = clean_token(row.get("review_status", ""))
    reviewed_label = clean_label(row.get("reviewed_label", ""))
    weight = parse_weight(row.get("sample_weight", ""))

    if status in EXCLUDE_STATUSES:
        return "exclude"
    if weight is not None and weight <= 0:
        return "exclude"
    if reviewed_label and reviewed_label != original_label:
        return "relabel"
    if weight is not None and weight < 1:
        return "downweight"
    return "keep"


def has_review_content(row: pd.Series) -> bool:
    fields = ["review_status", "reviewed_label", "review_action", "sample_weight", "reviewer_notes"]
    return any(clean_cell(row.get(field, "")) for field in fields)


def load_review_decisions(review_csv: Path, feature_sample_ids: set[str], label_by_sample: dict[str, str]) -> pd.DataFrame:
    review = pd.read_csv(review_csv, keep_default_na=False)
    required = {"sample_id", "reviewed_label", "review_action", "sample_weight"}
    missing = required - set(review.columns)
    if missing:
        raise RuntimeError(f"Review template is missing required columns: {sorted(missing)}")

    review = review[review.apply(has_review_content, axis=1)].copy()
    if review.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "original_label",
                "reviewed_label",
                "review_action",
                "sample_weight",
                "exclude_from_training",
                "review_status",
                "reviewer_notes",
            ]
        )

    duplicated = sorted(review["sample_id"][review["sample_id"].duplicated()].astype(str).unique())
    if duplicated:
        raise RuntimeError(f"Duplicate reviewed sample_id values found: {duplicated[:10]}")

    decisions = []
    errors = []
    for _, row in review.iterrows():
        sample_id = clean_cell(row.get("sample_id", ""))
        if sample_id not in feature_sample_ids:
            errors.append(f"{sample_id}: sample_id does not exist in features CSV")
            continue

        original_label = label_by_sample[sample_id]
        reviewed_label = clean_label(row.get("reviewed_label", ""))
        if reviewed_label and reviewed_label not in LABELS:
            errors.append(f"{sample_id}: reviewed_label must be one of {LABELS}, got '{reviewed_label}'")
            continue

        try:
            action = infer_action(row, original_label)
            raw_weight = parse_weight(row.get("sample_weight", ""))
        except ValueError as exc:
            errors.append(f"{sample_id}: {exc}")
            continue

        if action not in VALID_ACTIONS:
            errors.append(f"{sample_id}: review_action must be one of {sorted(VALID_ACTIONS)}, got '{action}'")
            continue
        if action == "relabel" and not reviewed_label:
            errors.append(f"{sample_id}: relabel action requires reviewed_label")
            continue
        if action == "keep" and reviewed_label and reviewed_label != original_label:
            errors.append(f"{sample_id}: reviewed_label differs from original_label; use review_action=relabel")
            continue

        if action == "exclude":
            applied_weight = 0.0
            exclude = True
            final_label = reviewed_label or original_label
        elif action == "downweight":
            applied_weight = raw_weight if raw_weight is not None else 0.5
            exclude = False
            final_label = reviewed_label or original_label
        elif action == "relabel":
            applied_weight = raw_weight if raw_weight is not None else 1.0
            exclude = False
            final_label = reviewed_label
        else:
            applied_weight = raw_weight if raw_weight is not None else 1.0
            exclude = False
            final_label = original_label

        decisions.append(
            {
                "sample_id": sample_id,
                "original_label": original_label,
                "reviewed_label": final_label,
                "review_action": action,
                "sample_weight": float(applied_weight),
                "exclude_from_training": bool(exclude),
                "review_status": clean_cell(row.get("review_status", "")),
                "reviewer_notes": clean_cell(row.get("reviewer_notes", "")),
            }
        )

    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(f"Review validation failed with {len(errors)} error(s):\n{preview}")

    return pd.DataFrame(decisions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual label-review decisions to v2 features.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--review", default=None)
    parser.add_argument("--out", default=None, help="Training CSV output. Defaults to features_reviewed.csv.")
    parser.add_argument("--all-out", default=None, help="All rows output, including excluded samples.")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v2_dir(base))
    audit_dir = ensure_dir(out_dir / "label_audit")
    features_csv = Path(args.features) if args.features else out_dir / "features_enriched.csv"
    review_csv = Path(args.review) if args.review else audit_dir / "label_review_template.csv"
    training_out = Path(args.out) if args.out else out_dir / "features_reviewed.csv"
    all_out = Path(args.all_out) if args.all_out else out_dir / "features_reviewed_all.csv"

    df = pd.read_csv(features_csv)
    if "sample_id" not in df.columns or "label" not in df.columns:
        raise RuntimeError("Features CSV must contain sample_id and label columns.")
    if df["sample_id"].duplicated().any():
        duplicated = sorted(df["sample_id"][df["sample_id"].duplicated()].astype(str).unique())
        raise RuntimeError(f"Features CSV has duplicate sample_id values: {duplicated[:10]}")

    df = df.copy()
    df["original_label"] = df["label"].astype(str)
    df["label_review_action"] = "none"
    df["label_review_status"] = ""
    df["label_reviewer_notes"] = ""
    df["sample_weight"] = 1.0
    df["exclude_from_training"] = False

    sample_ids = set(df["sample_id"].astype(str))
    label_by_sample = dict(zip(df["sample_id"].astype(str), df["label"].astype(str), strict=True))
    decisions = load_review_decisions(review_csv, sample_ids, label_by_sample)

    decision_by_id = decisions.set_index("sample_id").to_dict("index") if not decisions.empty else {}
    for idx, row in df.iterrows():
        sample_id = str(row["sample_id"])
        decision = decision_by_id.get(sample_id)
        if not decision:
            continue
        df.at[idx, "label"] = decision["reviewed_label"]
        df.at[idx, "label_review_action"] = decision["review_action"]
        df.at[idx, "label_review_status"] = decision["review_status"]
        df.at[idx, "label_reviewer_notes"] = decision["reviewer_notes"]
        df.at[idx, "sample_weight"] = decision["sample_weight"]
        df.at[idx, "exclude_from_training"] = decision["exclude_from_training"]

    ensure_dir(training_out.parent)
    ensure_dir(all_out.parent)
    active = df[~df["exclude_from_training"].astype(bool)].copy()
    active.to_csv(training_out, index=False)
    df.to_csv(all_out, index=False)
    decisions.to_csv(audit_dir / "label_review_applied.csv", index=False)

    summary = {
        "generated_at": utc_now_iso(),
        "features_csv": str(features_csv),
        "review_csv": str(review_csv),
        "training_out": str(training_out),
        "all_out": str(all_out),
        "input_rows": int(len(df)),
        "training_rows": int(len(active)),
        "decisions": int(len(decisions)),
        "decision_actions": dict(Counter(decisions["review_action"])) if not decisions.empty else {},
        "labels_before": dict(Counter(df["original_label"].astype(str))),
        "labels_after_training": dict(Counter(active["label"].astype(str))),
        "excluded_rows": int(df["exclude_from_training"].astype(bool).sum()),
        "weighted_rows": int((active["sample_weight"].astype(float) != 1.0).sum()),
    }
    write_json(audit_dir / "label_review_summary.json", summary)

    print("=== V2 LABEL REVIEW APPLY ===", flush=True)
    print(f"decisions: {summary['decisions']} actions: {summary['decision_actions']}", flush=True)
    print(f"training_rows: {summary['training_rows']} labels: {summary['labels_after_training']}", flush=True)
    print(f"training_csv: {training_out}", flush=True)
    print(f"all_rows_csv: {all_out}", flush=True)


if __name__ == "__main__":
    main()
