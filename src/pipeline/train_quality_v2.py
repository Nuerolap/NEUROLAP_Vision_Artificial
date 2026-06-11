import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .benchmark_models_v2 import (
    LABEL_ORDER,
    build_feature_sets,
    candidate_models,
    evaluate_candidate,
    flatten_result,
    prediction_frame,
    prepare_matrix,
)
from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json


def parse_csv_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend([part.strip() for part in str(value).split(",") if part.strip()])
    return out


def select_model(results: list[dict], tolerance: float) -> dict:
    best_macro = max(row["macro_f1_mean"] for row in results)
    pool = [row for row in results if row["macro_f1_mean"] >= best_macro - tolerance]
    return min(pool, key=lambda row: (row["model_size_kb"], -row["balanced_accuracy_mean"], -row["accuracy_mean"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the balanced NEUROLAP v2 classifier.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--feature-sets", nargs="*", default=["no_source_meta", "motion_quality", "compact_quality"])
    parser.add_argument("--candidates", nargs="*", default=["histgb_80_leaf15", "histgb_120_leaf7", "extra_trees_220_depth8", "svc_select80_c8"])
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--selection-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else artifacts_v2_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "features_enriched.csv"
    df = pd.read_csv(features_csv)
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"].astype(str).str.lower() == "ok"].copy()

    y = df["label"].astype(str).to_numpy()
    labels = [label for label in LABEL_ORDER if label in set(y)] or sorted(set(y))
    group_col = args.group_col or ("content_group_id" if "content_group_id" in df.columns else "group_id")
    groups = df[group_col].astype(str).to_numpy()
    n_splits = min(args.n_splits, min(Counter(y).values()), len(set(groups)))

    requested_feature_sets = set(parse_csv_values(args.feature_sets))
    feature_sets = {name: cols for name, cols in build_feature_sets(df).items() if name in requested_feature_sets}
    requested_candidates = set(parse_csv_values(args.candidates))
    models = {name: model for name, model in candidate_models(args.random_state).items() if not requested_candidates or name in requested_candidates}
    if not feature_sets:
        raise RuntimeError("No feature sets selected.")
    if not models:
        raise RuntimeError("No candidates selected.")

    print("=== V2 TRAIN QUALITY ===", flush=True)
    print(f"features: {features_csv}", flush=True)
    print(f"rows: {len(df)} labels: {dict(Counter(y))}", flush=True)
    print(f"group_col: {group_col} groups: {len(set(groups))}", flush=True)

    results = []
    fitted = {}
    oof_predictions = {}
    for feature_set, feature_cols in feature_sets.items():
        X = prepare_matrix(df, feature_cols)
        for candidate, estimator in models.items():
            print(f"[TRAIN] {feature_set} / {candidate}", flush=True)
            result, model, oof_pred = evaluate_candidate(estimator, X, y, groups, labels, n_splits, args.random_state)
            result.update({"feature_set": feature_set, "feature_count": len(feature_cols), "candidate": candidate, "feature_cols": feature_cols})
            fitted[(feature_set, candidate)] = model
            oof_predictions[(feature_set, candidate)] = oof_pred
            print(
                f"  acc={result['accuracy_mean']:.3f} bal={result['balanced_accuracy_mean']:.3f} "
                f"macro={result['macro_f1_mean']:.3f} size={result['model_size_kb']:.1f}KB",
                flush=True,
            )
            results.append(result)

    selected = select_model(results, args.selection_tolerance)
    selected_key = (selected["feature_set"], selected["candidate"])
    selected_model = fitted[selected_key]
    selected_oof = oof_predictions[selected_key]

    flat = pd.DataFrame([flatten_result(row) for row in results]).sort_values(
        ["macro_f1_mean", "balanced_accuracy_mean", "accuracy_mean", "model_size_kb"],
        ascending=[False, False, False, True],
    )
    flat.to_csv(out_dir / "train_quality_v2_models.csv", index=False)
    model_out = out_dir / "video_quality_v2.joblib"
    joblib.dump(
        {
            "model": selected_model,
            "feature_cols": selected["feature_cols"],
            "classes": labels,
            "selected_candidate": selected["candidate"],
            "selected_feature_set": selected["feature_set"],
            "metadata": {
                "trained_at": utc_now_iso(),
                "features_csv": str(features_csv),
                "group_col": group_col,
                "rows": int(len(df)),
                "labels": dict(Counter(y)),
                **{key: value for key, value in selected.items() if key != "feature_cols"},
            },
        },
        model_out,
    )

    preds = prediction_frame(selected_model, df, selected["feature_cols"])
    preds["oof_pred"] = selected_oof
    preds.to_csv(out_dir / "predictions_v2.csv", index=False)
    preds[preds["source_set"].astype(str) == "new_raw"].to_csv(out_dir / "predictions_new_v2.csv", index=False)
    errors = preds[preds["label"].astype(str) != preds["oof_pred"].astype(str)].copy()
    errors.to_csv(out_dir / "errors_v2.csv", index=False)
    write_json(
        out_dir / "metrics_v2.json",
        {
            "generated_at": utc_now_iso(),
            "features_csv": str(features_csv),
            "rows": int(len(df)),
            "labels": dict(Counter(y)),
            "group_col": group_col,
            "n_splits": int(n_splits),
            "selected": {key: value for key, value in selected.items() if key != "feature_cols"},
            "results": [{key: value for key, value in row.items() if key != "feature_cols"} for row in results],
        },
    )
    print("\n=== SELECTED V2 MODEL ===", flush=True)
    print(
        f"{selected['feature_set']} / {selected['candidate']} "
        f"acc={selected['accuracy_mean']:.3f} bal={selected['balanced_accuracy_mean']:.3f} "
        f"macro={selected['macro_f1_mean']:.3f} size={selected['model_size_kb']:.1f}KB",
        flush=True,
    )
    print(f"[OK] model -> {model_out}", flush=True)


if __name__ == "__main__":
    main()
