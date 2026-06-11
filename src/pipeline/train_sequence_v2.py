import argparse
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.base import clone

from .accuracy_search_v2 import balanced_indices, maybe_jitter, model_size_kb
from .benchmark_models_v2 import LABEL_ORDER, build_feature_sets, prepare_matrix
from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json


META_COLUMNS = {
    "sample_id",
    "group_id",
    "content_group_id",
    "label",
    "source_set",
    "source_path",
    "original_name",
    "normalized_stem",
    "duplicate_status",
    "content_duplicate_status",
    "sequence_status",
    "sequence_error",
}


def sequence_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("seq") and pd.api.types.is_numeric_dtype(df[col])]


def candidate_models(random_state: int) -> dict[str, object]:
    return {
        "ridge_select60": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=60),
            RidgeClassifier(class_weight="balanced", alpha=1.0, random_state=random_state),
        ),
        "logreg_select50": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=50),
            LogisticRegression(C=1.5, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=random_state),
        ),
        "svc_select30_c4": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=30),
            SVC(C=4.0, gamma="scale", class_weight="balanced"),
        ),
        "svc_select40_c6": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=40),
            SVC(C=6.0, gamma="scale", class_weight="balanced"),
        ),
        "svc_select60_c6": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=60),
            SVC(C=6.0, gamma="scale", class_weight="balanced"),
        ),
        "extra_trees_depth8": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def parse_csv_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend([part.strip() for part in str(value).split(",") if part.strip()])
    return out


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_recall": {label: float(report[label]["recall"]) for label in labels},
        "per_class_precision": {label: float(report[label]["precision"]) for label in labels},
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, labels: list[str], strategy: str, n_splits: int, random_state: int) -> tuple[dict, object, np.ndarray]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.empty_like(y, dtype=object)
    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        rng = np.random.default_rng(random_state + fold_idx)
        chosen_rel = balanced_indices(y[train_idx], strategy=strategy, rng=rng)
        chosen_idx = train_idx[chosen_rel]
        X_train = maybe_jitter(X[chosen_idx], strategy, X[train_idx], rng)
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y[chosen_idx])
        pred = model.predict(X[val_idx])
        oof_pred[val_idx] = pred
        fold_scores.append(score_predictions(y[val_idx], pred, labels))

    rng = np.random.default_rng(random_state + 1000)
    final_rel = balanced_indices(y, strategy=strategy, rng=rng)
    X_final = maybe_jitter(X[final_rel], strategy, X, rng)
    fitted = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(X_final, y[final_rel])
    oof = score_predictions(y, oof_pred, labels)
    result = {
        "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_scores])),
        "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in fold_scores])),
        "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_scores])),
        "oof_accuracy": oof["accuracy"],
        "oof_balanced_accuracy": oof["balanced_accuracy"],
        "oof_macro_f1": oof["macro_f1"],
        "per_class_recall": oof["per_class_recall"],
        "per_class_precision": oof["per_class_precision"],
        "per_class_f1": oof["per_class_f1"],
        "confusion_matrix": oof["confusion_matrix"],
        "model_size_kb": model_size_kb(fitted),
    }
    return result, fitted, oof_pred


def flatten(result: dict) -> dict:
    row = {key: value for key, value in result.items() if key not in {"per_class_recall", "per_class_precision", "per_class_f1", "confusion_matrix", "feature_cols"}}
    for label, value in result["per_class_recall"].items():
        row[f"recall_{label}"] = value
    for label, value in result["per_class_precision"].items():
        row[f"precision_{label}"] = value
    for label, value in result["per_class_f1"].items():
        row[f"f1_{label}"] = value
    return row


def prediction_frame(model, df: pd.DataFrame, feature_cols: list[str], oof_pred: np.ndarray) -> pd.DataFrame:
    X = prepare_matrix(df, feature_cols)
    out_cols = [
        "sample_id",
        "group_id",
        "content_group_id",
        "label",
        "source_set",
        "original_name",
        "normalized_stem",
        "duplicate_status",
        "content_duplicate_status",
        "source_path",
    ]
    out = df[[col for col in out_cols if col in df.columns]].copy()
    out["pred"] = model.predict(X)
    out["oof_pred"] = oof_pred
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sequence-feature classifiers for NEUROLAP v2.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--sequence-features", default=None)
    parser.add_argument("--aggregate-features", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--feature-sets", nargs="*", default=["sequence_only", "sequence_plus_agg"])
    parser.add_argument("--strategies", nargs="*", default=["full", "cap_amarillo"])
    parser.add_argument("--candidates", nargs="*", default=["ridge_select60", "logreg_select50", "svc_select30_c4", "svc_select40_c6"])
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else artifacts_v2_dir(base))
    seq_csv = Path(args.sequence_features) if args.sequence_features else out_dir / "sequence_features_32.csv"
    agg_csv = Path(args.aggregate_features) if args.aggregate_features else out_dir / "features_enriched.csv"
    seq = pd.read_csv(seq_csv)
    agg = pd.read_csv(agg_csv)
    if "sequence_status" in seq.columns:
        seq = seq[seq["sequence_status"].astype(str).str.lower() == "ok"].copy()
    if "extraction_status" in agg.columns:
        agg = agg[agg["extraction_status"].astype(str).str.lower() == "ok"].copy()

    df = seq.merge(
        agg.drop(columns=[col for col in META_COLUMNS if col in agg.columns and col != "sample_id"], errors="ignore"),
        on="sample_id",
        how="inner",
        suffixes=("", "_agg"),
    )
    y = df["label"].astype(str).to_numpy()
    labels = [label for label in LABEL_ORDER if label in set(y)] or sorted(set(y))
    group_col = args.group_col or ("content_group_id" if "content_group_id" in df.columns else "group_id")
    groups = df[group_col].astype(str).to_numpy()
    n_splits = min(args.n_splits, min(Counter(y).values()), len(set(groups)))

    seq_cols = sequence_columns(df)
    agg_sets = build_feature_sets(df)
    feature_sets = {
        "sequence_only": seq_cols,
        "sequence_plus_agg": seq_cols + [col for col in agg_sets["no_source_meta"] if col not in seq_cols],
        "sequence_plus_quality": seq_cols + [col for col in agg_sets["motion_quality"] if col not in seq_cols],
    }
    requested_sets = set(parse_csv_values(args.feature_sets))
    feature_sets = {name: cols for name, cols in feature_sets.items() if name in requested_sets}
    models_all = candidate_models(args.random_state)
    requested_models = set(parse_csv_values(args.candidates))
    models = {name: model for name, model in models_all.items() if name in requested_models}
    strategies = parse_csv_values(args.strategies) or ["full"]

    print("=== V2 SEQUENCE TRAINING ===", flush=True)
    print(f"sequence_features: {seq_csv}", flush=True)
    print(f"rows: {len(df)} labels: {dict(Counter(y))}", flush=True)
    print(f"group_col: {group_col} groups: {len(set(groups))}", flush=True)
    print(f"feature_sets: {', '.join(f'{name}={len(cols)}' for name, cols in feature_sets.items())}", flush=True)

    results = []
    fitted_by_key = {}
    oof_by_key = {}
    for feature_set, feature_cols in feature_sets.items():
        X = prepare_matrix(df, feature_cols)
        for strategy in strategies:
            for candidate, estimator in models.items():
                print(f"[SEQ] {feature_set} / {strategy} / {candidate}", flush=True)
                try:
                    result, fitted, oof_pred = evaluate(estimator, X, y, groups, labels, strategy, n_splits, args.random_state)
                except Exception as exc:
                    print(f"  [SKIP] {exc}", flush=True)
                    continue
                result.update({"feature_set": feature_set, "strategy": strategy, "candidate": candidate, "feature_count": len(feature_cols), "feature_cols": feature_cols})
                fitted_by_key[(feature_set, strategy, candidate)] = fitted
                oof_by_key[(feature_set, strategy, candidate)] = oof_pred
                print(
                    f"  acc={result['accuracy_mean']:.3f} bal={result['balanced_accuracy_mean']:.3f} "
                    f"macro={result['macro_f1_mean']:.3f} size={result['model_size_kb']:.1f}KB "
                    f"y_rec={result['per_class_recall'].get('amarillo', 0):.3f}",
                    flush=True,
                )
                results.append(result)

    if not results:
        raise RuntimeError("No sequence candidates succeeded.")
    flat = pd.DataFrame([flatten(row) for row in results]).sort_values(
        ["accuracy_mean", "balanced_accuracy_mean", "macro_f1_mean", "model_size_kb"],
        ascending=[False, False, False, True],
    )
    flat.to_csv(out_dir / "sequence_search_v2.csv", index=False)
    best = max(results, key=lambda row: (row["accuracy_mean"], row["balanced_accuracy_mean"], row["macro_f1_mean"], -row["model_size_kb"]))
    best_key = (best["feature_set"], best["strategy"], best["candidate"])
    best_model = fitted_by_key[best_key]
    model_out = out_dir / "video_quality_v2_sequence_best.joblib"
    joblib.dump(
        {
            "model": best_model,
            "feature_cols": best["feature_cols"],
            "classes": labels,
            "selected_candidate": best["candidate"],
            "selected_feature_set": best["feature_set"],
            "selected_strategy": best["strategy"],
            "metadata": {
                "trained_at": utc_now_iso(),
                "sequence_features": str(seq_csv),
                "aggregate_features": str(agg_csv),
                "group_col": group_col,
                **{key: value for key, value in best.items() if key != "feature_cols"},
            },
        },
        model_out,
    )
    preds = prediction_frame(best_model, df, best["feature_cols"], oof_by_key[best_key])
    preds.to_csv(out_dir / "predictions_sequence_best_v2.csv", index=False)
    write_json(
        out_dir / "metrics_sequence_v2.json",
        {
            "generated_at": utc_now_iso(),
            "sequence_features": str(seq_csv),
            "aggregate_features": str(agg_csv),
            "rows": int(len(df)),
            "labels": dict(Counter(y)),
            "group_col": group_col,
            "selected": {key: value for key, value in best.items() if key != "feature_cols"},
            "results": [{key: value for key, value in row.items() if key != "feature_cols"} for row in results],
        },
    )
    print("\n=== SELECTED SEQUENCE MODEL ===", flush=True)
    print(
        f"{best['feature_set']} / {best['strategy']} / {best['candidate']} "
        f"acc={best['accuracy_mean']:.3f} bal={best['balanced_accuracy_mean']:.3f} "
        f"macro={best['macro_f1_mean']:.3f} size={best['model_size_kb']:.1f}KB",
        flush=True,
    )
    print(f"[OK] model -> {model_out}", flush=True)


if __name__ == "__main__":
    main()
