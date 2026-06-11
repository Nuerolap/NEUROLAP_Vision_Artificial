import argparse
import io
import pickle
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json


LABEL_ORDER = ["amarillo", "rojo", "verde"]
TEXT_METADATA_COLUMNS = {
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
    "content_fingerprint",
    "keypoints_path",
    "processed_at",
    "extraction_status",
    "error",
}
NUMERIC_METADATA_COLUMNS = {
    "target_width",
    "target_height",
    "model_complexity",
    "processing_seconds",
    "source_width",
    "source_height",
    "frames_read",
}
TEMPORAL_TOKENS = (
    "_lm",
    "_bin",
    "_phase",
    "_seq",
    "coverage",
    "miss",
    "gap",
    "centroid",
    "bbox",
    "spread",
    "speed",
    "motion",
    "jitter",
    "accel",
    "edge",
    "margin",
    "framing",
    "oob",
    "hand_distance",
)
QUALITY_TOKENS = (
    "coverage",
    "miss",
    "gap",
    "speed",
    "motion",
    "jitter",
    "bbox",
    "spread",
    "duration",
    "fps",
    "frames_used",
    "edge",
    "margin",
    "framing",
    "oob",
)


def numeric_feature_columns(df: pd.DataFrame, *, drop_source_metadata: bool = True) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    excluded = set(TEXT_METADATA_COLUMNS)
    if drop_source_metadata:
        excluded.update(NUMERIC_METADATA_COLUMNS)
    return [col for col in numeric_cols if col not in excluded]


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    all_numeric = numeric_feature_columns(df, drop_source_metadata=False)
    no_source_meta = numeric_feature_columns(df, drop_source_metadata=True)
    temporal_only = [col for col in no_source_meta if any(token in col for token in TEMPORAL_TOKENS)]
    motion_quality = [col for col in no_source_meta if any(token in col for token in QUALITY_TOKENS)]
    compact_quality = [
        col
        for col in motion_quality
        if "_lm" not in col and "_bin" not in col and "_phase" not in col and not col.endswith("_raw")
    ]
    return {
        "all_numeric": all_numeric,
        "no_source_meta": no_source_meta,
        "temporal_only": temporal_only or no_source_meta,
        "motion_quality": motion_quality or no_source_meta,
        "compact_quality": compact_quality or motion_quality or no_source_meta,
    }


def prepare_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if not feature_cols:
        raise ValueError("feature_cols is empty")
    work = df.reindex(columns=feature_cols).copy()
    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    medians = work.median(numeric_only=True)
    work = work.fillna(medians).fillna(0.0)
    return work.astype(np.float32).to_numpy()


def model_size_kb(model) -> float:
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return float(len(buffer.getvalue()) / 1024.0)


def candidate_models(random_state: int) -> dict[str, object]:
    return {
        "histgb_80_leaf15": HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.06,
            l2_regularization=0.05,
            max_leaf_nodes=15,
            class_weight="balanced",
            random_state=random_state,
        ),
        "histgb_120_leaf7": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            l2_regularization=0.1,
            max_leaf_nodes=7,
            class_weight="balanced",
            random_state=random_state,
        ),
        "extra_trees_220_depth8": ExtraTreesClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "rf_220_depth8": RandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "svc_select80_c8": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=80),
            SVC(C=8.0, gamma="scale", class_weight="balanced"),
        ),
        "svc_select60_c8": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=60),
            SVC(C=8.0, gamma="scale", class_weight="balanced"),
        ),
    }


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


def evaluate_candidate(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, labels: list[str], n_splits: int, random_state: int) -> tuple[dict, object, np.ndarray]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.empty_like(y, dtype=object)
    fold_scores = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for train_idx, val_idx in cv.split(X, y, groups):
        model = clone(estimator)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train_idx], y[train_idx])
        fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        pred = model.predict(X[val_idx])
        predict_seconds += time.perf_counter() - started
        oof_pred[val_idx] = pred
        fold_scores.append(score_predictions(y[val_idx], pred, labels))

    fitted = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(X, y)

    oof = score_predictions(y, oof_pred, labels)
    result = {
        "accuracy_mean": float(np.mean([row["accuracy"] for row in fold_scores])),
        "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in fold_scores])),
        "macro_f1_mean": float(np.mean([row["macro_f1"] for row in fold_scores])),
        "weighted_f1_mean": float(np.mean([row["weighted_f1"] for row in fold_scores])),
        "oof_accuracy": oof["accuracy"],
        "oof_balanced_accuracy": oof["balanced_accuracy"],
        "oof_macro_f1": oof["macro_f1"],
        "oof_weighted_f1": oof["weighted_f1"],
        "per_class_recall": oof["per_class_recall"],
        "per_class_precision": oof["per_class_precision"],
        "per_class_f1": oof["per_class_f1"],
        "confusion_matrix": oof["confusion_matrix"],
        "fit_seconds_total": float(fit_seconds),
        "predict_seconds_total": float(predict_seconds),
        "predict_ms_per_sample": float((predict_seconds / max(len(y), 1)) * 1000.0),
        "model_size_kb": model_size_kb(fitted),
    }
    return result, fitted, oof_pred


def flatten_result(result: dict) -> dict:
    row = {key: value for key, value in result.items() if key not in {"per_class_recall", "per_class_precision", "per_class_f1", "confusion_matrix", "feature_cols"}}
    for label, value in result.get("per_class_recall", {}).items():
        row[f"recall_{label}"] = value
    for label, value in result.get("per_class_precision", {}).items():
        row[f"precision_{label}"] = value
    for label, value in result.get("per_class_f1", {}).items():
        row[f"f1_{label}"] = value
    return row


def prediction_frame(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
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
    return out


def parse_csv_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend([part.strip() for part in str(value).split(",") if part.strip()])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark lightweight classifiers for NEUROLAP v2 features.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--feature-sets", nargs="*", default=["no_source_meta", "motion_quality", "compact_quality"])
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
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

    print("=== V2 BENCHMARK ===", flush=True)
    print(f"features: {features_csv}", flush=True)
    print(f"rows: {len(df)} labels: {dict(Counter(y))}", flush=True)
    print(f"group_col: {group_col} groups: {len(set(groups))}", flush=True)

    results = []
    fitted = {}
    for feature_set, feature_cols in feature_sets.items():
        X = prepare_matrix(df, feature_cols)
        for candidate, estimator in models.items():
            print(f"[BENCH] {feature_set} / {candidate}", flush=True)
            result, model, _ = evaluate_candidate(estimator, X, y, groups, labels, n_splits, args.random_state)
            result.update({"feature_set": feature_set, "feature_count": len(feature_cols), "candidate": candidate, "feature_cols": feature_cols})
            fitted[(feature_set, candidate)] = model
            print(
                f"  acc={result['accuracy_mean']:.3f} bal={result['balanced_accuracy_mean']:.3f} "
                f"macro={result['macro_f1_mean']:.3f} size={result['model_size_kb']:.1f}KB",
                flush=True,
            )
            results.append(result)

    if not results:
        raise RuntimeError("No benchmark candidates ran.")

    best = max(results, key=lambda row: (row["balanced_accuracy_mean"], row["accuracy_mean"], row["macro_f1_mean"], -row["model_size_kb"]))
    best_model = fitted[(best["feature_set"], best["candidate"])]
    flat = pd.DataFrame([flatten_result(row) for row in results]).sort_values(
        ["balanced_accuracy_mean", "accuracy_mean", "macro_f1_mean", "model_size_kb"],
        ascending=[False, False, False, True],
    )
    flat.to_csv(out_dir / "benchmark_models_v2.csv", index=False)
    write_json(
        out_dir / "benchmark_models_v2.json",
        {
            "generated_at": utc_now_iso(),
            "features_csv": str(features_csv),
            "rows": int(len(df)),
            "labels": dict(Counter(y)),
            "group_col": group_col,
            "best": {key: value for key, value in best.items() if key != "feature_cols"},
            "results": [{key: value for key, value in row.items() if key != "feature_cols"} for row in results],
        },
    )
    joblib.dump(
        {
            "model": best_model,
            "feature_cols": best["feature_cols"],
            "classes": labels,
            "selected_candidate": best["candidate"],
            "selected_feature_set": best["feature_set"],
            "metadata": {key: value for key, value in best.items() if key != "feature_cols"},
        },
        out_dir / "video_quality_v2_benchmark_best.joblib",
    )
    prediction_frame(best_model, df, best["feature_cols"]).to_csv(out_dir / "predictions_benchmark_best.csv", index=False)
    print("\n=== TOP RESULTS ===", flush=True)
    print(flat.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
