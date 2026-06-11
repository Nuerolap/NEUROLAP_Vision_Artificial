import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .accuracy_search_v2 import balanced_indices, candidate_models, maybe_jitter, model_size_kb
from .benchmark_models_v2 import LABEL_ORDER, build_feature_sets, prepare_matrix
from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json


CANONICAL_MODULE = "Videos.pipeline.calibrate_flat_bias_v2"
sys.modules.setdefault(CANONICAL_MODULE, sys.modules[__name__])


class BiasAdjustedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, class_bias: dict[str, float] | None = None):
        self.estimator = estimator
        self.class_bias = class_bias or {}

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.classes_ = np.asarray(getattr(self.estimator_, "classes_", sorted(set(y))), dtype=object)
        return self

    def _scores(self, X):
        estimator = getattr(self, "estimator_", self.estimator)
        if hasattr(estimator, "decision_function"):
            raw = estimator.decision_function(X)
        elif hasattr(estimator, "predict_proba"):
            raw = estimator.predict_proba(X)
        else:
            pred = estimator.predict(X)
            classes = np.asarray(getattr(estimator, "classes_", sorted(set(pred))), dtype=object)
            raw = np.zeros((len(pred), len(classes)), dtype=float)
            for idx, label in enumerate(classes):
                raw[:, idx] = pred == label
            return raw, classes
        classes = np.asarray(getattr(estimator, "classes_", []), dtype=object)
        raw = np.asarray(raw, dtype=float)
        if raw.ndim == 1 and len(classes) == 2:
            raw = np.column_stack([-raw, raw])
        return raw, classes

    def decision_function(self, X):
        raw, classes = self._scores(X)
        bias = np.array([float(self.class_bias.get(str(label), 0.0)) for label in classes], dtype=float)
        return raw + bias

    def predict(self, X):
        raw, classes = self._scores(X)
        adjusted = raw + np.array([float(self.class_bias.get(str(label), 0.0)) for label in classes], dtype=float)
        return classes[np.argmax(adjusted, axis=1)]


BiasAdjustedClassifier.__module__ = CANONICAL_MODULE


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


def oof_scores(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, labels: list[str], strategy: str, n_splits: int, random_state: int) -> tuple[np.ndarray, list[str]]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = np.zeros((len(y), len(labels)), dtype=float)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        rng = np.random.default_rng(random_state + fold_idx)
        chosen_rel = balanced_indices(y[train_idx], strategy=strategy, rng=rng)
        chosen_idx = train_idx[chosen_rel]
        X_train = maybe_jitter(X[chosen_idx], strategy, X[train_idx], rng)
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y[chosen_idx])
        raw = model.decision_function(X[val_idx]) if hasattr(model, "decision_function") else model.predict_proba(X[val_idx])
        raw = np.asarray(raw, dtype=float)
        classes = [str(label) for label in model.classes_]
        for label_idx, label in enumerate(labels):
            scores[val_idx, label_idx] = raw[:, classes.index(label)]
    return scores, labels


def bias_grid(labels: list[str], low: float, high: float, steps: int) -> list[dict[str, float]]:
    values = np.linspace(low, high, steps)
    anchor = labels[-1]
    others = labels[:-1]
    grids = np.meshgrid(*([values] * len(others)), indexing="ij")
    biases = []
    for combo in zip(*(grid.ravel() for grid in grids)):
        row = {label: float(value) for label, value in zip(others, combo)}
        row[anchor] = 0.0
        biases.append(row)
    return biases


def flatten_metrics(metrics: dict) -> dict:
    row = {key: value for key, value in metrics.items() if key not in {"per_class_recall", "per_class_precision", "per_class_f1", "confusion_matrix"}}
    for label, value in metrics["per_class_recall"].items():
        row[f"recall_{label}"] = value
    for label, value in metrics["per_class_precision"].items():
        row[f"precision_{label}"] = value
    for label, value in metrics["per_class_f1"].items():
        row[f"f1_{label}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate class biases for a flat multiclass SVC using grouped OOF scores.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--feature-set", default="no_source_meta")
    parser.add_argument("--candidate", default="svc_select60_c8")
    parser.add_argument("--strategy", default="cap_amarillo")
    parser.add_argument("--min-yellow-recall", type=float, default=0.50)
    parser.add_argument("--bias-low", type=float, default=-1.0)
    parser.add_argument("--bias-high", type=float, default=1.0)
    parser.add_argument("--bias-steps", type=int, default=41)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--suffix", default="biased")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v2_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "features_enriched.csv"
    model_path = Path(args.base_model) if args.base_model else out_dir / "video_quality_v2_yellow_limited.joblib"
    df = pd.read_csv(features_csv)
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"].astype(str).str.lower() == "ok"].copy()
    y = df["label"].astype(str).to_numpy()
    labels = [label for label in LABEL_ORDER if label in set(y)] or sorted(set(y))
    group_col = "content_group_id" if "content_group_id" in df.columns else "group_id"
    groups = df[group_col].astype(str).to_numpy()
    n_splits = min(args.n_splits, min(Counter(y).values()), len(set(groups)))
    feature_cols = build_feature_sets(df)[args.feature_set]
    X = prepare_matrix(df, feature_cols)

    estimator = candidate_models(args.random_state)[args.candidate]
    scores, labels = oof_scores(estimator, X, y, groups, labels, args.strategy, n_splits, args.random_state)
    rows = []
    for bias in bias_grid(labels, args.bias_low, args.bias_high, args.bias_steps):
        bias_vector = np.array([bias.get(label, 0.0) for label in labels], dtype=float)
        pred = np.array(labels, dtype=object)[np.argmax(scores + bias_vector, axis=1)]
        metrics = score_predictions(y, pred, labels)
        row = flatten_metrics(metrics)
        for label in labels:
            row[f"bias_{label}"] = bias.get(label, 0.0)
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table[table.get("recall_amarillo", 0.0) >= args.min_yellow_recall]
    if eligible.empty:
        eligible = table
        sort_cols = ["recall_amarillo", "accuracy", "balanced_accuracy", "macro_f1"]
    else:
        sort_cols = ["accuracy", "balanced_accuracy", "macro_f1", "precision_amarillo"]
    selected = eligible.sort_values(sort_cols, ascending=False).iloc[0].to_dict()
    class_bias = {label: float(selected.get(f"bias_{label}", 0.0)) for label in labels}

    bundle = joblib.load(model_path)
    base_model = bundle["model"]
    biased_model = BiasAdjustedClassifier(base_model, class_bias=class_bias)
    biased_model.classes_ = np.asarray(labels, dtype=object)
    suffix = args.suffix.strip().replace(" ", "_")
    model_out = out_dir / f"video_quality_v2_yellow_limited_{suffix}.joblib"
    joblib.dump(
        {
            "model": biased_model,
            "feature_cols": bundle.get("feature_cols", feature_cols),
            "classes": labels,
            "selected_candidate": args.candidate,
            "selected_feature_set": args.feature_set,
            "selected_strategy": args.strategy,
            "class_bias": class_bias,
            "metadata": {
                "trained_at": utc_now_iso(),
                "features_csv": str(features_csv),
                "base_model": str(model_path),
                "group_col": group_col,
                "min_yellow_recall": float(args.min_yellow_recall),
                "selected": selected,
                "model_size_kb": model_size_kb(biased_model),
            },
        },
        model_out,
    )
    preds = df[
        [
            col
            for col in [
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
            if col in df.columns
        ]
    ].copy()
    preds["pred"] = biased_model.predict(X)
    preds.to_csv(out_dir / f"predictions_yellow_limited_{suffix}.csv", index=False)
    if "source_set" in preds.columns:
        preds[preds["source_set"].astype(str) == "new_raw"].to_csv(
            out_dir / f"predictions_new_yellow_limited_{suffix}.csv",
            index=False,
        )
    table.to_csv(out_dir / f"flat_bias_search_{suffix}.csv", index=False)
    write_json(
        out_dir / f"metrics_yellow_limited_{suffix}.json",
        {
            "generated_at": utc_now_iso(),
            "features_csv": str(features_csv),
            "base_model": str(model_path),
            "output_model": str(model_out),
            "rows": int(len(df)),
            "labels": dict(Counter(y)),
            "group_col": group_col,
            "selected": selected,
            "class_bias": class_bias,
            "model_size_kb": model_size_kb(biased_model),
        },
    )
    print("=== FLAT BIAS CALIBRATION ===", flush=True)
    print(f"bias: {class_bias}", flush=True)
    print(
        f"acc={selected['accuracy']:.3f} bal={selected['balanced_accuracy']:.3f} "
        f"macro={selected['macro_f1']:.3f} rec_y={selected.get('recall_amarillo', 0.0):.3f} "
        f"prec_y={selected.get('precision_amarillo', 0.0):.3f}",
        flush=True,
    )
    print(f"[OK] model -> {model_out}", flush=True)


if __name__ == "__main__":
    main()
