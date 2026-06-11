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
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .benchmark_models_v2 import build_feature_sets, prepare_matrix
from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json

try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
except Exception:  # pragma: no cover - optional dependency
    BalancedRandomForestClassifier = None
    EasyEnsembleClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


class EncodedClassifier(BaseEstimator, ClassifierMixin):
    """Adapter for estimators that require numeric labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder()
        encoded = self.encoder_.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        if hasattr(self.estimator_, "set_params"):
            params = self.estimator_.get_params()
            updates = {}
            if "num_class" in params:
                updates["num_class"] = len(self.encoder_.classes_)
            if updates:
                self.estimator_.set_params(**updates)
        self.estimator_.fit(X, encoded)
        self.classes_ = self.encoder_.classes_
        return self

    def predict(self, X):
        encoded = self.estimator_.predict(X)
        encoded = np.asarray(encoded).astype(int)
        return self.encoder_.inverse_transform(encoded)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def model_size_kb(model) -> float:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return float(len(buf.getvalue()) / 1024.0)


def parse_csv_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend([part.strip() for part in str(value).split(",") if part.strip()])
    return out


def candidate_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "svc_select80_c5": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=80),
            SVC(C=5.0, gamma="scale", class_weight="balanced"),
        ),
        "svc_select80_c8": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=80),
            SVC(C=8.0, gamma="scale", class_weight="balanced"),
        ),
        "svc_select80_c12": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=80),
            SVC(C=12.0, gamma="scale", class_weight="balanced"),
        ),
        "svc_select60_c8": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=60),
            SVC(C=8.0, gamma="scale", class_weight="balanced"),
        ),
        "svc_full_c3": make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", class_weight="balanced")),
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
        "extra_trees_360_depth10": ExtraTreesClassifier(
            n_estimators=360,
            max_depth=10,
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
        "rf_360_depth10": RandomForestClassifier(
            n_estimators=360,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "mlp_select80_64": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=80),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=1200,
                early_stopping=True,
                n_iter_no_change=25,
                random_state=random_state,
            ),
        ),
        "vote_svc_hist_rf": VotingClassifier(
            estimators=[
                (
                    "svc",
                    make_pipeline(
                        StandardScaler(),
                        SelectPercentile(f_classif, percentile=80),
                        SVC(C=8.0, gamma="scale", class_weight="balanced"),
                    ),
                ),
                (
                    "hist",
                    HistGradientBoostingClassifier(
                        max_iter=80,
                        learning_rate=0.06,
                        l2_regularization=0.05,
                        max_leaf_nodes=15,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=220,
                        max_depth=8,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ],
            voting="hard",
        ),
    }

    if BalancedRandomForestClassifier is not None:
        models["balanced_rf_220_depth8"] = BalancedRandomForestClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            replacement=True,
        )
    if EasyEnsembleClassifier is not None:
        models["easy_ensemble_80"] = EasyEnsembleClassifier(
            n_estimators=80,
            random_state=random_state,
            n_jobs=-1,
        )
    if LGBMClassifier is not None:
        models["lgbm_120_leaf7_bal"] = LGBMClassifier(
            objective="multiclass",
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=7,
            max_depth=4,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        models["lgbm_180_leaf15"] = LGBMClassifier(
            objective="multiclass",
            n_estimators=180,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=6,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=0.3,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    if XGBClassifier is not None:
        models["xgb_120_depth2"] = EncodedClassifier(
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=120,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
        )
        models["xgb_180_depth3"] = EncodedClassifier(
            XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=180,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss",
            )
        )
    return models


def balanced_indices(
    y: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
    yellow_label: str = "amarillo",
) -> np.ndarray:
    if strategy == "full":
        return np.arange(len(y))
    labels, counts = np.unique(y, return_counts=True)
    count_by_label = dict(zip(labels, counts))
    target = int(count_by_label.get(yellow_label, min(counts)))
    selected = []
    for label in labels:
        idx = np.where(y == label)[0]
        replace = len(idx) < target
        picked = rng.choice(idx, size=target, replace=replace)
        selected.extend(picked.tolist())
    rng.shuffle(selected)
    return np.array(selected, dtype=int)


def maybe_jitter(X: np.ndarray, strategy: str, train_reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if "jitter" not in strategy:
        return X
    scale_text = strategy.rsplit("_", 1)[-1]
    try:
        scale = float(scale_text)
    except ValueError:
        scale = 0.02
    std = np.nanstd(train_reference, axis=0)
    std = np.where(np.isfinite(std), std, 0.0)
    return X + rng.normal(0.0, std * scale, size=X.shape)


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


def evaluate_candidate(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    strategy: str,
    n_splits: int,
    random_state: int,
) -> tuple[dict, object]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.empty_like(y, dtype=object)
    fold_scores = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        rng = np.random.default_rng(random_state + fold_idx)
        chosen_rel = balanced_indices(y[train_idx], strategy=strategy, rng=rng)
        chosen_idx = train_idx[chosen_rel]
        X_train = maybe_jitter(X[chosen_idx], strategy, X[train_idx], rng)
        y_train = y[chosen_idx]
        model = clone(estimator)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(X[val_idx])
        predict_seconds += time.perf_counter() - start
        oof_pred[val_idx] = pred
        fold_scores.append(score_predictions(y[val_idx], pred, labels))

    rng = np.random.default_rng(random_state + 1000)
    final_rel = balanced_indices(y, strategy=strategy, rng=rng)
    X_final = maybe_jitter(X[final_rel], strategy, X, rng)
    y_final = y[final_rel]
    fitted = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(X_final, y_final)

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
        "train_rows_final": int(len(y_final)),
        "train_counts_final": dict(Counter(y_final)),
    }
    return result, fitted


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


def flatten(result: dict) -> dict:
    row = {key: value for key, value in result.items() if key not in {"per_class_recall", "per_class_precision", "per_class_f1", "confusion_matrix", "feature_cols", "folds"}}
    for label, value in result["per_class_recall"].items():
        row[f"recall_{label}"] = value
    for label, value in result["per_class_precision"].items():
        row[f"precision_{label}"] = value
    for label, value in result["per_class_f1"].items():
        row[f"f1_{label}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy-focused model search for NEUROLAP v2 features.")
    parser.add_argument("--base", default=str(videos_base()), help="Videos directory.")
    parser.add_argument("--features", default=None, help="Input features CSV.")
    parser.add_argument("--out-dir", default=None, help="Artifact output directory.")
    parser.add_argument("--group-col", default=None, help="Grouping column. Defaults to content_group_id if present.")
    parser.add_argument("--feature-sets", nargs="*", default=["no_source_meta", "temporal_only", "motion_quality"])
    parser.add_argument("--strategies", nargs="*", default=["full", "cap_amarillo", "cap_amarillo_jitter_0.02"])
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else artifacts_v2_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "features_enriched.csv"
    df = pd.read_csv(features_csv)
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"].astype(str).str.lower() == "ok"].copy()

    labels = sorted(df["label"].astype(str).unique().tolist())
    y = df["label"].astype(str).values
    group_col = args.group_col or ("content_group_id" if "content_group_id" in df.columns else "group_id")
    groups = df[group_col].astype(str).values
    n_splits = min(int(args.n_splits), min(Counter(y).values()), len(set(groups)))

    feature_sets_all = build_feature_sets(df)
    requested_feature_sets = set(parse_csv_values(args.feature_sets))
    feature_sets = {name: cols for name, cols in feature_sets_all.items() if name in requested_feature_sets}
    if not feature_sets:
        raise RuntimeError(f"No matching feature sets: {sorted(requested_feature_sets)}")

    models_all = candidate_models(args.random_state)
    requested_candidates = set(parse_csv_values(args.candidates))
    models = {name: model for name, model in models_all.items() if not requested_candidates or name in requested_candidates}
    if not models:
        raise RuntimeError(f"No matching candidates: {sorted(requested_candidates)}")

    strategies = parse_csv_values(args.strategies) or ["full"]
    print("=== V2 ACCURACY SEARCH ===", flush=True)
    print(f"features: {features_csv}", flush=True)
    print(f"rows: {len(df)}", flush=True)
    print(f"labels: {dict(Counter(y))}", flush=True)
    print(f"group_col: {group_col}", flush=True)
    print(f"groups: {len(set(groups))}", flush=True)
    print(f"feature_sets: {', '.join(f'{name}={len(cols)}' for name, cols in feature_sets.items())}", flush=True)
    print(f"strategies: {strategies}", flush=True)
    print(f"candidates: {len(models)}", flush=True)

    results = []
    fitted_by_key = {}
    for feature_set, feature_cols in feature_sets.items():
        X = prepare_matrix(df, feature_cols)
        for strategy in strategies:
            for candidate, estimator in models.items():
                print(f"[SEARCH] {feature_set} / {strategy} / {candidate}", flush=True)
                try:
                    result, fitted = evaluate_candidate(
                        estimator,
                        X,
                        y,
                        groups,
                        labels,
                        strategy=strategy,
                        n_splits=n_splits,
                        random_state=int(args.random_state),
                    )
                except Exception as exc:
                    print(f"  [SKIP] {exc}", flush=True)
                    continue
                result.update(
                    {
                        "feature_set": feature_set,
                        "feature_count": int(len(feature_cols)),
                        "strategy": strategy,
                        "candidate": candidate,
                        "feature_cols": feature_cols,
                    }
                )
                fitted_by_key[(feature_set, strategy, candidate)] = fitted
                print(
                    f"  acc={result['accuracy_mean']:.3f} "
                    f"bal={result['balanced_accuracy_mean']:.3f} "
                    f"macro={result['macro_f1_mean']:.3f} "
                    f"size={result['model_size_kb']:.1f}KB "
                    f"pred={result['predict_ms_per_sample']:.4f}ms",
                    flush=True,
                )
                results.append(result)

    if not results:
        raise RuntimeError("No successful candidates.")

    flat = pd.DataFrame([flatten(row) for row in results]).sort_values(
        ["accuracy_mean", "balanced_accuracy_mean", "macro_f1_mean", "model_size_kb"],
        ascending=[False, False, False, True],
    )
    csv_out = out_dir / "accuracy_search_models.csv"
    json_out = out_dir / "accuracy_search_models.json"
    flat.to_csv(csv_out, index=False)

    best_accuracy = max(results, key=lambda row: (row["accuracy_mean"], row["balanced_accuracy_mean"], row["macro_f1_mean"]))
    best_balanced = max(results, key=lambda row: (row["balanced_accuracy_mean"], row["accuracy_mean"], row["macro_f1_mean"]))
    best_yellow_limited = max(
        [row for row in results if row["strategy"].startswith("cap_amarillo")] or results,
        key=lambda row: (row["accuracy_mean"], row["balanced_accuracy_mean"], row["macro_f1_mean"]),
    )

    selected = best_accuracy
    selected_key = (selected["feature_set"], selected["strategy"], selected["candidate"])
    selected_model = fitted_by_key[selected_key]
    model_out = out_dir / "video_quality_v2_accuracy_best.joblib"
    joblib.dump(
        {
            "model": selected_model,
            "feature_cols": selected["feature_cols"],
            "classes": labels,
            "selected_candidate": selected["candidate"],
            "selected_feature_set": selected["feature_set"],
            "selected_strategy": selected["strategy"],
            "metadata": {
                "trained_at": utc_now_iso(),
                "features_csv": str(features_csv),
                "group_col": group_col,
                "rows": int(len(df)),
                "labels": dict(Counter(y)),
                "accuracy_mean": selected["accuracy_mean"],
                "balanced_accuracy_mean": selected["balanced_accuracy_mean"],
                "macro_f1_mean": selected["macro_f1_mean"],
                "model_size_kb": selected["model_size_kb"],
            },
        },
        model_out,
    )
    preds = prediction_frame(selected_model, df, selected["feature_cols"])
    preds.to_csv(out_dir / "predictions_accuracy_best.csv", index=False)
    preds[preds["source_set"].astype(str) == "new_raw"].to_csv(out_dir / "predictions_new_accuracy_best.csv", index=False)

    yellow_key = (
        best_yellow_limited["feature_set"],
        best_yellow_limited["strategy"],
        best_yellow_limited["candidate"],
    )
    yellow_model = fitted_by_key[yellow_key]
    yellow_model_out = out_dir / "video_quality_v2_yellow_limited.joblib"
    joblib.dump(
        {
            "model": yellow_model,
            "feature_cols": best_yellow_limited["feature_cols"],
            "classes": labels,
            "selected_candidate": best_yellow_limited["candidate"],
            "selected_feature_set": best_yellow_limited["feature_set"],
            "selected_strategy": best_yellow_limited["strategy"],
            "metadata": {
                "trained_at": utc_now_iso(),
                "features_csv": str(features_csv),
                "group_col": group_col,
                "rows": int(len(df)),
                "labels": dict(Counter(y)),
                "accuracy_mean": best_yellow_limited["accuracy_mean"],
                "balanced_accuracy_mean": best_yellow_limited["balanced_accuracy_mean"],
                "macro_f1_mean": best_yellow_limited["macro_f1_mean"],
                "model_size_kb": best_yellow_limited["model_size_kb"],
            },
        },
        yellow_model_out,
    )
    yellow_preds = prediction_frame(yellow_model, df, best_yellow_limited["feature_cols"])
    yellow_preds.to_csv(out_dir / "predictions_yellow_limited.csv", index=False)
    yellow_preds[yellow_preds["source_set"].astype(str) == "new_raw"].to_csv(out_dir / "predictions_new_yellow_limited.csv", index=False)

    summary = {
        "generated_at": utc_now_iso(),
        "features_csv": str(features_csv),
        "rows": int(len(df)),
        "labels": dict(Counter(y)),
        "group_col": group_col,
        "groups": int(len(set(groups))),
        "n_splits": int(n_splits),
        "best_accuracy": {key: best_accuracy[key] for key in best_accuracy if key != "feature_cols"},
        "best_balanced_accuracy": {key: best_balanced[key] for key in best_balanced if key != "feature_cols"},
        "best_yellow_limited": {key: best_yellow_limited[key] for key in best_yellow_limited if key != "feature_cols"},
        "results": [{key: row[key] for key in row if key != "feature_cols"} for row in results],
    }
    write_json(json_out, summary)

    print("\n=== TOP ACCURACY RESULTS ===", flush=True)
    columns = [
        "feature_set",
        "strategy",
        "candidate",
        "feature_count",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "model_size_kb",
        "predict_ms_per_sample",
        "recall_amarillo",
        "recall_rojo",
        "recall_verde",
    ]
    print(flat.head(args.top)[[col for col in columns if col in flat.columns]].to_string(index=False), flush=True)
    print("\n=== SELECTION ===", flush=True)
    print(
        f"best_accuracy={selected['feature_set']} / {selected['strategy']} / {selected['candidate']} "
        f"acc={selected['accuracy_mean']:.3f} "
        f"bal={selected['balanced_accuracy_mean']:.3f} "
        f"macro={selected['macro_f1_mean']:.3f} "
        f"size={selected['model_size_kb']:.1f}KB",
        flush=True,
    )
    print(
        f"best_yellow_limited={best_yellow_limited['feature_set']} / {best_yellow_limited['strategy']} / {best_yellow_limited['candidate']} "
        f"acc={best_yellow_limited['accuracy_mean']:.3f} "
        f"bal={best_yellow_limited['balanced_accuracy_mean']:.3f} "
        f"macro={best_yellow_limited['macro_f1_mean']:.3f}",
        flush=True,
    )
    print(f"[OK] model -> {model_out}", flush=True)
    print(f"[OK] yellow_limited_model -> {yellow_model_out}", flush=True)
    print(f"[OK] csv -> {csv_out}", flush=True)
    print(f"[OK] json -> {json_out}", flush=True)


if __name__ == "__main__":
    main()
