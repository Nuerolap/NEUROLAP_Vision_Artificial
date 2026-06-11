import argparse
import io
import pickle
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .benchmark_models_v2 import LABEL_ORDER, build_feature_sets, prepare_matrix
from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


CANONICAL_MODULE = "Videos.pipeline.train_two_stage_v2"
sys.modules.setdefault(CANONICAL_MODULE, sys.modules[__name__])

YELLOW = "amarillo"
NOT_YELLOW = "no_amarillo"
LABELS = ["amarillo", "rojo", "verde"]


class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, stage1, stage2, yellow_threshold: float, feature_cols: list[str], metadata: dict | None = None):
        self.stage1 = stage1
        self.stage2 = stage2
        self.yellow_threshold = float(yellow_threshold)
        self.feature_cols = list(feature_cols)
        self.metadata = metadata or {}
        self.classes_ = np.array(LABELS, dtype=object)

    def yellow_score(self, X: np.ndarray) -> np.ndarray:
        return yellow_score(self.stage1, X)

    def predict_details(self, X: np.ndarray) -> pd.DataFrame:
        scores = self.yellow_score(X)
        stage2_pred = self.stage2.predict(X)
        pred = np.where(scores >= self.yellow_threshold, YELLOW, stage2_pred)
        return pd.DataFrame(
            {
                "pred": pred,
                "yellow_score": scores,
                "yellow_threshold": self.yellow_threshold,
                "stage1_pred": np.where(scores >= self.yellow_threshold, YELLOW, NOT_YELLOW),
                "stage2_pred": stage2_pred,
            }
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_details(X)["pred"].to_numpy()


TwoStageClassifier.__module__ = CANONICAL_MODULE


def model_size_kb(model) -> float:
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return float(len(buffer.getvalue()) / 1024.0)


def parse_csv_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend([part.strip() for part in str(value).split(",") if part.strip()])
    return out


def stage1_candidates(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "s1_logreg_l2": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=75),
            LogisticRegression(C=1.5, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=random_state),
        ),
        "s1_svc_select60_c6": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=60),
            SVC(C=6.0, gamma="scale", class_weight="balanced"),
        ),
        "s1_svc_select80_c8": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=80),
            SVC(C=8.0, gamma="scale", class_weight="balanced"),
        ),
        "s1_histgb_80_leaf7": HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            l2_regularization=0.1,
            max_leaf_nodes=7,
            class_weight="balanced",
            random_state=random_state,
        ),
        "s1_extra_trees_180_depth7": ExtraTreesClassifier(
            n_estimators=180,
            max_depth=7,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if LGBMClassifier is not None:
        models["s1_lgbm_100_leaf7"] = LGBMClassifier(
            objective="binary",
            n_estimators=100,
            learning_rate=0.04,
            num_leaves=7,
            max_depth=4,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    return models


def stage2_candidates(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "s2_logreg_l2": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=75),
            LogisticRegression(C=1.5, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=random_state),
        ),
        "s2_svc_select60_c6": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=60),
            SVC(C=6.0, gamma="scale", class_weight="balanced"),
        ),
        "s2_svc_select80_c8": make_pipeline(
            StandardScaler(),
            SelectPercentile(f_classif, percentile=80),
            SVC(C=8.0, gamma="scale", class_weight="balanced"),
        ),
        "s2_histgb_100_leaf7": HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.045,
            l2_regularization=0.1,
            max_leaf_nodes=7,
            class_weight="balanced",
            random_state=random_state,
        ),
        "s2_extra_trees_220_depth8": ExtraTreesClassifier(
            n_estimators=220,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if LGBMClassifier is not None:
        models["s2_lgbm_120_leaf7"] = LGBMClassifier(
            objective="binary",
            n_estimators=120,
            learning_rate=0.035,
            num_leaves=7,
            max_depth=4,
            min_child_samples=8,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    return models


def choose_indices(y: np.ndarray, strategy: str, rng: np.random.Generator, target_label: str | None = None) -> np.ndarray:
    if strategy == "full":
        return np.arange(len(y), dtype=int)
    labels, counts = np.unique(y, return_counts=True)
    if target_label and target_label in set(labels):
        target = int(dict(zip(labels, counts))[target_label])
    else:
        target = int(np.min(counts))
    selected: list[int] = []
    for label in labels:
        idx = np.where(y == label)[0]
        replace = len(idx) < target
        picked = rng.choice(idx, size=target, replace=replace)
        selected.extend(picked.tolist())
    rng.shuffle(selected)
    return np.array(selected, dtype=int)


def fit_with_strategy(estimator, X: np.ndarray, y: np.ndarray, strategy: str, rng: np.random.Generator, target_label: str | None = None):
    rel = choose_indices(y, strategy, rng, target_label=target_label)
    model = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X[rel], y[rel])
    return model, rel


def estimator_classes(estimator) -> np.ndarray | None:
    classes = getattr(estimator, "classes_", None)
    if classes is not None:
        return np.asarray(classes)
    if hasattr(estimator, "steps") and estimator.steps:
        return getattr(estimator.steps[-1][1], "classes_", None)
    return None


def yellow_score(estimator, X: np.ndarray) -> np.ndarray:
    classes = estimator_classes(estimator)
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
            classes = estimator_classes(estimator)
            if classes is not None and YELLOW in list(classes):
                return proba[:, list(classes).index(YELLOW)].astype(float)
        except Exception:
            pass
    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(X)
        decision = np.asarray(decision, dtype=float)
        if decision.ndim > 1:
            if classes is not None and YELLOW in list(classes):
                return decision[:, list(classes).index(YELLOW)]
            return decision[:, 0]
        if classes is not None and len(classes) == 2:
            classes_list = list(classes)
            return decision if classes_list[1] == YELLOW else -decision
        return decision
    pred = estimator.predict(X)
    return (np.asarray(pred) == YELLOW).astype(float)


def threshold_grid(scores: np.ndarray, steps: int = 401) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([0.0], dtype=float)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    linear = np.linspace(lo - 1e-9, hi + 1e-9, steps)
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, steps))
    return np.unique(np.concatenate([linear, quantiles]))


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] = LABELS) -> dict:
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


def evaluate_thresholds(y: np.ndarray, scores: np.ndarray, stage2_pred: np.ndarray, min_yellow_recall: float) -> dict:
    best = None
    fallback = None
    for threshold in threshold_grid(scores):
        pred = np.where(scores >= threshold, YELLOW, stage2_pred)
        metrics = score_predictions(y, pred)
        yellow_pred_rate = float(np.mean(pred == YELLOW))
        row = {
            **metrics,
            "yellow_threshold": float(threshold),
            "yellow_pred_rate": yellow_pred_rate,
            "meets_yellow_recall_floor": bool(metrics["per_class_recall"][YELLOW] >= min_yellow_recall),
        }
        if fallback is None or (
            row["per_class_recall"][YELLOW],
            row["balanced_accuracy"],
            row["macro_f1"],
            row["accuracy"],
        ) > (
            fallback["per_class_recall"][YELLOW],
            fallback["balanced_accuracy"],
            fallback["macro_f1"],
            fallback["accuracy"],
        ):
            fallback = row
        if not row["meets_yellow_recall_floor"]:
            continue
        if best is None or (
            row["accuracy"],
            row["balanced_accuracy"],
            row["macro_f1"],
            row["per_class_precision"][YELLOW],
            -row["yellow_pred_rate"],
        ) > (
            best["accuracy"],
            best["balanced_accuracy"],
            best["macro_f1"],
            best["per_class_precision"][YELLOW],
            -best["yellow_pred_rate"],
        ):
            best = row
    return best or fallback


def evaluate_pair(
    stage1_estimator,
    stage2_estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    stage1_strategy: str,
    stage2_strategy: str,
    n_splits: int,
    random_state: int,
    min_yellow_recall: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    y_stage1 = np.where(y == YELLOW, YELLOW, NOT_YELLOW)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_scores = np.zeros(len(y), dtype=float)
    oof_stage2 = np.empty(len(y), dtype=object)
    fit_seconds = 0.0
    predict_seconds = 0.0

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        rng = np.random.default_rng(random_state + fold_idx)
        started = time.perf_counter()
        stage1_model, _ = fit_with_strategy(stage1_estimator, X[train_idx], y_stage1[train_idx], stage1_strategy, rng, target_label=YELLOW)
        non_yellow_train = train_idx[y[train_idx] != YELLOW]
        stage2_model, _ = fit_with_strategy(stage2_estimator, X[non_yellow_train], y[non_yellow_train], stage2_strategy, rng)
        fit_seconds += time.perf_counter() - started

        started = time.perf_counter()
        oof_scores[val_idx] = yellow_score(stage1_model, X[val_idx])
        oof_stage2[val_idx] = stage2_model.predict(X[val_idx])
        predict_seconds += time.perf_counter() - started

    selected_threshold = evaluate_thresholds(y, oof_scores, oof_stage2, min_yellow_recall)
    pred = np.where(oof_scores >= selected_threshold["yellow_threshold"], YELLOW, oof_stage2)
    result = {
        **selected_threshold,
        "oof_pred": pred,
        "oof_yellow_score": oof_scores,
        "oof_stage2_pred": oof_stage2,
        "fit_seconds_total": float(fit_seconds),
        "predict_seconds_total": float(predict_seconds),
        "predict_ms_per_sample": float((predict_seconds / max(len(y), 1)) * 1000.0),
    }
    return result, oof_scores, oof_stage2


def fit_final(
    stage1_estimator,
    stage2_estimator,
    X: np.ndarray,
    y: np.ndarray,
    stage1_strategy: str,
    stage2_strategy: str,
    threshold: float,
    feature_cols: list[str],
    random_state: int,
    metadata: dict,
) -> TwoStageClassifier:
    rng = np.random.default_rng(random_state + 9000)
    y_stage1 = np.where(y == YELLOW, YELLOW, NOT_YELLOW)
    stage1_model, stage1_rel = fit_with_strategy(stage1_estimator, X, y_stage1, stage1_strategy, rng, target_label=YELLOW)
    non_yellow_idx = np.where(y != YELLOW)[0]
    stage2_model, stage2_rel = fit_with_strategy(stage2_estimator, X[non_yellow_idx], y[non_yellow_idx], stage2_strategy, rng)
    final_metadata = dict(metadata)
    final_metadata["stage1_train_counts"] = dict(Counter(y_stage1[stage1_rel]))
    final_metadata["stage2_train_counts"] = dict(Counter(y[non_yellow_idx][stage2_rel]))
    return TwoStageClassifier(stage1_model, stage2_model, threshold, feature_cols, metadata=final_metadata)


def flatten_result(result: dict) -> dict:
    row = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "per_class_recall",
            "per_class_precision",
            "per_class_f1",
            "confusion_matrix",
            "feature_cols",
            "oof_pred",
            "oof_yellow_score",
            "oof_stage2_pred",
        }
    }
    for label, value in result.get("per_class_recall", {}).items():
        row[f"recall_{label}"] = value
    for label, value in result.get("per_class_precision", {}).items():
        row[f"precision_{label}"] = value
    for label, value in result.get("per_class_f1", {}).items():
        row[f"f1_{label}"] = value
    return row


def prediction_frame(model: TwoStageClassifier, df: pd.DataFrame, X: np.ndarray, oof: dict | None = None) -> pd.DataFrame:
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
        "max_miss",
        "any_hand_coverage",
        "both_hands_coverage",
        "seq_any_hand_coverage",
        "seq_both_hands_coverage",
        "left_edge_frame_any_ratio_05",
        "right_edge_frame_any_ratio_05",
        "left_oob_frame_any_ratio",
        "right_oob_frame_any_ratio",
    ]
    out = df[[col for col in out_cols if col in df.columns]].copy()
    details = model.predict_details(X)
    out = pd.concat([out.reset_index(drop=True), details.reset_index(drop=True)], axis=1)
    if oof:
        out["oof_pred"] = oof["oof_pred"]
        out["oof_yellow_score"] = oof["oof_yellow_score"]
        out["oof_stage2_pred"] = oof["oof_stage2_pred"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a two-stage NEUROLAP v2 classifier: amarillo first, then rojo/verde.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--group-col", default=None)
    parser.add_argument("--feature-sets", nargs="*", default=["no_source_meta", "motion_quality"])
    parser.add_argument("--stage1-candidates", nargs="*", default=None)
    parser.add_argument("--stage2-candidates", nargs="*", default=None)
    parser.add_argument("--stage1-strategies", nargs="*", default=["full", "cap_amarillo"])
    parser.add_argument("--stage2-strategies", nargs="*", default=["full", "balanced"])
    parser.add_argument("--min-yellow-recall", type=float, default=0.55)
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
    group_col = args.group_col or ("content_group_id" if "content_group_id" in df.columns else "group_id")
    groups = df[group_col].astype(str).to_numpy()
    n_splits = min(args.n_splits, min(Counter(y).values()), len(set(groups)))

    feature_sets_all = build_feature_sets(df)
    requested_feature_sets = set(parse_csv_values(args.feature_sets))
    feature_sets = {name: cols for name, cols in feature_sets_all.items() if name in requested_feature_sets}
    stage1_all = stage1_candidates(args.random_state)
    stage2_all = stage2_candidates(args.random_state)
    requested_stage1 = set(parse_csv_values(args.stage1_candidates))
    requested_stage2 = set(parse_csv_values(args.stage2_candidates))
    stage1_models = {name: model for name, model in stage1_all.items() if not requested_stage1 or name in requested_stage1}
    stage2_models = {name: model for name, model in stage2_all.items() if not requested_stage2 or name in requested_stage2}
    stage1_strategies = parse_csv_values(args.stage1_strategies) or ["full"]
    stage2_strategies = parse_csv_values(args.stage2_strategies) or ["full"]

    print("=== V2 TWO-STAGE TRAINING ===", flush=True)
    print(f"features: {features_csv}", flush=True)
    print(f"rows: {len(df)} labels: {dict(Counter(y))}", flush=True)
    print(f"group_col: {group_col} groups: {len(set(groups))}", flush=True)
    print(f"min_yellow_recall: {args.min_yellow_recall:.3f}", flush=True)

    results = []
    best_oof = None
    for feature_set, feature_cols in feature_sets.items():
        X = prepare_matrix(df, feature_cols)
        for stage1_strategy in stage1_strategies:
            for stage2_strategy in stage2_strategies:
                for stage1_name, stage1_estimator in stage1_models.items():
                    for stage2_name, stage2_estimator in stage2_models.items():
                        print(f"[TWO-STAGE] {feature_set} / {stage1_strategy}+{stage2_strategy} / {stage1_name} -> {stage2_name}", flush=True)
                        try:
                            result, _, _ = evaluate_pair(
                                stage1_estimator,
                                stage2_estimator,
                                X,
                                y,
                                groups,
                                stage1_strategy,
                                stage2_strategy,
                                n_splits,
                                args.random_state,
                                args.min_yellow_recall,
                            )
                        except Exception as exc:
                            print(f"  [SKIP] {exc}", flush=True)
                            continue
                        result.update(
                            {
                                "feature_set": feature_set,
                                "feature_count": len(feature_cols),
                                "stage1_candidate": stage1_name,
                                "stage2_candidate": stage2_name,
                                "stage1_strategy": stage1_strategy,
                                "stage2_strategy": stage2_strategy,
                                "feature_cols": feature_cols,
                            }
                        )
                        print(
                            f"  acc={result['accuracy']:.3f} bal={result['balanced_accuracy']:.3f} "
                            f"macro={result['macro_f1']:.3f} y_rec={result['per_class_recall'][YELLOW]:.3f} "
                            f"thr={result['yellow_threshold']:.4f}",
                            flush=True,
                        )
                        results.append(result)

    if not results:
        raise RuntimeError("No two-stage candidates ran.")

    selected = max(
        results,
        key=lambda row: (
            row["meets_yellow_recall_floor"],
            row["accuracy"],
            row["balanced_accuracy"],
            row["macro_f1"],
            row["per_class_recall"][YELLOW],
            -row["feature_count"],
        ),
    )
    feature_cols = selected["feature_cols"]
    X = prepare_matrix(df, feature_cols)
    stage1_estimator = stage1_all[selected["stage1_candidate"]]
    stage2_estimator = stage2_all[selected["stage2_candidate"]]
    final_model = fit_final(
        stage1_estimator,
        stage2_estimator,
        X,
        y,
        selected["stage1_strategy"],
        selected["stage2_strategy"],
        selected["yellow_threshold"],
        feature_cols,
        args.random_state,
        {
            "trained_at": utc_now_iso(),
            "features_csv": str(features_csv),
            "group_col": group_col,
            "rows": int(len(df)),
            "labels": dict(Counter(y)),
            "selected": {key: value for key, value in selected.items() if key not in {"feature_cols", "oof_pred", "oof_yellow_score", "oof_stage2_pred"}},
        },
    )
    selected["model_size_kb"] = model_size_kb(final_model)

    flat = pd.DataFrame([flatten_result(row) for row in results]).sort_values(
        ["meets_yellow_recall_floor", "accuracy", "balanced_accuracy", "macro_f1", "recall_amarillo"],
        ascending=[False, False, False, False, False],
    )
    flat.to_csv(out_dir / "two_stage_search_v2.csv", index=False)

    model_out = out_dir / "video_quality_v2_two_stage.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_cols": feature_cols,
            "classes": LABELS,
            "selected_feature_set": selected["feature_set"],
            "selected_stage1": selected["stage1_candidate"],
            "selected_stage2": selected["stage2_candidate"],
            "yellow_threshold": selected["yellow_threshold"],
            "metadata": final_model.metadata,
        },
        model_out,
    )

    best_oof = {
        "oof_pred": selected["oof_pred"],
        "oof_yellow_score": selected["oof_yellow_score"],
        "oof_stage2_pred": selected["oof_stage2_pred"],
    }
    preds = prediction_frame(final_model, df, X, oof=best_oof)
    preds.to_csv(out_dir / "predictions_two_stage_v2.csv", index=False)
    preds[preds["source_set"].astype(str) == "new_raw"].to_csv(out_dir / "predictions_new_two_stage_v2.csv", index=False)
    errors = preds[preds["label"].astype(str) != preds["oof_pred"].astype(str)].copy()
    errors.to_csv(out_dir / "errors_two_stage_v2.csv", index=False)

    summary = {
        "generated_at": utc_now_iso(),
        "features_csv": str(features_csv),
        "rows": int(len(df)),
        "labels": dict(Counter(y)),
        "group_col": group_col,
        "n_splits": int(n_splits),
        "min_yellow_recall": float(args.min_yellow_recall),
        "selected": {key: value for key, value in selected.items() if key not in {"feature_cols", "oof_pred", "oof_yellow_score", "oof_stage2_pred"}},
        "model_size_kb": selected["model_size_kb"],
        "results": [{key: value for key, value in row.items() if key not in {"feature_cols", "oof_pred", "oof_yellow_score", "oof_stage2_pred"}} for row in results],
    }
    write_json(out_dir / "metrics_two_stage_v2.json", summary)

    print("\n=== SELECTED TWO-STAGE MODEL ===", flush=True)
    print(
        f"{selected['feature_set']} / {selected['stage1_strategy']}+{selected['stage2_strategy']} / "
        f"{selected['stage1_candidate']} -> {selected['stage2_candidate']}",
        flush=True,
    )
    print(
        f"acc={selected['accuracy']:.3f} bal={selected['balanced_accuracy']:.3f} macro={selected['macro_f1']:.3f} "
        f"recall_yellow={selected['per_class_recall'][YELLOW]:.3f} precision_yellow={selected['per_class_precision'][YELLOW]:.3f} "
        f"threshold={selected['yellow_threshold']:.4f} size={selected['model_size_kb']:.1f}KB",
        flush=True,
    )
    print(f"[OK] model -> {model_out}", flush=True)


if __name__ == "__main__":
    main()
