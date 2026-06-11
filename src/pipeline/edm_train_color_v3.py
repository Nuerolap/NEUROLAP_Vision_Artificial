from __future__ import annotations

import argparse
import io
import pickle
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import ensure_dir, utc_now_iso, videos_base, write_json
from .edm_common_v3 import COLOR_ORDER, artifacts_v3_dir, color_from_score, score_from_color_proxy
from .edm_train_v3 import prepare_matrix
from .benchmark_models_v2 import numeric_feature_columns as v2_numeric_feature_columns


TEXT_COLUMNS = {
    "sample_id",
    "source_path",
    "label_actual",
    "weak_color_from_folder",
    "source_set",
    "original_name",
    "normalized_stem",
    "group_id",
    "content_group_id",
    "duplicate_status",
    "content_duplicate_status",
    "calidad_video",
    "framing_level",
    "keypoints_path",
    "edm_feature_status",
    "edm_feature_error",
}
TARGET_COLUMNS = {
    "weak_score_from_folder",
    "target_score_0_6",
    "score_0_6_final",
    "computed_score_0_6",
    "score_0_6",
}
DECISION_LABELS = ["rojo", "amarillo", "verde", "requiere_revision"]
LABEL_TO_SCORE = {"rojo": 1, "amarillo": 3, "verde": 5, "requiere_revision": np.nan}


def numeric_feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    cols = []
    for col in df.columns:
        if col in TEXT_COLUMNS or col in TARGET_COLUMNS:
            continue
        if not (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
            continue
        cols.append(col)
    if feature_set == "all":
        return cols
    if feature_set == "global":
        return [col for col in cols if "_rep" not in col]
    if feature_set == "reps":
        rep_tokens = ("left_rep", "right_rep", "edm_", "global")
        return [col for col in cols if any(token in col for token in rep_tokens)]
    if feature_set == "compact":
        banned = ("_p10", "_p25", "_p50", "_p75", "_p90", "_min", "_max")
        return [col for col in cols if not any(token in col for token in banned)]
    raise ValueError(f"Unknown feature_set: {feature_set}")


def model_size_kb(model) -> float:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return float(len(buf.getvalue()) / 1024.0)


def candidate_models(random_state: int) -> dict[str, object]:
    svc60 = make_pipeline(
        StandardScaler(),
        SelectPercentile(f_classif, percentile=60),
        SVC(C=6.0, gamma="scale", class_weight="balanced", probability=True, random_state=random_state),
    )
    svc80 = make_pipeline(
        StandardScaler(),
        SelectPercentile(f_classif, percentile=80),
        SVC(C=8.0, gamma="scale", class_weight="balanced", probability=True, random_state=random_state),
    )
    hist = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.04,
        max_leaf_nodes=11,
        l2_regularization=0.12,
        class_weight="balanced",
        random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=260,
        max_depth=9,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return {
        "svc_select60_c6": svc60,
        "svc_select80_c8": svc80,
        "histgb_120_leaf11": hist,
        "extra_trees_260_depth9": ExtraTreesClassifier(
            n_estimators=260,
            max_depth=9,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "rf_260_depth9": rf,
        "vote_svc_hist_rf": VotingClassifier(
            estimators=[
                ("svc", svc60),
                ("hist", hist),
                ("rf", rf),
            ],
            voting="soft",
            weights=[2, 1, 1],
        ),
    }


def merge_v2_features(features: pd.DataFrame, v2_features_csv: Path | None) -> pd.DataFrame:
    if not v2_features_csv:
        return features
    if not v2_features_csv.exists():
        raise FileNotFoundError(str(v2_features_csv))
    v2 = pd.read_csv(v2_features_csv)
    if "sample_id" not in v2.columns:
        raise RuntimeError(f"{v2_features_csv} must contain sample_id")
    v2_cols = v2_numeric_feature_columns(v2, drop_source_metadata=True)
    v2_work = v2[["sample_id", *v2_cols]].copy()
    rename = {col: f"v2_{col}" for col in v2_cols}
    v2_work = v2_work.rename(columns=rename)
    return features.merge(v2_work, on="sample_id", how="left")


def drop_cross_label_groups(data: pd.DataFrame) -> pd.DataFrame:
    group_col = "content_group_id" if "content_group_id" in data.columns else "normalized_stem"
    if group_col not in data.columns:
        return data
    label_counts = data.groupby(group_col)["target_color"].nunique(dropna=True)
    conflict_groups = set(label_counts[label_counts > 1].index.astype(str))
    if not conflict_groups:
        return data
    return data[~data[group_col].astype(str).isin(conflict_groups)].copy()


def load_training_table(
    features_csv: Path,
    annotations_csv: Path | None,
    allow_folder_proxy: bool,
    exclude_label_conflicts: bool,
    v2_features_csv: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    features = pd.read_csv(features_csv)
    features = features[features["edm_feature_status"].astype(str).str.lower() == "ok"].copy()
    features = merge_v2_features(features, v2_features_csv)

    if annotations_csv and annotations_csv.exists():
        ann = pd.read_csv(annotations_csv)
        if "annotation_complete" in ann.columns:
            ann = ann[ann["annotation_complete"].astype(bool)].copy()
        if not ann.empty:
            keep = ["sample_id", "score_0_6_final", "color_final_validated"]
            missing = [col for col in keep if col not in ann.columns]
            if missing:
                raise RuntimeError(f"Validated annotations missing columns: {missing}")
            data = features.merge(ann[keep], on="sample_id", how="inner")
            data["target_color"] = data["color_final_validated"].astype(str).str.casefold()
            data["target_score_0_6"] = pd.to_numeric(data["score_0_6_final"], errors="coerce")
            data = data.dropna(subset=["target_score_0_6"])
            return (drop_cross_label_groups(data) if exclude_label_conflicts else data), "manual_annotations"

    if not allow_folder_proxy:
        raise RuntimeError("No complete manual annotations found. Pass --allow-folder-proxy only for a non-clinical proxy baseline.")
    features["target_color"] = features["weak_color_from_folder"].astype(str).str.casefold()
    features["target_score_0_6"] = features["target_color"].map(score_from_color_proxy)
    features = features.dropna(subset=["target_score_0_6"])
    return (drop_cross_label_groups(features) if exclude_label_conflicts else features), "folder_proxy_not_clinical"


def predict_with_confidence(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(X).astype(object)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(model.classes_, dtype=object)
        conf = np.max(proba, axis=1)
        pred = classes[np.argmax(proba, axis=1)]
        return pred.astype(object), conf
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        spread = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
        conf = 1.0 / (1.0 + np.exp(-spread))
        return pred, conf
    return pred, np.ones(len(pred), dtype=float)


def apply_decision_policy(raw_pred: np.ndarray, confidence: np.ndarray, quality: np.ndarray, green_threshold: float) -> np.ndarray:
    decisions = []
    for pred, conf, q in zip(raw_pred, confidence, quality):
        pred = str(pred)
        q = str(q).strip().casefold()
        if pred == "verde":
            if q == "high_risk" and conf < max(0.45, green_threshold):
                decisions.append("requiere_revision")
            elif conf < green_threshold:
                decisions.append("amarillo")
            else:
                decisions.append("verde")
        else:
            decisions.append(pred)
    return np.asarray(decisions, dtype=object)


def score_predictions(y_true: np.ndarray, raw_pred: np.ndarray, confidence: np.ndarray, quality: np.ndarray, green_threshold: float) -> tuple[dict, pd.DataFrame]:
    decision = apply_decision_policy(raw_pred, confidence, quality, green_threshold)
    non_green = y_true != "verde"
    false_green = float(np.mean(decision[non_green] == "verde")) if np.any(non_green) else 0.0
    review_rate = float(np.mean(decision == "requiere_revision"))
    raw_report = classification_report(y_true, raw_pred, labels=list(COLOR_ORDER), output_dict=True, zero_division=0)
    decision_report = classification_report(y_true, decision, labels=DECISION_LABELS, output_dict=True, zero_division=0)
    metrics = {
        "raw_color_accuracy": float(accuracy_score(y_true, raw_pred)),
        "decision_color_accuracy": float(np.mean(decision == y_true)),
        "raw_macro_f1": float(f1_score(y_true, raw_pred, labels=list(COLOR_ORDER), average="macro", zero_division=0)),
        "decision_macro_f1": float(f1_score(y_true, decision, labels=DECISION_LABELS, average="macro", zero_division=0)),
        "false_green_rate": false_green,
        "review_rate": review_rate,
        "green_threshold": float(green_threshold),
        "raw_recall_rojo": float(raw_report["rojo"]["recall"]),
        "raw_recall_amarillo": float(raw_report["amarillo"]["recall"]),
        "raw_recall_verde": float(raw_report["verde"]["recall"]),
        "decision_recall_rojo": float(decision_report["rojo"]["recall"]),
        "decision_recall_amarillo": float(decision_report["amarillo"]["recall"]),
        "decision_recall_verde": float(decision_report["verde"]["recall"]),
        "raw_confusion_matrix": confusion_matrix(y_true, raw_pred, labels=list(COLOR_ORDER)).tolist(),
        "decision_confusion_matrix": confusion_matrix(y_true, decision, labels=DECISION_LABELS).tolist(),
        "raw_report": raw_report,
        "decision_report": decision_report,
    }
    pred_df = pd.DataFrame(
        {
            "target_color": y_true,
            "target_score_0_6": [score_from_color_proxy(v) for v in y_true],
            "pred_color_raw": raw_pred,
            "decision_color": decision,
            "confidence": confidence,
            "pred_score_0_6": [LABEL_TO_SCORE.get(str(v), np.nan) for v in decision],
            "calidad_video": quality,
            "is_false_green": non_green & (decision == "verde"),
        }
    )
    return metrics, pred_df


def tune_threshold(y_true: np.ndarray, raw_pred: np.ndarray, confidence: np.ndarray, quality: np.ndarray, max_false_green: float, max_review_rate: float) -> tuple[dict, pd.DataFrame]:
    rows = []
    preds = {}
    for threshold in np.linspace(0.25, 0.85, 61):
        metrics, pred_df = score_predictions(y_true, raw_pred, confidence, quality, float(threshold))
        rows.append(metrics)
        preds[float(threshold)] = pred_df
    table = pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows])
    eligible = table[(table["false_green_rate"] <= max_false_green) & (table["review_rate"] <= max_review_rate)]
    if eligible.empty:
        eligible = table.sort_values(["false_green_rate", "review_rate", "decision_macro_f1", "decision_color_accuracy"], ascending=[True, True, False, False])
    else:
        eligible = eligible.sort_values(["decision_macro_f1", "decision_color_accuracy", "false_green_rate", "review_rate"], ascending=[False, False, True, True])
    selected_threshold = float(eligible.iloc[0]["green_threshold"])
    selected_metrics = next(row for row in rows if abs(float(row["green_threshold"]) - selected_threshold) < 1e-12)
    return selected_metrics, preds[selected_threshold]


def by_quality_metrics(pred_df: pd.DataFrame) -> dict:
    out = {}
    for quality, group in pred_df.groupby("calidad_video"):
        non_green = group["target_color"].astype(str) != "verde"
        out[str(quality)] = {
            "rows": int(len(group)),
            "decision_color_accuracy": float(np.mean(group["decision_color"].astype(str) == group["target_color"].astype(str))),
            "decision_macro_f1": float(f1_score(group["target_color"], group["decision_color"], labels=DECISION_LABELS, average="macro", zero_division=0)),
            "false_green_rate": float(np.mean(group.loc[non_green, "decision_color"].astype(str) == "verde")) if np.any(non_green) else 0.0,
            "review_rate": float(np.mean(group["decision_color"].astype(str) == "requiere_revision")),
        }
    return out


def evaluate_candidate(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, quality: np.ndarray, n_splits: int, random_state: int, max_false_green: float, max_review_rate: float) -> tuple[dict, object, pd.DataFrame]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.empty(len(y), dtype=object)
    oof_conf = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in cv.split(X, y, groups):
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train_idx], y[train_idx])
        pred, conf = predict_with_confidence(model, X[val_idx])
        oof_pred[val_idx] = pred
        oof_conf[val_idx] = conf
    metrics, pred_df = tune_threshold(y, oof_pred, oof_conf, quality, max_false_green, max_review_rate)
    fitted = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(X, y)
    metrics["model_size_kb"] = model_size_kb(fitted)
    return metrics, fitted, pred_df


def flatten_metrics(result: dict) -> dict:
    keys = [
        "feature_set",
        "candidate",
        "raw_color_accuracy",
        "decision_color_accuracy",
        "raw_macro_f1",
        "decision_macro_f1",
        "false_green_rate",
        "review_rate",
        "green_threshold",
        "raw_recall_rojo",
        "raw_recall_amarillo",
        "raw_recall_verde",
        "decision_recall_rojo",
        "decision_recall_amarillo",
        "decision_recall_verde",
        "model_size_kb",
    ]
    return {key: result.get(key) for key in keys}


def select_result(results: list[dict], max_false_green: float, max_review_rate: float) -> dict:
    eligible = [
        row
        for row in results
        if row["false_green_rate"] <= max_false_green and row["review_rate"] <= max_review_rate
    ]
    if eligible:
        return sorted(
            eligible,
            key=lambda row: (
                -row["decision_color_accuracy"],
                -row["decision_macro_f1"],
                -row["raw_color_accuracy"],
                row["false_green_rate"],
                row["review_rate"],
                row["model_size_kb"],
            ),
        )[0]
    return sorted(
        results,
        key=lambda row: (
            row["false_green_rate"],
            row["review_rate"],
            -row["decision_macro_f1"],
            -row["decision_color_accuracy"],
            row["model_size_kb"],
        ),
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an EDM v3 conservative color scorer.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--v2-features", default=None, help="Optional v2 enriched features CSV to merge as prefixed helper features.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--allow-folder-proxy", action="store_true")
    parser.add_argument("--exclude-label-conflicts", action="store_true")
    parser.add_argument("--feature-sets", nargs="*", default=["compact", "reps", "all"])
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-false-green", type=float, default=0.03)
    parser.add_argument("--max-review-rate", type=float, default=0.18)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else artifacts_v3_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "edm_features.csv"
    annotations_csv = Path(args.annotations) if args.annotations else out_dir / "edm_annotations_validated.csv"
    v2_features_csv = Path(args.v2_features) if args.v2_features else None
    data, target_source = load_training_table(features_csv, annotations_csv, args.allow_folder_proxy, args.exclude_label_conflicts, v2_features_csv)
    if data.empty:
        raise RuntimeError("No rows available for v3 color training.")

    y = data["target_color"].astype(str).to_numpy()
    groups = data["content_group_id"].astype(str).to_numpy() if "content_group_id" in data.columns else data["sample_id"].astype(str).to_numpy()
    quality = data["calidad_video"].astype(str).to_numpy() if "calidad_video" in data.columns else np.array(["unknown"] * len(data), dtype=object)
    n_splits = min(args.n_splits, min(Counter(y).values()), len(set(groups)))
    if n_splits < 2:
        raise RuntimeError(f"Need at least 2 folds, got {n_splits}.")

    requested = set(args.candidates or [])
    models = {name: model for name, model in candidate_models(args.random_state).items() if not requested or name in requested}
    results = []
    fitted_by_key = {}
    preds_by_key = {}
    for feature_set in args.feature_sets:
        feature_cols = numeric_feature_columns(data, feature_set)
        X = prepare_matrix(data, feature_cols)
        for name, estimator in models.items():
            key = (feature_set, name)
            print(f"[EDM COLOR] {feature_set} / {name}", flush=True)
            metrics, fitted, pred_df = evaluate_candidate(
                estimator,
                X,
                y,
                groups,
                quality,
                n_splits,
                args.random_state,
                args.max_false_green,
                args.max_review_rate,
            )
            metrics.update({"feature_set": feature_set, "candidate": name, "feature_cols": feature_cols, "feature_count": len(feature_cols), "by_quality": by_quality_metrics(pred_df)})
            results.append(metrics)
            fitted_by_key[key] = fitted
            preds_by_key[key] = pred_df
            print(
                f"  raw_acc={metrics['raw_color_accuracy']:.3f} dec_acc={metrics['decision_color_accuracy']:.3f} "
                f"raw_macro={metrics['raw_macro_f1']:.3f} dec_macro={metrics['decision_macro_f1']:.3f} "
                f"false_green={metrics['false_green_rate']:.3f} review={metrics['review_rate']:.3f}",
                flush=True,
            )

    selected = select_result(results, args.max_false_green, args.max_review_rate)
    selected_key = (selected["feature_set"], selected["candidate"])
    selected_model = fitted_by_key[selected_key]
    selected_preds = preds_by_key[selected_key].copy()
    meta_cols = [
        col
        for col in (
            "sample_id",
            "source_path",
            "label_actual",
            "source_set",
            "original_name",
            "content_group_id",
            "duplicate_status",
            "content_duplicate_status",
            "calidad_video",
        )
        if col in data.columns
    ]
    predictions = pd.concat([data[meta_cols].reset_index(drop=True), selected_preds.reset_index(drop=True)], axis=1)

    suffix = "_clean" if args.exclude_label_conflicts else ""
    model_out = out_dir / f"edm_color_v3_conservative{suffix}.joblib"
    predictions_out = out_dir / f"edm_color_predictions_v3{suffix}.csv"
    metrics_out = out_dir / f"edm_color_metrics_v3{suffix}.json"
    models_out = out_dir / f"edm_color_models_v3{suffix}.csv"
    joblib.dump(
        {
            "model": selected_model,
            "feature_cols": selected["feature_cols"],
            "target_source": target_source,
            "selected_feature_set": selected["feature_set"],
            "selected_candidate": selected["candidate"],
            "classes": list(COLOR_ORDER),
            "decision_labels": DECISION_LABELS,
            "decision_policy": {
                "green_threshold": float(selected["green_threshold"]),
                "high_risk_low_green_confidence": "requiere_revision",
                "low_green_confidence": "amarillo",
                "max_false_green": float(args.max_false_green),
                "max_review_rate": float(args.max_review_rate),
            },
            "metadata": {
                "trained_at": utc_now_iso(),
                "features_csv": str(features_csv),
                "v2_features_csv": str(v2_features_csv) if v2_features_csv else None,
                "annotations_csv": str(annotations_csv),
                "rows": int(len(data)),
                "target_source": target_source,
                "exclude_label_conflicts": bool(args.exclude_label_conflicts),
                "group_col": "content_group_id",
                "n_splits": int(n_splits),
                "selected": {key: value for key, value in selected.items() if key not in {"feature_cols", "raw_report", "decision_report", "raw_confusion_matrix", "decision_confusion_matrix", "by_quality"}},
            },
        },
        model_out,
    )
    predictions.to_csv(predictions_out, index=False)
    pd.DataFrame([flatten_metrics(row) for row in results]).sort_values(
        ["decision_color_accuracy", "decision_macro_f1", "raw_color_accuracy", "false_green_rate", "review_rate", "model_size_kb"],
        ascending=[False, False, False, True, True, True],
    ).to_csv(models_out, index=False)
    write_json(
        metrics_out,
        {
            "generated_at": utc_now_iso(),
            "features_csv": str(features_csv),
            "v2_features_csv": str(v2_features_csv) if v2_features_csv else None,
            "annotations_csv": str(annotations_csv),
            "rows": int(len(data)),
            "labels": dict(Counter(y)),
            "quality": dict(Counter(quality)),
            "target_source": target_source,
            "exclude_label_conflicts": bool(args.exclude_label_conflicts),
            "n_splits": int(n_splits),
            "selected": selected,
            "results": results,
        },
    )
    print("\n=== EDM COLOR V3 SELECTED ===", flush=True)
    print(
        f"{selected['feature_set']} / {selected['candidate']} target={target_source} "
        f"raw_acc={selected['raw_color_accuracy']:.3f} dec_acc={selected['decision_color_accuracy']:.3f} "
        f"raw_macro={selected['raw_macro_f1']:.3f} dec_macro={selected['decision_macro_f1']:.3f} "
        f"false_green={selected['false_green_rate']:.3f} review={selected['review_rate']:.3f}",
        flush=True,
    )
    print(f"[OK] model -> {model_out}", flush=True)
    print(f"[OK] predictions -> {predictions_out}", flush=True)


if __name__ == "__main__":
    main()
