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
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .common import ensure_dir, utc_now_iso, videos_base, write_json
from .edm_common_v3 import COLOR_ORDER, artifacts_v3_dir, color_from_score


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


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in TEXT_COLUMNS or col in TARGET_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return cols


def prepare_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    work = df.reindex(columns=feature_cols).copy()
    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    medians = work.median(numeric_only=True)
    return work.fillna(medians).fillna(0.0).astype(np.float32).to_numpy()


def model_size_kb(model) -> float:
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return float(len(buf.getvalue()) / 1024.0)


def candidate_models(random_state: int) -> dict[str, object]:
    return {
        "histgb_ord_80": HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            random_state=random_state,
        ),
        "histgb_ord_140": HistGradientBoostingRegressor(
            max_iter=140,
            learning_rate=0.035,
            max_leaf_nodes=11,
            l2_regularization=0.12,
            random_state=random_state,
        ),
        "extra_trees_ord_240": ExtraTreesRegressor(
            n_estimators=240,
            max_depth=9,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "rf_ord_240": RandomForestRegressor(
            n_estimators=240,
            max_depth=9,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "svr_rbf_c4": make_pipeline(StandardScaler(), SVR(C=4.0, gamma="scale", epsilon=0.15)),
    }


def score_to_color_array(scores: np.ndarray) -> np.ndarray:
    rounded = np.clip(np.rint(scores), 0, 6)
    return np.array([color_from_score(value) for value in rounded], dtype=object)


def confidence_from_score(score: float) -> float:
    score = float(np.clip(score, 0.0, 6.0))
    if score >= 4.5:
        return float(np.clip((score - 4.5) / 1.5, 0.0, 1.0))
    if score >= 2.5:
        return float(np.clip(min(score - 2.5, 4.5 - score) / 1.0, 0.0, 1.0))
    return float(np.clip((2.5 - score) / 2.5, 0.0, 1.0))


def decision_color(raw_color: str, confidence: float, quality: str, min_green_confidence: float) -> str:
    if raw_color != "verde":
        return raw_color
    quality = str(quality).strip().casefold()
    if quality == "high_risk" or confidence < (min_green_confidence * 0.7):
        return "requiere_revision"
    if quality == "review" or confidence < min_green_confidence:
        return "amarillo"
    return "verde"


def metrics_for(y_score: np.ndarray, pred_score: np.ndarray, quality: np.ndarray, min_green_confidence: float) -> tuple[dict, pd.DataFrame]:
    pred_score = np.clip(pred_score.astype(float), 0.0, 6.0)
    pred_round = np.clip(np.rint(pred_score), 0, 6).astype(int)
    y_round = np.clip(np.rint(y_score.astype(float)), 0, 6).astype(int)
    y_color = score_to_color_array(y_score)
    raw_color = score_to_color_array(pred_score)
    confidence = np.array([confidence_from_score(value) for value in pred_score], dtype=float)
    decisions = np.array(
        [decision_color(color, conf, q, min_green_confidence) for color, conf, q in zip(raw_color, confidence, quality)],
        dtype=object,
    )
    non_green = y_color != "verde"
    false_green = float(np.mean(decisions[non_green] == "verde")) if np.any(non_green) else 0.0
    decision_accuracy = float(np.mean(decisions == y_color))
    metrics = {
        "mae_score": float(mean_absolute_error(y_score, pred_score)),
        "rounded_score_accuracy": float(accuracy_score(y_round, pred_round)),
        "raw_color_accuracy": float(accuracy_score(y_color, raw_color)),
        "decision_color_accuracy": decision_accuracy,
        "raw_macro_f1": float(f1_score(y_color, raw_color, labels=list(COLOR_ORDER), average="macro", zero_division=0)),
        "decision_macro_f1": float(f1_score(y_color, decisions, labels=DECISION_LABELS, average="macro", zero_division=0)),
        "false_green_rate": false_green,
        "review_rate": float(np.mean(decisions == "requiere_revision")),
        "raw_confusion_matrix": confusion_matrix(y_color, raw_color, labels=list(COLOR_ORDER)).tolist(),
        "decision_confusion_matrix": confusion_matrix(y_color, decisions, labels=DECISION_LABELS).tolist(),
        "raw_report": classification_report(y_color, raw_color, labels=list(COLOR_ORDER), output_dict=True, zero_division=0),
        "decision_report": classification_report(y_color, decisions, labels=DECISION_LABELS, output_dict=True, zero_division=0),
    }
    pred_df = pd.DataFrame(
        {
            "target_score_0_6": y_score,
            "target_color": y_color,
            "pred_score_raw": pred_score,
            "pred_score_rounded": pred_round,
            "pred_color_raw": raw_color,
            "decision_color": decisions,
            "confidence": confidence,
            "calidad_video": quality,
            "is_false_green": (non_green & (decisions == "verde")),
        }
    )
    return metrics, pred_df


def by_quality_metrics(pred_df: pd.DataFrame) -> dict:
    out = {}
    for quality, group in pred_df.groupby("calidad_video"):
        non_green = group["target_color"].astype(str) != "verde"
        out[str(quality)] = {
            "rows": int(len(group)),
            "mae_score": float(mean_absolute_error(group["target_score_0_6"], group["pred_score_raw"])),
            "decision_color_accuracy": float(np.mean(group["decision_color"].astype(str) == group["target_color"].astype(str))),
            "false_green_rate": float(np.mean(group.loc[non_green, "decision_color"].astype(str) == "verde")) if np.any(non_green) else 0.0,
            "review_rate": float(np.mean(group["decision_color"].astype(str) == "requiere_revision")),
        }
    return out


def flatten_metrics(row: dict) -> dict:
    return {
        "candidate": row["candidate"],
        "mae_score": row["mae_score"],
        "rounded_score_accuracy": row["rounded_score_accuracy"],
        "raw_color_accuracy": row["raw_color_accuracy"],
        "decision_color_accuracy": row["decision_color_accuracy"],
        "raw_macro_f1": row["raw_macro_f1"],
        "decision_macro_f1": row["decision_macro_f1"],
        "false_green_rate": row["false_green_rate"],
        "review_rate": row["review_rate"],
        "model_size_kb": row["model_size_kb"],
    }


def load_training_table(features_csv: Path, annotations_csv: Path | None, allow_folder_proxy: bool) -> tuple[pd.DataFrame, str]:
    features = pd.read_csv(features_csv)
    features = features[features["edm_feature_status"].astype(str).str.lower() == "ok"].copy()
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
            data["target_score_0_6"] = pd.to_numeric(data["score_0_6_final"], errors="coerce")
            return data.dropna(subset=["target_score_0_6"]), "manual_annotations"
    if not allow_folder_proxy:
        raise RuntimeError("No complete manual annotations found. Run validation after filling edm_annotation_template.csv, or pass --allow-folder-proxy for a non-clinical smoke baseline.")
    features["target_score_0_6"] = pd.to_numeric(features["weak_score_from_folder"], errors="coerce")
    return features.dropna(subset=["target_score_0_6"]), "folder_proxy_not_clinical"


def evaluate_candidate(estimator, X: np.ndarray, y: np.ndarray, groups: np.ndarray, strata: np.ndarray, quality: np.ndarray, n_splits: int, random_state: int, min_green_confidence: float) -> tuple[dict, object, pd.DataFrame]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = np.zeros(len(y), dtype=float)
    for train_idx, val_idx in cv.split(X, strata, groups):
        model = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X[train_idx], y[train_idx])
        oof_pred[val_idx] = model.predict(X[val_idx])

    metrics, pred_df = metrics_for(y, oof_pred, quality, min_green_confidence)
    fitted = clone(estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted.fit(X, y)
    metrics["model_size_kb"] = model_size_kb(fitted)
    return metrics, fitted, pred_df


def select_result(results: list[dict]) -> dict:
    return sorted(
        results,
        key=lambda row: (
            row["false_green_rate"],
            row["mae_score"],
            -row["decision_macro_f1"],
            row["model_size_kb"],
        ),
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EDM-aware v3 score and conservative color baseline.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--allow-folder-proxy", action="store_true")
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-green-confidence", type=float, default=0.35)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else artifacts_v3_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "edm_features.csv"
    annotations_csv = Path(args.annotations) if args.annotations else out_dir / "edm_annotations_validated.csv"
    data, target_source = load_training_table(features_csv, annotations_csv, args.allow_folder_proxy)
    if data.empty:
        raise RuntimeError("No rows available for v3 training.")

    y = data["target_score_0_6"].astype(float).to_numpy()
    strata = np.array([color_from_score(value) for value in y], dtype=object)
    groups = data["content_group_id"].astype(str).to_numpy() if "content_group_id" in data.columns else data["sample_id"].astype(str).to_numpy()
    quality = data["calidad_video"].astype(str).to_numpy() if "calidad_video" in data.columns else np.array(["unknown"] * len(data), dtype=object)
    feature_cols = numeric_feature_columns(data)
    X = prepare_matrix(data, feature_cols)
    n_splits = min(args.n_splits, min(Counter(strata).values()), len(set(groups)))
    if n_splits < 2:
        raise RuntimeError(f"Need at least 2 folds, got {n_splits}.")

    requested = set(args.candidates or [])
    candidates = {name: model for name, model in candidate_models(args.random_state).items() if not requested or name in requested}
    results = []
    fitted_by_name = {}
    preds_by_name = {}
    for name, estimator in candidates.items():
        print(f"[EDM TRAIN] {name}", flush=True)
        metrics, fitted, pred_df = evaluate_candidate(
            estimator,
            X,
            y,
            groups,
            strata,
            quality,
            n_splits,
            args.random_state,
            args.min_green_confidence,
        )
        metrics["candidate"] = name
        metrics["by_quality"] = by_quality_metrics(pred_df)
        results.append(metrics)
        fitted_by_name[name] = fitted
        preds_by_name[name] = pred_df
        print(
            f"  mae={metrics['mae_score']:.3f} acc={metrics['decision_color_accuracy']:.3f} "
            f"macro={metrics['decision_macro_f1']:.3f} false_green={metrics['false_green_rate']:.3f} "
            f"size={metrics['model_size_kb']:.1f}KB",
            flush=True,
        )

    selected = select_result(results)
    model = fitted_by_name[selected["candidate"]]
    selected_preds = preds_by_name[selected["candidate"]].copy()
    meta_cols = [
        col
        for col in (
            "sample_id",
            "source_path",
            "label_actual",
            "source_set",
            "original_name",
            "content_group_id",
            "calidad_video",
        )
        if col in data.columns
    ]
    predictions = pd.concat([data[meta_cols].reset_index(drop=True), selected_preds.reset_index(drop=True)], axis=1)

    model_out = out_dir / "edm_score_v3_baseline.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target_source": target_source,
            "decision_labels": DECISION_LABELS,
            "color_order": list(COLOR_ORDER),
            "decision_policy": {
                "min_green_confidence": float(args.min_green_confidence),
                "green_high_risk_action": "requiere_revision",
                "green_review_or_low_confidence_action": "amarillo",
            },
            "metadata": {
                "trained_at": utc_now_iso(),
                "features_csv": str(features_csv),
                "annotations_csv": str(annotations_csv),
                "rows": int(len(data)),
                "target_source": target_source,
                "group_col": "content_group_id",
                "n_splits": int(n_splits),
                "selected_candidate": selected["candidate"],
                "selected_metrics": {key: value for key, value in selected.items() if key not in {"raw_report", "decision_report", "raw_confusion_matrix", "decision_confusion_matrix", "by_quality"}},
            },
        },
        model_out,
    )
    predictions.to_csv(out_dir / "edm_predictions_v3_baseline.csv", index=False)
    pd.DataFrame([flatten_metrics(row) for row in results]).sort_values(
        ["false_green_rate", "mae_score", "decision_macro_f1", "model_size_kb"],
        ascending=[True, True, False, True],
    ).to_csv(out_dir / "edm_train_v3_models.csv", index=False)
    write_json(
        out_dir / "edm_metrics_v3_baseline.json",
        {
            "generated_at": utc_now_iso(),
            "features_csv": str(features_csv),
            "annotations_csv": str(annotations_csv),
            "rows": int(len(data)),
            "labels": dict(Counter(strata)),
            "quality": dict(Counter(quality)),
            "target_source": target_source,
            "n_splits": int(n_splits),
            "selected_candidate": selected["candidate"],
            "selected": selected,
            "results": results,
        },
    )
    print("\n=== EDM V3 SELECTED ===", flush=True)
    print(
        f"{selected['candidate']} target={target_source} mae={selected['mae_score']:.3f} "
        f"acc={selected['decision_color_accuracy']:.3f} macro={selected['decision_macro_f1']:.3f} "
        f"false_green={selected['false_green_rate']:.3f}",
        flush=True,
    )
    print(f"[OK] model -> {model_out}", flush=True)


if __name__ == "__main__":
    main()
