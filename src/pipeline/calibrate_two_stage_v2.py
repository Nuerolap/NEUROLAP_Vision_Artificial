import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from .benchmark_models_v2 import prepare_matrix
from .common import artifacts_v2_dir, ensure_dir, utc_now_iso, videos_base, write_json
from .train_two_stage_v2 import LABELS, YELLOW, TwoStageClassifier


def threshold_grid(scores: np.ndarray, steps: int = 801) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([0.0], dtype=float)
    return np.unique(
        np.r_[
            np.linspace(float(scores.min()) - 1e-9, float(scores.max()) + 1e-9, steps),
            np.quantile(scores, np.linspace(0.0, 1.0, steps)),
        ]
    )


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_recall": {label: float(report[label]["recall"]) for label in LABELS},
        "per_class_precision": {label: float(report[label]["precision"]) for label in LABELS},
        "per_class_f1": {label: float(report[label]["f1-score"]) for label in LABELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def flatten(metrics: dict) -> dict:
    row = {key: value for key, value in metrics.items() if key not in {"per_class_recall", "per_class_precision", "per_class_f1", "confusion_matrix"}}
    for label, value in metrics["per_class_recall"].items():
        row[f"recall_{label}"] = value
    for label, value in metrics["per_class_precision"].items():
        row[f"precision_{label}"] = value
    for label, value in metrics["per_class_f1"].items():
        row[f"f1_{label}"] = value
    return row


def select_threshold(table: pd.DataFrame, min_yellow_recall: float) -> pd.Series:
    eligible = table[table["recall_amarillo"] >= min_yellow_recall].copy()
    if eligible.empty:
        eligible = table.copy()
        sort_cols = ["recall_amarillo", "accuracy", "balanced_accuracy", "macro_f1"]
    else:
        sort_cols = ["accuracy", "balanced_accuracy", "macro_f1", "precision_amarillo"]
    return eligible.sort_values(sort_cols, ascending=False).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the yellow threshold for a trained two-stage v2 model.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--features", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--min-yellow-recall", type=float, default=0.50)
    parser.add_argument("--suffix", default="practical")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v2_dir(base))
    features_csv = Path(args.features) if args.features else out_dir / "features_enriched.csv"
    model_path = Path(args.model) if args.model else out_dir / "video_quality_v2_two_stage.joblib"
    predictions_path = Path(args.predictions) if args.predictions else out_dir / "predictions_two_stage_v2.csv"

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    preds = pd.read_csv(predictions_path)
    required = {"label", "oof_yellow_score", "oof_stage2_pred"}
    missing = sorted(required - set(preds.columns))
    if missing:
        raise RuntimeError(f"Missing OOF columns in predictions file: {missing}")

    y = preds["label"].astype(str).to_numpy()
    scores = preds["oof_yellow_score"].astype(float).to_numpy()
    stage2_pred = preds["oof_stage2_pred"].astype(str).to_numpy()
    rows = []
    for threshold in threshold_grid(scores):
        pred = np.where(scores >= threshold, YELLOW, stage2_pred)
        metrics = score(y, pred)
        row = flatten(metrics)
        row["yellow_threshold"] = float(threshold)
        row["yellow_pred_rate"] = float(np.mean(pred == YELLOW))
        rows.append(row)
    table = pd.DataFrame(rows)
    selected = select_threshold(table, args.min_yellow_recall)

    model.yellow_threshold = float(selected["yellow_threshold"])
    model.metadata = dict(getattr(model, "metadata", {}) or {})
    model.metadata["calibrated_at"] = utc_now_iso()
    model.metadata["calibration_min_yellow_recall"] = float(args.min_yellow_recall)
    model.metadata["calibration_metrics"] = selected.to_dict()
    bundle["yellow_threshold"] = float(selected["yellow_threshold"])
    bundle["metadata"] = model.metadata

    suffix = args.suffix.strip().replace(" ", "_")
    model_out = out_dir / f"video_quality_v2_two_stage_{suffix}.joblib"
    joblib.dump(bundle, model_out)

    df = pd.read_csv(features_csv)
    if "extraction_status" in df.columns:
        df = df[df["extraction_status"].astype(str).str.lower() == "ok"].copy()
    X = prepare_matrix(df, feature_cols)
    details = model.predict_details(X)
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
    out = pd.concat([df[[col for col in out_cols if col in df.columns]].reset_index(drop=True), details.reset_index(drop=True)], axis=1)
    pred_out = out_dir / f"predictions_two_stage_{suffix}_v2.csv"
    out.to_csv(pred_out, index=False)
    out[out["source_set"].astype(str) == "new_raw"].to_csv(out_dir / f"predictions_new_two_stage_{suffix}_v2.csv", index=False)
    out[out["label"].astype(str) != out["pred"].astype(str)].to_csv(out_dir / f"errors_two_stage_{suffix}_v2.csv", index=False)

    table_out = out_dir / f"two_stage_thresholds_{suffix}_v2.csv"
    table.sort_values(["accuracy", "balanced_accuracy", "macro_f1"], ascending=False).to_csv(table_out, index=False)
    metrics_out = out_dir / f"metrics_two_stage_{suffix}_v2.json"
    write_json(
        metrics_out,
        {
            "generated_at": utc_now_iso(),
            "source_model": str(model_path),
            "output_model": str(model_out),
            "features_csv": str(features_csv),
            "predictions_csv": str(pred_out),
            "threshold_table": str(table_out),
            "min_yellow_recall": float(args.min_yellow_recall),
            "selected": selected.to_dict(),
        },
    )
    print("=== TWO-STAGE THRESHOLD CALIBRATION ===", flush=True)
    print(f"min_yellow_recall: {args.min_yellow_recall:.3f}", flush=True)
    print(
        f"threshold={selected['yellow_threshold']:.6f} acc={selected['accuracy']:.3f} "
        f"bal={selected['balanced_accuracy']:.3f} macro={selected['macro_f1']:.3f} "
        f"rec_y={selected['recall_amarillo']:.3f} prec_y={selected['precision_amarillo']:.3f}",
        flush=True,
    )
    print(f"[OK] model -> {model_out}", flush=True)
    print(f"[OK] predictions -> {pred_out}", flush=True)


if __name__ == "__main__":
    main()
