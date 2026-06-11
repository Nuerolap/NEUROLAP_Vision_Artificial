from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .common import ensure_dir, utc_now_iso, videos_base, write_json
from .edm_annotation_queue_v3 import clean_text, is_complete, stable_jitter
from .edm_common_v3 import artifacts_v3_dir
from .edm_train_color_v3 import apply_decision_policy, merge_v2_features
from .edm_train_v3 import prepare_matrix


COLOR_ORDER = ("rojo", "amarillo", "verde")
PILOT_COLUMNS = (
    "pilot_selected",
    "pilot_rank",
    "pilot_within_label_rank",
    "pilot_priority_score",
    "pilot_reason",
    "pilot_generated_at",
)


def default_model_path(out_dir: Path) -> Path:
    candidates = (
        out_dir / "color_hybrid_manual_timing_fg05" / "edm_color_v3_conservative.joblib",
        out_dir / "color_hybrid_manual_fg05" / "edm_color_v3_conservative.joblib",
        out_dir / "color_hybrid_clean_fg05" / "edm_color_v3_conservative_clean.joblib",
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    if "sample_id" not in df.columns:
        raise RuntimeError(f"{path} must contain sample_id")
    for col in PILOT_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def model_probabilities(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    model = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    for col in feature_cols:
        if col not in features.columns:
            features[col] = np.nan
    X = prepare_matrix(features, feature_cols)
    if hasattr(model, "predict_proba"):
        raw_proba = np.asarray(model.predict_proba(X), dtype=float)
        classes = [str(cls) for cls in getattr(model, "classes_", COLOR_ORDER)]
        proba = pd.DataFrame(0.0, index=features.index, columns=[f"p_{label}" for label in COLOR_ORDER])
        for idx, cls in enumerate(classes):
            col = f"p_{cls}"
            if col in proba.columns:
                proba[col] = raw_proba[:, idx]
    else:
        raw_pred = np.asarray(model.predict(X), dtype=object)
        proba = pd.DataFrame(0.0, index=features.index, columns=[f"p_{label}" for label in COLOR_ORDER])
        for row_idx, pred in zip(features.index, raw_pred):
            col = f"p_{str(pred)}"
            if col in proba.columns:
                proba.loc[row_idx, col] = 1.0

    p_values = proba[[f"p_{label}" for label in COLOR_ORDER]].to_numpy(dtype=float)
    confidence = p_values.max(axis=1)
    raw_pred = np.asarray(COLOR_ORDER, dtype=object)[p_values.argmax(axis=1)]
    eps = 1e-9
    entropy = -np.sum(p_values * np.log(np.clip(p_values, eps, 1.0)), axis=1) / math.log(len(COLOR_ORDER))

    quality = (
        features["calidad_video"].astype(str).to_numpy()
        if "calidad_video" in features.columns
        else np.asarray(["unknown"] * len(features), dtype=object)
    )
    policy = bundle.get("decision_policy", {})
    green_threshold = float(policy.get("green_threshold", 0.5))
    decision = apply_decision_policy(raw_pred, confidence, quality, green_threshold)

    out = features[["sample_id"]].copy()
    for col in proba.columns:
        out[col] = proba[col].astype(float)
    out["model_raw_pred"] = raw_pred
    out["model_decision"] = decision
    out["model_confidence"] = confidence
    out["model_entropy"] = entropy
    out["model_green_threshold"] = green_threshold
    return out


def score_yellow_boost(row: pd.Series) -> tuple[float, str]:
    label = clean_text(row.get("label_actual", "")).casefold()
    quality = clean_text(row.get("calidad_video", "")).casefold()
    source_set = clean_text(row.get("source_set", "")).casefold()
    raw_pred = clean_text(row.get("model_raw_pred", "")).casefold()
    decision = clean_text(row.get("model_decision", "")).casefold()

    p_amarillo = float(row.get("p_amarillo", 0.0) or 0.0)
    p_rojo = float(row.get("p_rojo", 0.0) or 0.0)
    p_verde = float(row.get("p_verde", 0.0) or 0.0)
    confidence = float(row.get("model_confidence", 0.0) or 0.0)
    entropy = float(row.get("model_entropy", 0.0) or 0.0)

    reasons: list[str] = []
    score = 0.0
    if label == "amarillo":
        score += 1000.0
        reasons.append("carpeta_amarillo_pendiente")
    else:
        score += p_amarillo * 170.0
        score += entropy * 75.0
        score += (1.0 - confidence) * 55.0
        if raw_pred == "amarillo":
            score += 75.0
            reasons.append("modelo_predice_amarillo")
        if decision == "amarillo":
            score += 45.0
            reasons.append("politica_conservadora_amarillo")
        if abs(p_rojo - p_verde) < 0.16 and p_amarillo > 0.12:
            score += 18.0
            reasons.append("frontera_rojo_verde_con_amarillo")
        if p_amarillo >= 0.24:
            reasons.append(f"p_amarillo={p_amarillo:.2f}")
        if entropy >= 0.80:
            reasons.append(f"alta_incertidumbre={entropy:.2f}")

    if quality == "ok":
        score += 9.0
        reasons.append("calidad_ok")
    elif quality == "review":
        score += 5.0
        reasons.append("calidad_review")
    elif quality == "high_risk":
        score -= 4.0
        reasons.append("calidad_high_risk")

    if source_set == "new_raw":
        score += 3.0
        reasons.append("fuente_nueva")

    score += stable_jitter(clean_text(row.get("sample_id", ""))) * 0.01
    if not reasons:
        reasons.append("posible_caso_frontera")
    return round(score, 4), "; ".join(reasons)


def add_within_label_rank(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("label_actual", sort=False):
        group = group.copy()
        group["pilot_within_label_rank"] = range(1, len(group) + 1)
        parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else df


def select_fill_balanced(non_yellow: pd.DataFrame, slots: int) -> pd.DataFrame:
    if slots <= 0 or non_yellow.empty:
        return non_yellow.head(0).copy()
    selected_indices: list[int] = []
    labels = [label for label in ("rojo", "verde") if (non_yellow["label_actual"] == label).any()]
    if not labels:
        return non_yellow.sort_values("yellow_boost_priority_score", ascending=False).head(slots).copy()

    base = slots // len(labels)
    remainder = slots % len(labels)
    for idx, label in enumerate(labels):
        quota = base + (1 if idx < remainder else 0)
        group = non_yellow[non_yellow["label_actual"] == label].sort_values(
            "yellow_boost_priority_score", ascending=False
        )
        selected_indices.extend(group.head(quota).index.tolist())

    if len(selected_indices) < slots:
        remaining = non_yellow.drop(index=selected_indices, errors="ignore").sort_values(
            "yellow_boost_priority_score", ascending=False
        )
        selected_indices.extend(remaining.head(slots - len(selected_indices)).index.tolist())

    return non_yellow.loc[selected_indices].copy()


def build_yellow_boost_queue(
    annotations: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    target_total: int,
    include_complete: bool,
) -> pd.DataFrame:
    data = annotations.merge(scores, on="sample_id", how="left")
    data["label_actual"] = data["label_actual"].astype(str).str.strip().str.casefold()
    data["annotation_complete_for_queue"] = data.apply(is_complete, axis=1)
    if not include_complete:
        data = data[~data["annotation_complete_for_queue"].astype(bool)].copy()

    for col in ("p_rojo", "p_amarillo", "p_verde", "model_confidence", "model_entropy"):
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    for col in ("model_raw_pred", "model_decision"):
        if col not in data.columns:
            data[col] = ""
        data[col] = data[col].fillna("").astype(str)

    priority = data.apply(score_yellow_boost, axis=1)
    data["yellow_boost_priority_score"] = [item[0] for item in priority]
    data["yellow_boost_reason"] = [item[1] for item in priority]

    data = data.sort_values(
        ["yellow_boost_priority_score", "model_entropy", "p_amarillo", "sample_id"],
        ascending=[False, False, False, True],
    )
    yellow = data[data["label_actual"] == "amarillo"].copy()
    yellow = yellow.head(target_total).copy()
    slots = max(0, target_total - len(yellow))
    non_yellow = data[data["label_actual"] != "amarillo"].copy()
    fill = select_fill_balanced(non_yellow, slots)

    selected = pd.concat([yellow, fill], ignore_index=True)
    selected = selected.sort_values(
        ["label_actual", "yellow_boost_priority_score", "model_entropy", "sample_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    selected = add_within_label_rank(selected)
    selected = selected.sort_values(
        ["label_actual", "pilot_within_label_rank"],
        ascending=[True, True],
    ).reset_index(drop=True)

    generated_at = utc_now_iso()
    selected["pilot_selected"] = "1"
    selected["pilot_rank"] = range(1, len(selected) + 1)
    selected["pilot_priority_score"] = selected["yellow_boost_priority_score"]
    selected["pilot_reason"] = selected["yellow_boost_reason"]
    selected["pilot_generated_at"] = generated_at
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a focused EDM v3 queue to boost borderline/yellow annotations.")
    parser.add_argument("--base", default=str(videos_base()))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--features", default=None)
    parser.add_argument("--v2-features", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--scores-out", default=None)
    parser.add_argument("--target-total", type=int, default=60)
    parser.add_argument("--include-complete", action="store_true")
    args = parser.parse_args()

    base = Path(args.base).resolve()
    out_dir = ensure_dir(artifacts_v3_dir(base))
    annotations_csv = Path(args.annotations).resolve() if args.annotations else out_dir / "edm_annotations_working.csv"
    features_csv = Path(args.features).resolve() if args.features else out_dir / "edm_features.csv"
    v2_features_csv = Path(args.v2_features).resolve() if args.v2_features else base / "artifacts" / "v2" / "features_enriched.csv"
    model_path = Path(args.model).resolve() if args.model else default_model_path(out_dir)
    out_csv = Path(args.out).resolve() if args.out else out_dir / "edm_yellow_boost_queue.csv"
    scores_out = Path(args.scores_out).resolve() if args.scores_out else out_dir / "edm_yellow_boost_scores.csv"

    if args.target_total <= 0:
        raise RuntimeError("--target-total must be positive")
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    if not features_csv.exists():
        raise FileNotFoundError(str(features_csv))

    annotations = load_annotations(annotations_csv)
    features = pd.read_csv(features_csv)
    if "edm_feature_status" in features.columns:
        features = features[features["edm_feature_status"].astype(str).str.casefold() == "ok"].copy()
    features = merge_v2_features(features, v2_features_csv if v2_features_csv.exists() else None)
    bundle = joblib.load(model_path)
    scores = model_probabilities(bundle, features)
    scores.to_csv(scores_out, index=False, encoding="utf-8-sig")

    selected = build_yellow_boost_queue(
        annotations,
        scores,
        target_total=args.target_total,
        include_complete=args.include_complete,
    )
    ensure_dir(out_csv.parent)
    selected.to_csv(out_csv, index=False, encoding="utf-8-sig")

    all_data = annotations.copy()
    all_data["label_actual"] = all_data["label_actual"].astype(str).str.strip().str.casefold()
    all_data["annotation_complete_for_queue"] = all_data.apply(is_complete, axis=1)
    remaining = all_data if args.include_complete else all_data[~all_data["annotation_complete_for_queue"].astype(bool)]
    summary = {
        "generated_at": utc_now_iso(),
        "annotations_csv": str(annotations_csv),
        "features_csv": str(features_csv),
        "v2_features_csv": str(v2_features_csv) if v2_features_csv.exists() else None,
        "model": str(model_path),
        "out_csv": str(out_csv),
        "scores_out": str(scores_out),
        "target_total": int(args.target_total),
        "selected_rows": int(len(selected)),
        "include_complete": bool(args.include_complete),
        "completed_rows": int(all_data["annotation_complete_for_queue"].astype(bool).sum()),
        "remaining_rows": int(len(remaining)),
        "remaining_by_label": dict(Counter(remaining["label_actual"].astype(str))),
        "selected_by_label": dict(Counter(selected["label_actual"].astype(str))),
        "selected_by_quality": dict(Counter(selected.get("calidad_video", pd.Series(dtype=str)).astype(str))),
        "selected_by_source_set": dict(Counter(selected.get("source_set", pd.Series(dtype=str)).astype(str))),
        "selected_model_raw_pred": dict(Counter(selected.get("model_raw_pred", pd.Series(dtype=str)).astype(str))),
        "selected_model_decision": dict(Counter(selected.get("model_decision", pd.Series(dtype=str)).astype(str))),
        "mean_selected_p_amarillo": float(pd.to_numeric(selected.get("p_amarillo", 0), errors="coerce").fillna(0).mean()) if len(selected) else 0.0,
        "mean_selected_entropy": float(pd.to_numeric(selected.get("model_entropy", 0), errors="coerce").fillna(0).mean()) if len(selected) else 0.0,
    }
    write_json(out_dir / "edm_yellow_boost_queue_summary.json", summary)

    print("=== EDM V3 YELLOW BOOST QUEUE ===", flush=True)
    print(f"selected: {summary['selected_rows']} -> {out_csv}", flush=True)
    print(f"remaining: {summary['remaining_rows']} labels={summary['remaining_by_label']}", flush=True)
    print(f"selected labels: {summary['selected_by_label']}", flush=True)
    print(f"selected quality: {summary['selected_by_quality']}", flush=True)
    print(f"model raw: {summary['selected_model_raw_pred']}", flush=True)
    print(f"model decision: {summary['selected_model_decision']}", flush=True)


if __name__ == "__main__":
    main()
