from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common import artifacts_v2_dir, ensure_dir, resolve_source_path, videos_base


REP_COLUMNS = (
    "left_rep1",
    "left_rep2",
    "left_rep3",
    "right_rep1",
    "right_rep2",
    "right_rep3",
)
REP_STATUS_VALUES = ("", "correcta", "parcial", "incorrecta", "no_visible")
ERROR_COLUMNS = (
    "orden_incorrecto",
    "sin_puno",
    "sin_filo",
    "sin_palma",
    "mano_fuera",
    "no_responde",
)
EXTRA_REP_COLUMNS = (
    "left_extra_reps",
    "right_extra_reps",
    "extra_reps_note",
)
EXERCISE_TIMING_COLUMNS = (
    "exercise_start_s",
    "exercise_end_s",
    "exercise_timing_note",
)
LABEL_TO_SCORE_PROXY = {"rojo": 1, "amarillo": 3, "verde": 5}
COLOR_ORDER = ("rojo", "amarillo", "verde")
DECISION_COLORS = ("rojo", "amarillo", "verde", "requiere_revision")


def artifacts_v3_dir(base: Optional[Path] = None) -> Path:
    root = Path(base) if base else videos_base()
    return root / "artifacts" / "v3"


def score_from_repetitions(row: pd.Series) -> int:
    return int(sum(str(row.get(col, "")).strip().casefold() == "correcta" for col in REP_COLUMNS))


def partial_count(row: pd.Series) -> int:
    return int(sum(str(row.get(col, "")).strip().casefold() == "parcial" for col in REP_COLUMNS))


def color_from_score(score: float, *, no_response: bool = False) -> str:
    if no_response:
        return "rojo"
    try:
        value = float(score)
    except Exception:
        return ""
    if not np.isfinite(value):
        return ""
    if value >= 5:
        return "verde"
    if value >= 3:
        return "amarillo"
    return "rojo"


def score_from_color_proxy(label: str) -> int:
    return int(LABEL_TO_SCORE_PROXY.get(str(label).strip().casefold(), 0))


def normalize_review_status(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().casefold()
    aliases = {
        "correct": "correcta",
        "ok": "correcta",
        "si": "correcta",
        "sí": "correcta",
        "partial": "parcial",
        "parcialmente": "parcial",
        "incorrect": "incorrecta",
        "mal": "incorrecta",
        "not_visible": "no_visible",
        "novisible": "no_visible",
        "no visible": "no_visible",
        "": "",
    }
    return aliases.get(text, text)


def truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().casefold()
    return text in {"1", "true", "t", "yes", "y", "si", "sí", "x", "ok"}


def load_v2_table(path: Path, *, required: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    return df


def load_v3_source_table(
    base: Path,
    index_csv: Path | None = None,
    features_csv: Path | None = None,
    framing_csv: Path | None = None,
) -> pd.DataFrame:
    out_v2 = artifacts_v2_dir(base)
    index_path = index_csv or out_v2 / "video_index.csv"
    features_path = features_csv or out_v2 / "features_enriched.csv"
    framing_path = framing_csv or out_v2 / "hand_framing_report.csv"
    index = load_v2_table(index_path, required=("sample_id", "label", "source_path"))
    features = load_v2_table(features_path, required=("sample_id", "label", "source_path"))
    feature_cols = [
        col
        for col in (
            "sample_id",
            "content_group_id",
            "content_duplicate_status",
            "content_fingerprint",
            "source_duration_s",
            "source_frames",
            "source_fps",
            "frames_used",
            "keypoints_path",
            "extraction_status",
            "any_hand_coverage",
            "both_hands_coverage",
            "left_coverage",
            "right_coverage",
            "max_miss",
            "framing_level",
            "framing_risk_score",
            "max_edge_frame_any_ratio_05",
        )
        if col in features.columns
    ]
    merged = index.merge(features[feature_cols], on="sample_id", how="left", suffixes=("", "_feature"))
    if framing_path.exists():
        framing = load_v2_table(framing_path, required=("sample_id", "framing_level"))
        framing_cols = [
            col
            for col in (
                "sample_id",
                "miss_left_pct",
                "miss_right_pct",
                "max_miss",
                "any_hand_coverage",
                "both_hands_coverage",
                "max_edge_frame_any_ratio_05",
                "framing_risk_score",
                "framing_level",
                "framing_note",
            )
            if col in framing.columns
        ]
        merged = merged.merge(framing[framing_cols], on="sample_id", how="left", suffixes=("", "_framing"))
        for col in (
            "max_miss",
            "any_hand_coverage",
            "both_hands_coverage",
            "max_edge_frame_any_ratio_05",
            "framing_risk_score",
            "framing_level",
        ):
            framing_col = f"{col}_framing"
            if framing_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[framing_col].combine_first(merged[col])
                else:
                    merged[col] = merged[framing_col]
    merged["label_actual"] = merged["label"].astype(str)
    merged["source_exists"] = merged["source_path"].apply(lambda p: resolve_source_path(str(p), base).exists())
    return merged


def quality_from_row(row: pd.Series) -> str:
    level = str(row.get("framing_level", "")).strip().casefold()
    if level in {"ok", "review", "high_risk"}:
        return level
    any_cov = float(row.get("any_hand_coverage", 0.0) or 0.0)
    both_cov = float(row.get("both_hands_coverage", 0.0) or 0.0)
    max_miss = float(row.get("max_miss", 100.0) or 100.0)
    if any_cov < 0.75 or max_miss >= 85:
        return "high_risk"
    if any_cov < 0.90 or both_cov < 0.35 or max_miss >= 65:
        return "review"
    return "ok"


def ensure_v3_dir(base: Path) -> Path:
    return ensure_dir(artifacts_v3_dir(base))
