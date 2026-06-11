from __future__ import annotations

import argparse
import json
import mimetypes
import re
import shutil
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import pandas as pd

from .common import ensure_dir, resolve_source_path, utc_now_iso, videos_base
from .edm_common_v3 import (
    ERROR_COLUMNS,
    EXERCISE_TIMING_COLUMNS,
    EXTRA_REP_COLUMNS,
    REP_COLUMNS,
    REP_STATUS_VALUES,
    artifacts_v3_dir,
    color_from_score,
    normalize_review_status,
    score_from_repetitions,
    truthy,
)
from .edm_hand_order_v3 import HAND_ORDER_COLUMNS


APP_VERSION = "v3-annotation-local-1"
DEFAULT_PORT = 8765
TEXT_COLUMNS = ("confidence_human", "reviewer", "reviewer_notes")
APP_COLUMNS = ("annotation_updated_at", "annotation_app_version")
HAND_ORDER_OVERRIDE_COLUMN = "hand_order_override"
PILOT_COLUMNS = (
    "pilot_selected",
    "pilot_rank",
    "pilot_within_label_rank",
    "pilot_priority_score",
    "pilot_reason",
    "pilot_generated_at",
)
EDITABLE_COLUMNS = (*REP_COLUMNS, *ERROR_COLUMNS, *EXTRA_REP_COLUMNS, *EXERCISE_TIMING_COLUMNS, *TEXT_COLUMNS, HAND_ORDER_OVERRIDE_COLUMN)
REQUIRED_COLUMNS = ("sample_id", "source_path", "label_actual", "review_sheet", *REP_COLUMNS)


def clean_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def response_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def bool_cell(value: bool) -> str:
    return "1" if value else ""


def is_complete_row(row: pd.Series | dict[str, Any]) -> bool:
    return all(normalize_review_status(row.get(col, "")) in REP_STATUS_VALUES[1:] for col in REP_COLUMNS)


def is_started_row(row: pd.Series | dict[str, Any]) -> bool:
    if any(clean_cell(row.get(col, "")) for col in REP_COLUMNS):
        return True
    if any(truthy(row.get(col, "")) for col in ERROR_COLUMNS):
        return True
    if any(clean_cell(row.get(col, "")) for col in EXTRA_REP_COLUMNS):
        return True
    if any(clean_cell(row.get(col, "")) for col in EXERCISE_TIMING_COLUMNS):
        return True
    if clean_cell(row.get(HAND_ORDER_OVERRIDE_COLUMN, "")):
        return True
    if clean_cell(row.get("score_0_6", "")) or clean_cell(row.get("color_final", "")):
        return True
    return False


def computed_score_and_color(row: pd.Series | dict[str, Any]) -> tuple[int, str]:
    series = pd.Series({col: normalize_review_status(row.get(col, "")) for col in REP_COLUMNS})
    score = int(score_from_repetitions(series))
    return score, color_from_score(score, no_response=truthy(row.get("no_responde", "")))


def row_priority(row: dict[str, Any]) -> tuple[int, int, int, str]:
    complete_rank = 1 if row.get("annotation_complete") else 0
    pilot_selected = 0 if clean_cell(row.get("pilot_selected", "")) == "1" else 1
    try:
        pilot_rank = int(float(clean_cell(row.get("pilot_rank", ""))))
    except Exception:
        pilot_rank = 999999
    label = clean_cell(row.get("label_actual", "")).casefold()
    quality = clean_cell(row.get("calidad_video", "")).casefold()
    label_rank = {"amarillo": 0, "rojo": 1, "verde": 2}.get(label, 3)
    quality_rank = {"high_risk": 0, "review": 1, "ok": 2}.get(quality, 3)
    return complete_rank, pilot_selected, pilot_rank, label_rank, quality_rank, clean_cell(row.get("sample_id", ""))


class AnnotationStore:
    def __init__(
        self,
        base: Path,
        template_csv: Path,
        working_csv: Path,
        queue_csv: Path | None = None,
        hand_order_csv: Path | None = None,
    ) -> None:
        self.base = base.resolve()
        self.template_csv = template_csv.resolve()
        self.working_csv = working_csv.resolve()
        self.queue_csv = queue_csv.resolve() if queue_csv else None
        self.hand_order_csv = hand_order_csv.resolve() if hand_order_csv else None
        self.backup_csv = self.working_csv.with_suffix(self.working_csv.suffix + ".bak")
        self.lock = threading.Lock()
        self._ensure_working_csv()
        self.df = self._read_csv()

    def _ensure_working_csv(self) -> None:
        if not self.template_csv.exists():
            raise RuntimeError(f"Annotation template not found: {self.template_csv}")
        ensure_dir(self.working_csv.parent)
        if not self.working_csv.exists():
            shutil.copy2(self.template_csv, self.working_csv)

    def _read_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.working_csv, keep_default_na=False, dtype=str)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise RuntimeError(f"Annotation CSV missing required columns: {missing}")
        if df["sample_id"].duplicated().any():
            duplicated = sorted(df.loc[df["sample_id"].duplicated(), "sample_id"].astype(str).unique())
            raise RuntimeError(f"Duplicate sample_id values in annotations: {duplicated[:10]}")
        for col in ("score_0_6", "color_final", "annotation_status", *EDITABLE_COLUMNS, *APP_COLUMNS):
            if col not in df.columns:
                df[col] = ""
        df = self._merge_pilot_queue(df)
        df = self._merge_hand_order(df)
        return df

    def _merge_pilot_queue(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in PILOT_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])
        if not self.queue_csv or not self.queue_csv.exists():
            for col in PILOT_COLUMNS:
                df[col] = ""
            df["pilot_selected"] = "0"
            return df
        queue = pd.read_csv(self.queue_csv, keep_default_na=False, dtype=str)
        keep = [col for col in ("sample_id", *PILOT_COLUMNS) if col in queue.columns]
        if "sample_id" not in keep:
            for col in PILOT_COLUMNS:
                df[col] = ""
            df["pilot_selected"] = "0"
            return df
        queue = queue[keep].drop_duplicates(subset=["sample_id"]).copy()
        merged = df.merge(queue, on="sample_id", how="left")
        for col in PILOT_COLUMNS:
            if col not in merged.columns:
                merged[col] = ""
            merged[col] = merged[col].fillna("").astype(str)
        merged["pilot_selected"] = merged["pilot_selected"].replace("", "0")
        return merged

    def _merge_hand_order(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in HAND_ORDER_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])
        if not self.hand_order_csv or not self.hand_order_csv.exists():
            for col in HAND_ORDER_COLUMNS:
                df[col] = ""
            df["hand_order_mirror_suggested"] = "0"
            return df
        audit = pd.read_csv(self.hand_order_csv, keep_default_na=False, dtype=str)
        keep = [col for col in ("sample_id", *HAND_ORDER_COLUMNS) if col in audit.columns]
        if "sample_id" not in keep:
            for col in HAND_ORDER_COLUMNS:
                df[col] = ""
            df["hand_order_mirror_suggested"] = "0"
            return df
        audit = audit[keep].drop_duplicates(subset=["sample_id"]).copy()
        merged = df.merge(audit, on="sample_id", how="left")
        for col in HAND_ORDER_COLUMNS:
            if col not in merged.columns:
                merged[col] = ""
            merged[col] = merged[col].fillna("").astype(str)
        merged["hand_order_mirror_suggested"] = merged["hand_order_mirror_suggested"].replace("", "0")
        return merged

    def _write_csv(self) -> None:
        ensure_dir(self.working_csv.parent)
        if self.working_csv.exists():
            shutil.copy2(self.working_csv, self.backup_csv)
        tmp = self.working_csv.with_suffix(self.working_csv.suffix + ".tmp")
        self.df.to_csv(tmp, index=False, encoding="utf-8-sig")
        tmp.replace(self.working_csv)

    def _row_index(self, sample_id: str) -> int:
        matches = self.df.index[self.df["sample_id"].astype(str) == sample_id].tolist()
        if not matches:
            raise KeyError(f"Unknown sample_id: {sample_id}")
        return int(matches[0])

    def _record_from_row(self, idx: int) -> dict[str, Any]:
        row = self.df.loc[idx]
        record = {col: clean_cell(row.get(col, "")) for col in self.df.columns}
        record["row_index"] = idx
        record["annotation_complete"] = is_complete_row(record)
        record["annotation_started"] = is_started_row(record)
        score, color = computed_score_and_color(record)
        record["computed_score_0_6"] = score
        record["computed_color"] = color
        sample_id = quote(record["sample_id"], safe="")
        record["video_url"] = f"/media/video/{sample_id}"
        record["review_sheet_url"] = f"/media/sheet/{sample_id}"
        record["download_video_url"] = f"/media/video/{sample_id}?download=1"
        return record

    def _progress(self) -> dict[str, Any]:
        records = [self._record_from_row(int(idx)) for idx in self.df.index]
        total = len(records)
        complete = sum(1 for row in records if row["annotation_complete"])
        started = sum(1 for row in records if row["annotation_started"])
        by_label: dict[str, int] = {}
        by_quality: dict[str, int] = {}
        for row in records:
            by_label[row.get("label_actual", "")] = by_label.get(row.get("label_actual", ""), 0) + 1
            by_quality[row.get("calidad_video", "")] = by_quality.get(row.get("calidad_video", ""), 0) + 1
        return {
            "total": total,
            "started": started,
            "complete": complete,
            "remaining": total - complete,
            "by_label": by_label,
            "by_quality": by_quality,
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            rows = [self._record_from_row(int(idx)) for idx in self.df.index]
            rows.sort(key=row_priority)
            return {
                "app_version": APP_VERSION,
                "generated_at": utc_now_iso(),
                "working_csv": str(self.working_csv),
                "template_csv": str(self.template_csv),
                "queue_csv": str(self.queue_csv) if self.queue_csv else "",
                "hand_order_csv": str(self.hand_order_csv) if self.hand_order_csv else "",
                "pilot_available": bool(self.queue_csv and self.queue_csv.exists() and self.df["pilot_selected"].astype(str).eq("1").any()),
                "hand_order_available": bool(self.hand_order_csv and self.hand_order_csv.exists()),
                "backup_csv": str(self.backup_csv),
                "rows": rows,
                "progress": self._progress(),
                "rep_columns": list(REP_COLUMNS),
                "error_columns": list(ERROR_COLUMNS),
                "status_values": list(REP_STATUS_VALUES[1:]),
            }

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_id = clean_cell(payload.get("sample_id", ""))
        if not sample_id:
            raise ValueError("sample_id is required")
        with self.lock:
            idx = self._row_index(sample_id)
            for col in REP_COLUMNS:
                if col not in payload:
                    continue
                value = normalize_review_status(payload.get(col, ""))
                if value not in REP_STATUS_VALUES:
                    raise ValueError(f"{col} invalid value: {value}")
                self.df.at[idx, col] = value
            for col in ERROR_COLUMNS:
                if col in payload:
                    self.df.at[idx, col] = bool_cell(bool(payload.get(col)))
            for col in EXTRA_REP_COLUMNS:
                if col not in payload:
                    continue
                if col == "extra_reps_note":
                    self.df.at[idx, col] = clean_cell(payload.get(col, ""))[:500]
                    continue
                try:
                    value = int(float(clean_cell(payload.get(col, "")) or "0"))
                except ValueError:
                    value = 0
                self.df.at[idx, col] = str(max(0, min(value, 12)))
            for col in EXERCISE_TIMING_COLUMNS:
                if col not in payload:
                    continue
                if col == "exercise_timing_note":
                    self.df.at[idx, col] = clean_cell(payload.get(col, ""))[:500]
                    continue
                text = clean_cell(payload.get(col, ""))
                if not text:
                    self.df.at[idx, col] = ""
                    continue
                try:
                    value = max(0.0, float(text))
                except ValueError:
                    value = 0.0
                self.df.at[idx, col] = f"{value:.3f}".rstrip("0").rstrip(".")
            if HAND_ORDER_OVERRIDE_COLUMN in payload:
                override = clean_cell(payload.get(HAND_ORDER_OVERRIDE_COLUMN, "")).casefold()
                if override not in {"", "mirror_on", "mirror_off"}:
                    raise ValueError(f"{HAND_ORDER_OVERRIDE_COLUMN} invalid value: {override}")
                self.df.at[idx, HAND_ORDER_OVERRIDE_COLUMN] = override
            for col in TEXT_COLUMNS:
                if col in payload:
                    value = clean_cell(payload.get(col, ""))
                    if col == "reviewer_notes":
                        value = value[:5000]
                    else:
                        value = value[:120]
                    self.df.at[idx, col] = value

            row = self.df.loc[idx].to_dict()
            started = is_started_row(row)
            complete = is_complete_row(row)
            if started:
                score, color = computed_score_and_color(row)
                self.df.at[idx, "score_0_6"] = str(score)
                self.df.at[idx, "color_final"] = color
            else:
                self.df.at[idx, "score_0_6"] = ""
                self.df.at[idx, "color_final"] = ""
            self.df.at[idx, "annotation_status"] = "complete" if complete else ("in_progress" if started else "")
            self.df.at[idx, "annotation_updated_at"] = utc_now_iso() if started else ""
            self.df.at[idx, "annotation_app_version"] = APP_VERSION if started else ""
            self._write_csv()
            return {
                "row": self._record_from_row(idx),
                "progress": self._progress(),
                "working_csv": str(self.working_csv),
                "backup_csv": str(self.backup_csv),
            }

    def media_path(self, sample_id: str, kind: str) -> Path:
        with self.lock:
            idx = self._row_index(sample_id)
            row = self.df.loc[idx]
            column = "source_path" if kind == "video" else "review_sheet"
            value = clean_cell(row.get(column, ""))
            if not value:
                raise FileNotFoundError(f"{column} is blank for {sample_id}")
            path = resolve_source_path(value, self.base).resolve()
            try:
                path.relative_to(self.base)
            except ValueError as exc:
                raise PermissionError(f"Refusing to serve file outside base: {path}") from exc
            if not path.exists():
                raise FileNotFoundError(path)
            return path


APP_HTML = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NEUROLAP EDM v3 Annotation</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f7;
      --panel: #ffffff;
      --panel-2: #eef2f3;
      --ink: #202427;
      --muted: #66727a;
      --line: #d7dddf;
      --green: #207a45;
      --yellow: #a36b00;
      --red: #a83232;
      --blue: #2563a7;
      --focus: #1f6feb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 14px;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
    }
    button:hover { border-color: #aeb8bd; background: #f9fbfb; }
    button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
      outline: 2px solid var(--focus);
      outline-offset: 1px;
    }
    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(300px, 340px) minmax(0, 1fr);
      overflow-x: hidden;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: #fbfcfc;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      min-width: 0;
      overflow: hidden;
    }
    .side-head {
      padding: 12px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
      min-width: 0;
    }
    .brand {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      min-width: 0;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    .subtle { color: var(--muted); font-size: 12px; }
    .progress {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      min-width: 0;
    }
    .metric:last-child { grid-column: 1 / -1; }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 8px;
      min-width: 0;
    }
    .metric strong {
      display: block;
      font-size: 18px;
      line-height: 1.2;
    }
    .filters {
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 7px;
      min-width: 0;
    }
    .filters select, .filters input {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
    }
    .list {
      overflow: auto;
      padding: 8px;
      display: grid;
      gap: 6px;
      min-width: 0;
    }
    .item {
      width: 100%;
      min-width: 0;
      text-align: left;
      display: grid;
      gap: 4px;
      border-radius: 6px;
      background: #fff;
      padding: 9px;
    }
    .item.active { border-color: var(--blue); box-shadow: 0 0 0 1px var(--blue); }
    .item.complete { background: #f0f7f2; }
    .item-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      min-width: 0;
    }
    .sample {
      font-weight: 700;
      overflow-wrap: anywhere;
      line-height: 1.25;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 7px;
      font-size: 11px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      white-space: nowrap;
    }
    .pill.verde { color: var(--green); border-color: #9cccaa; background: #edf8ef; }
    .pill.amarillo { color: var(--yellow); border-color: #e1c36d; background: #fff7df; }
    .pill.rojo { color: var(--red); border-color: #e0aaa7; background: #fff0ef; }
    .main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }
    .topbar {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      min-width: 0;
    }
    #currentTitle { overflow-wrap: anywhere; }
    .top-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .workspace {
      min-height: 0;
      overflow: auto;
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
      gap: 16px;
      align-items: start;
    }
    .pane {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }
    .pane-head {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }
    .pane-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .pane-actions button {
      padding: 5px 8px;
      font-size: 12px;
    }
    .pane-body { padding: 12px; }
    video, .sheet-img {
      width: 100%;
      display: block;
      background: #111;
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    video { max-height: 58vh; }
    video.media-mirrored { transform: scaleX(-1); }
    .sheet-img { background: #202020; }
    .timing-tools {
      margin-top: 10px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      align-items: end;
    }
    .timing-tools button {
      min-height: 36px;
    }
    .time-field {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
    }
    .time-field input {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      color: var(--ink);
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .meta {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
    }
    .meta b { display: block; font-size: 11px; color: var(--muted); margin-bottom: 3px; }
    .meta span { overflow-wrap: anywhere; }
    .score-line {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-bottom: 12px;
    }
    .score {
      font-size: 24px;
      font-weight: 700;
      line-height: 1;
    }
    .rep-grid {
      display: grid;
      gap: 8px;
    }
    .rep-row {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      display: grid;
      grid-template-columns: 72px repeat(4, minmax(76px, 1fr));
      gap: 7px;
      align-items: center;
    }
    .rep-row.active {
      border-color: var(--blue);
      box-shadow: 0 0 0 1px var(--blue);
    }
    .rep-name { font-weight: 700; }
    .status-btn.active[data-status="correcta"] { background: #e8f6ec; border-color: #74b987; color: var(--green); }
    .status-btn.active[data-status="parcial"] { background: #fff5d7; border-color: #d5ad37; color: var(--yellow); }
    .status-btn.active[data-status="incorrecta"] { background: #fff0ef; border-color: #d79592; color: var(--red); }
    .status-btn.active[data-status="no_visible"] { background: #eef2f3; border-color: #9faab0; color: #465159; }
    .checks {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .check {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      display: flex;
      gap: 8px;
      align-items: center;
      background: #fff;
    }
    .form-grid {
      margin-top: 12px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    label.field { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
    label.field input, label.field select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 84px; resize: vertical; }
    .wide { grid-column: 1 / -1; }
    .path-text {
      overflow-wrap: anywhere;
      line-height: 1.25;
      max-height: 44px;
      overflow: hidden;
    }
    .status-text {
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
    }
    .danger-text { color: var(--red); }
    .empty {
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 1080px) {
      .shell { grid-template-columns: 300px minmax(0, 1fr); }
      .workspace { grid-template-columns: 1fr; }
      video { max-height: none; }
    }
    @media (max-width: 860px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar { min-height: auto; max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }
      .side-head { padding: 10px; }
      .list { max-height: 34vh; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .top-actions { justify-content: flex-start; }
      .timing-tools { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .rep-row { grid-template-columns: 1fr 1fr; }
      .rep-name { grid-column: 1 / -1; }
      .checks, .form-grid, .meta-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 520px) {
      .progress, .filters { grid-template-columns: 1fr; }
      .metric:last-child { grid-column: auto; }
      .timing-tools { grid-template-columns: 1fr; }
      .workspace { padding: 10px; }
      .top-actions button { flex: 1 1 120px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="side-head">
        <div class="brand">
          <h1>EDM v3</h1>
          <span class="subtle">anotacion local</span>
        </div>
        <div class="progress">
          <div class="metric"><strong id="metricComplete">0</strong><span>completos</span></div>
          <div class="metric"><strong id="metricStarted">0</strong><span>iniciados</span></div>
          <div class="metric"><strong id="metricTotal">0</strong><span>total</span></div>
          <div class="metric"><strong id="metricExtrasOpen">0</strong><span>extras sin fin</span></div>
        </div>
        <div class="filters">
          <select id="filterStatus" title="Filtro de estado">
            <option value="extras_without_end">Extras sin fin</option>
            <option value="pilot">Cola recomendada</option>
            <option value="unannotated">Sin completar</option>
            <option value="all">Todos</option>
            <option value="complete">Completos</option>
            <option value="started">Iniciados</option>
          </select>
          <select id="filterLabel" title="Filtro de etiqueta">
            <option value="">Color original</option>
            <option value="amarillo">Amarillo</option>
            <option value="rojo">Rojo</option>
            <option value="verde">Verde</option>
          </select>
          <select id="filterQuality" title="Filtro de calidad">
            <option value="">Calidad</option>
            <option value="high_risk">High risk</option>
            <option value="review">Review</option>
            <option value="ok">Ok</option>
          </select>
          <select id="filterSource" title="Filtro de fuente">
            <option value="">Fuente</option>
          </select>
          <input id="filterSearch" class="wide" type="search" placeholder="Buscar sample, nombre o nota">
        </div>
        <div class="subtle path-text" id="workingPath"></div>
      </div>
      <div class="list" id="sampleList"></div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div>
          <strong id="currentTitle">Cargando...</strong>
          <div class="status-text" id="saveStatus"></div>
        </div>
        <div class="top-actions">
          <button id="prevBtn" type="button">Anterior</button>
          <button id="nextBtn" type="button">Siguiente</button>
          <button id="nextOpenBtn" type="button">Siguiente sin anotar</button>
          <button id="saveBtn" type="button">Guardar</button>
        </div>
      </div>

      <div class="workspace" id="workspace">
        <section class="pane">
          <div class="pane-head">
            <strong>Video original</strong>
            <div class="pane-actions">
              <button id="mirrorBtn" type="button">Espejo OFF</button>
              <button id="mirrorAutoBtn" type="button">Auto</button>
              <a id="downloadLink" class="subtle" href="#" target="_blank" rel="noopener">abrir video</a>
            </div>
          </div>
          <div class="pane-body">
            <video id="video" controls preload="metadata"></video>
            <div class="timing-tools">
              <label class="time-field">Inicio ejercicio
                <input id="exerciseStart" type="number" min="0" step="0.001" placeholder="seg">
              </label>
              <label class="time-field">Fin ejercicio
                <input id="exerciseEnd" type="number" min="0" step="0.001" placeholder="opcional">
              </label>
              <button id="markStartBtn" type="button">Marcar inicio</button>
              <button id="markEndBtn" type="button">Marcar fin</button>
              <button id="clearTimingBtn" type="button">Limpiar timing</button>
              <label class="time-field wide">Nota timing
                <input id="exerciseTimingNote" type="text" maxlength="500" placeholder="ej. inicia despues de instrucciones">
              </label>
            </div>
            <div class="meta-grid">
              <div class="meta"><b>Etiqueta carpeta</b><span id="metaLabel"></span></div>
              <div class="meta"><b>Calidad captura</b><span id="metaQuality"></span></div>
              <div class="meta"><b>Fuente</b><span id="metaSource"></span></div>
              <div class="meta"><b>Nombre original</b><span id="metaOriginal"></span></div>
              <div class="meta"><b>Duracion</b><span id="metaDuration"></span></div>
              <div class="meta"><b>Cobertura manos</b><span id="metaCoverage"></span></div>
              <div class="meta"><b>Orden detectado</b><span id="metaHandOrder"></span></div>
              <div class="meta wide"><b>Piloto</b><span id="metaPilot"></span></div>
            </div>
          </div>
        </section>

        <section class="pane">
          <div class="pane-head">
            <strong>Revision y rubrica</strong>
            <span class="subtle">1=correcta, 2=parcial, 3=incorrecta, 4=no visible</span>
          </div>
          <div class="pane-body">
            <img id="reviewSheet" class="sheet-img" alt="review sheet">
            <div class="score-line">
              <span class="score" id="scoreText">0/6</span>
              <span class="pill" id="colorText">rojo</span>
              <button id="noResponseBtn" type="button">Marcar no responde</button>
            </div>
            <div class="rep-grid" id="repGrid"></div>
            <div class="checks" id="errorChecks"></div>
            <div class="form-grid">
              <label class="field">Reviewer
                <input id="reviewer" type="text" maxlength="120" autocomplete="off">
              </label>
              <label class="field">Confianza humana
                <select id="confidence">
                  <option value=""></option>
                  <option value="alta">alta</option>
                  <option value="media">media</option>
                  <option value="baja">baja</option>
                </select>
              </label>
              <label class="field">Extras izquierda
                <input id="leftExtraReps" type="number" min="0" max="12" step="1">
              </label>
              <label class="field">Extras derecha
                <input id="rightExtraReps" type="number" min="0" max="12" step="1">
              </label>
              <label class="field wide">Nota de repeticiones extra
                <input id="extraRepsNote" type="text" maxlength="500" autocomplete="off">
              </label>
              <label class="field wide">Notas
                <textarea id="notes" maxlength="5000"></textarea>
              </label>
            </div>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script>
    const repColumns = ["left_rep1", "left_rep2", "left_rep3", "right_rep1", "right_rep2", "right_rep3"];
    const repLabels = {
      left_rep1: "Izq 1", left_rep2: "Izq 2", left_rep3: "Izq 3",
      right_rep1: "Der 1", right_rep2: "Der 2", right_rep3: "Der 3"
    };
    const errorColumns = ["orden_incorrecto", "sin_puno", "sin_filo", "sin_palma", "mano_fuera", "no_responde"];
    const errorLabels = {
      orden_incorrecto: "Orden incorrecto",
      sin_puno: "Sin puno",
      sin_filo: "Sin filo",
      sin_palma: "Sin palma",
      mano_fuera: "Mano fuera",
      no_responde: "No responde"
    };
    const statuses = [
      ["correcta", "Correcta"],
      ["parcial", "Parcial"],
      ["incorrecta", "Incorrecta"],
      ["no_visible", "No visible"]
    ];
    let state = null;
    let visibleRows = [];
    let currentSampleId = null;
    let activeRep = "left_rep1";
    let saveTimer = null;
    let saving = false;
    let mirrorOn = false;

    const $ = (id) => document.getElementById(id);
    const currentRow = () => state.rows.find((row) => row.sample_id === currentSampleId);
    const isTypingTarget = (el) => ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName);

    function truthy(value) {
      return ["1", "true", "t", "yes", "y", "si", "x", "ok"].includes(String(value || "").trim().toLowerCase());
    }

    function computeScore(row) {
      return repColumns.reduce((acc, col) => acc + (row[col] === "correcta" ? 1 : 0), 0);
    }

    function colorFromScore(score) {
      if (score >= 5) return "verde";
      if (score >= 3) return "amarillo";
      return "rojo";
    }

    function formatTime(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric) || numeric < 0) return "";
      return String(Math.round(numeric * 1000) / 1000);
    }

    function autoMirrorFor(row) {
      return truthy(row?.hand_order_mirror_suggested);
    }

    function effectiveMirrorFor(row) {
      if (row?.hand_order_override === "mirror_on") return true;
      if (row?.hand_order_override === "mirror_off") return false;
      return autoMirrorFor(row);
    }

    function updateMirrorDisplay() {
      const row = currentRow();
      const video = $("video");
      if (mirrorOn) {
        video.classList.add("media-mirrored");
        $("mirrorBtn").textContent = "Espejo ON";
      } else {
        video.classList.remove("media-mirrored");
        $("mirrorBtn").textContent = "Espejo OFF";
      }
      $("mirrorAutoBtn").textContent = row?.hand_order_override ? "Volver auto" : "Auto";
    }

    function completeRow(row) {
      return repColumns.every((col) => statuses.some(([value]) => value === row[col]));
    }

    function numericCell(value) {
      const numeric = Number(String(value || "").trim());
      return Number.isFinite(numeric) ? numeric : 0;
    }

    function hasExtraReps(row) {
      return numericCell(row.left_extra_reps) > 0
        || numericCell(row.right_extra_reps) > 0
        || Boolean(String(row.extra_reps_note || "").trim());
    }

    function extrasWithoutEnd(row) {
      return completeRow(row) && hasExtraReps(row) && !String(row.exercise_end_s || "").trim();
    }

    function startedRow(row) {
      return repColumns.some((col) => row[col])
        || errorColumns.some((col) => truthy(row[col]))
        || Boolean(row.left_extra_reps || row.right_extra_reps || row.extra_reps_note)
        || Boolean(row.exercise_start_s || row.exercise_end_s || row.exercise_timing_note)
        || Boolean(row.hand_order_override)
        || Boolean(row.score_0_6 || row.color_final);
    }

    function updateProgress() {
      const progress = state.progress || {};
      $("metricComplete").textContent = progress.complete || 0;
      $("metricStarted").textContent = progress.started || 0;
      $("metricTotal").textContent = progress.total || 0;
      $("metricExtrasOpen").textContent = state.rows.filter(extrasWithoutEnd).length;
      const queueText = state.pilot_available ? "cola activa" : "sin cola activa";
      $("workingPath").textContent = `CSV de trabajo listo | ${queueText}`;
    }

    function populateSourceFilter() {
      const select = $("filterSource");
      const current = select.value;
      const sources = Array.from(new Set(state.rows.map((row) => row.source_set).filter(Boolean))).sort();
      select.innerHTML = '<option value="">Fuente</option>' + sources.map((src) => `<option value="${escapeHtml(src)}">${escapeHtml(src)}</option>`).join("");
      select.value = current;
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    function applyFilters() {
      const status = $("filterStatus").value;
      const label = $("filterLabel").value;
      const quality = $("filterQuality").value;
      const source = $("filterSource").value;
      const search = $("filterSearch").value.trim().toLowerCase();
      visibleRows = state.rows.filter((row) => {
        const complete = completeRow(row);
        const started = startedRow(row);
        const pilot = row.pilot_selected === "1";
        if (status === "extras_without_end" && !extrasWithoutEnd(row)) return false;
        if (status === "pilot" && (!pilot || complete)) return false;
        if (status === "unannotated" && complete) return false;
        if (status === "complete" && !complete) return false;
        if (status === "started" && !started) return false;
        if (label && row.label_actual !== label) return false;
        if (quality && row.calidad_video !== quality) return false;
        if (source && row.source_set !== source) return false;
        if (search) {
          const haystack = [row.sample_id, row.original_name, row.normalized_stem, row.reviewer_notes, row.pilot_reason].join(" ").toLowerCase();
          if (!haystack.includes(search)) return false;
        }
        return true;
      });
      renderList();
      if (!currentSampleId || !visibleRows.some((row) => row.sample_id === currentSampleId)) {
        selectSample(visibleRows[0]?.sample_id || null, { skipSave: true });
      }
    }

    function renderList() {
      const list = $("sampleList");
      if (!visibleRows.length) {
        list.innerHTML = '<div class="empty">No hay videos con estos filtros.</div>';
        return;
      }
      list.innerHTML = visibleRows.map((row) => {
        const complete = completeRow(row);
        const started = startedRow(row);
        const color = row.color_final || row.computed_color || colorFromScore(computeScore(row));
        const statusText = complete ? "complete" : (started ? "in progress" : "open");
        const pilotText = row.pilot_selected === "1" ? ` | cola #${escapeHtml(row.pilot_rank)}` : "";
        const extraText = extrasWithoutEnd(row) ? " | extra sin fin" : "";
        return `<button type="button" class="item ${row.sample_id === currentSampleId ? "active" : ""} ${complete ? "complete" : ""}" data-sample="${escapeHtml(row.sample_id)}">
          <div class="item-title">
            <span class="sample">${escapeHtml(row.sample_id)}</span>
            <span class="pill ${escapeHtml(color)}">${escapeHtml(color || row.label_actual)}</span>
          </div>
          <div class="subtle">${escapeHtml(row.label_actual)} | ${escapeHtml(row.calidad_video)} | ${escapeHtml(statusText)}${pilotText}${extraText}</div>
        </button>`;
      }).join("");
      list.querySelectorAll("[data-sample]").forEach((button) => {
        button.addEventListener("click", () => selectSample(button.dataset.sample));
      });
    }

    async function selectSample(sampleId, options = {}) {
      if (!options.skipSave) await saveNow();
      currentSampleId = sampleId;
      if (!sampleId) {
        $("workspace").innerHTML = '<div class="empty">No hay seleccion activa.</div>';
        return;
      }
      const row = currentRow();
      activeRep = repColumns.find((col) => !row[col]) || "left_rep1";
      renderCurrent();
      renderList();
    }

    function renderCurrent() {
      const row = currentRow();
      if (!row) return;
      mirrorOn = effectiveMirrorFor(row);
      $("currentTitle").textContent = row.pilot_selected === "1" ? `#${row.pilot_rank} ${row.sample_id}` : row.sample_id;
      $("video").src = row.video_url;
      updateMirrorDisplay();
      $("downloadLink").href = row.download_video_url;
      $("reviewSheet").src = row.review_sheet_url;
      $("metaLabel").textContent = row.label_actual || "";
      $("metaQuality").textContent = row.calidad_video || "";
      $("metaSource").textContent = row.source_set || "";
      $("metaOriginal").textContent = row.original_name || "";
      $("metaDuration").textContent = row.duration_s ? Number(row.duration_s).toFixed(1) + " s" : "";
      const anyCov = row.any_hand_coverage ? Number(row.any_hand_coverage).toFixed(2) : "";
      const bothCov = row.both_hands_coverage ? Number(row.both_hands_coverage).toFixed(2) : "";
      $("metaCoverage").textContent = anyCov || bothCov ? `any ${anyCov} / both ${bothCov}` : "";
      const handOrder = row.hand_order_status || "sin auditoria";
      const confidence = row.hand_order_confidence ? ` conf ${row.hand_order_confidence}` : "";
      const mirrorText = truthy(row.hand_order_mirror_suggested) ? " | espejo sugerido" : "";
      const overrideText = row.hand_order_override ? ` | override ${row.hand_order_override}` : "";
      $("metaHandOrder").textContent = `${handOrder}${confidence}${mirrorText}${overrideText}`;
      $("metaPilot").textContent = row.pilot_selected === "1" ? `#${row.pilot_rank} (${row.pilot_reason || "seleccionado"})` : "fuera de cola";
      $("reviewer").value = row.reviewer || "";
      $("confidence").value = row.confidence_human || "";
      $("leftExtraReps").value = row.left_extra_reps || "";
      $("rightExtraReps").value = row.right_extra_reps || "";
      $("extraRepsNote").value = row.extra_reps_note || "";
      $("exerciseStart").value = row.exercise_start_s || "";
      $("exerciseEnd").value = row.exercise_end_s || "";
      $("exerciseTimingNote").value = row.exercise_timing_note || "";
      $("notes").value = row.reviewer_notes || "";
      renderRepGrid(row);
      renderChecks(row);
      renderScore(row);
      setSaveStatus(row.annotation_complete ? "Completo" : (row.annotation_started ? "En progreso" : "Sin anotar"));
    }

    function renderScore(row) {
      const score = computeScore(row);
      const color = colorFromScore(score);
      $("scoreText").textContent = `${score}/6`;
      $("colorText").textContent = color;
      $("colorText").className = `pill ${color}`;
    }

    function renderRepGrid(row) {
      const grid = $("repGrid");
      grid.innerHTML = repColumns.map((col) => {
        const buttons = statuses.map(([value, label]) => {
          const active = row[col] === value ? "active" : "";
          return `<button type="button" class="status-btn ${active}" data-rep="${col}" data-status="${value}">${label}</button>`;
        }).join("");
        return `<div class="rep-row ${activeRep === col ? "active" : ""}" data-rep-row="${col}">
          <div class="rep-name">${repLabels[col]}</div>${buttons}
        </div>`;
      }).join("");
      grid.querySelectorAll("[data-rep-row]").forEach((el) => {
        el.addEventListener("click", () => {
          activeRep = el.dataset.repRow;
          renderRepGrid(currentRow());
        });
      });
      grid.querySelectorAll(".status-btn").forEach((button) => {
        button.addEventListener("click", (event) => {
          event.stopPropagation();
          setRepStatus(button.dataset.rep, button.dataset.status);
        });
      });
    }

    function renderChecks(row) {
      const checks = $("errorChecks");
      checks.innerHTML = errorColumns.map((col) => {
        return `<label class="check"><input type="checkbox" data-error="${col}" ${truthy(row[col]) ? "checked" : ""}> ${errorLabels[col]}</label>`;
      }).join("");
      checks.querySelectorAll("[data-error]").forEach((input) => {
        input.addEventListener("change", () => {
          const row = currentRow();
          row[input.dataset.error] = input.checked ? "1" : "";
          renderScore(row);
          scheduleSave();
        });
      });
    }

    function setRepStatus(rep, status) {
      const row = currentRow();
      if (!row) return;
      row[rep] = status;
      activeRep = repColumns[repColumns.indexOf(rep) + 1] || rep;
      row.score_0_6 = String(computeScore(row));
      row.color_final = colorFromScore(computeScore(row));
      row.annotation_complete = completeRow(row);
      row.annotation_started = startedRow(row);
      renderRepGrid(row);
      renderScore(row);
      renderList();
      scheduleSave();
    }

    function collectPayload() {
      const row = currentRow();
      const payload = { sample_id: row.sample_id };
      repColumns.forEach((col) => payload[col] = row[col] || "");
      errorColumns.forEach((col) => payload[col] = truthy(row[col]));
      payload.reviewer = $("reviewer").value;
      payload.confidence_human = $("confidence").value;
      payload.left_extra_reps = $("leftExtraReps").value;
      payload.right_extra_reps = $("rightExtraReps").value;
      payload.extra_reps_note = $("extraRepsNote").value;
      payload.exercise_start_s = $("exerciseStart").value;
      payload.exercise_end_s = $("exerciseEnd").value;
      payload.exercise_timing_note = $("exerciseTimingNote").value;
      payload.hand_order_override = row.hand_order_override || "";
      payload.reviewer_notes = $("notes").value;
      return payload;
    }

    function setSaveStatus(text, danger = false) {
      const el = $("saveStatus");
      el.textContent = text || "";
      el.className = danger ? "status-text danger-text" : "status-text";
    }

    function scheduleSave() {
      setSaveStatus("Guardando...");
      clearTimeout(saveTimer);
      saveTimer = setTimeout(saveNow, 350);
    }

    async function saveNow() {
      clearTimeout(saveTimer);
      if (!currentSampleId || saving || !state) return;
      saving = true;
      try {
        const res = await fetch("/api/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(collectPayload())
        });
        if (!res.ok) throw new Error(await res.text());
        const payload = await res.json();
        const idx = state.rows.findIndex((row) => row.sample_id === payload.row.sample_id);
        if (idx >= 0) state.rows[idx] = payload.row;
        state.progress = payload.progress;
        updateProgress();
        setSaveStatus("Guardado");
        renderList();
      } catch (err) {
        console.error(err);
        setSaveStatus("Error al guardar: " + err.message, true);
      } finally {
        saving = false;
      }
    }

    async function move(delta) {
      if (!visibleRows.length) return;
      const idx = Math.max(0, visibleRows.findIndex((row) => row.sample_id === currentSampleId));
      const next = visibleRows[Math.min(visibleRows.length - 1, Math.max(0, idx + delta))];
      if (next) await selectSample(next.sample_id);
    }

    async function nextOpen() {
      await saveNow();
      applyFilters();
      const currentIdx = visibleRows.findIndex((row) => row.sample_id === currentSampleId);
      const ordered = visibleRows.slice(currentIdx + 1).concat(visibleRows.slice(0, Math.max(0, currentIdx + 1)));
      const status = $("filterStatus").value;
      const next = status === "extras_without_end"
        ? ordered.find((row) => extrasWithoutEnd(row))
        : ordered.find((row) => !completeRow(row));
      if (next) await selectSample(next.sample_id, { skipSave: true });
    }

    function markNoResponse() {
      const row = currentRow();
      if (!row) return;
      repColumns.forEach((col) => row[col] = "incorrecta");
      row.no_responde = "1";
      row.score_0_6 = "0";
      row.color_final = "rojo";
      row.annotation_complete = true;
      row.annotation_started = true;
      renderCurrent();
      scheduleSave();
    }

    async function loadState() {
      const res = await fetch("/api/state");
      if (!res.ok) throw new Error(await res.text());
      state = await res.json();
      const extrasOpen = state.rows.filter(extrasWithoutEnd).length;
      if (extrasOpen > 0) {
        $("filterStatus").value = "extras_without_end";
      }
      if (!state.pilot_available && $("filterStatus").value === "pilot") {
        $("filterStatus").value = "unannotated";
      }
      if (extrasOpen === 0 && $("filterStatus").value === "extras_without_end") {
        $("filterStatus").value = state.pilot_available ? "pilot" : "all";
      }
      populateSourceFilter();
      updateProgress();
      applyFilters();
    }

    ["filterStatus", "filterLabel", "filterQuality", "filterSource"].forEach((id) => {
      $(id).addEventListener("change", applyFilters);
    });
    $("filterSearch").addEventListener("input", applyFilters);
    $("reviewer").addEventListener("input", scheduleSave);
    $("confidence").addEventListener("change", scheduleSave);
    $("leftExtraReps").addEventListener("input", () => {
      const row = currentRow();
      if (row) row.left_extra_reps = $("leftExtraReps").value;
      scheduleSave();
    });
    $("rightExtraReps").addEventListener("input", () => {
      const row = currentRow();
      if (row) row.right_extra_reps = $("rightExtraReps").value;
      scheduleSave();
    });
    $("extraRepsNote").addEventListener("input", () => {
      const row = currentRow();
      if (row) row.extra_reps_note = $("extraRepsNote").value;
      scheduleSave();
    });
    $("exerciseStart").addEventListener("input", () => {
      const row = currentRow();
      if (row) row.exercise_start_s = $("exerciseStart").value;
      scheduleSave();
    });
    $("exerciseEnd").addEventListener("input", () => {
      const row = currentRow();
      if (row) row.exercise_end_s = $("exerciseEnd").value;
      scheduleSave();
    });
    $("exerciseTimingNote").addEventListener("input", () => {
      const row = currentRow();
      if (row) row.exercise_timing_note = $("exerciseTimingNote").value;
      scheduleSave();
    });
    $("notes").addEventListener("input", scheduleSave);
    $("prevBtn").addEventListener("click", () => move(-1));
    $("nextBtn").addEventListener("click", () => move(1));
    $("nextOpenBtn").addEventListener("click", nextOpen);
    $("saveBtn").addEventListener("click", saveNow);
    $("noResponseBtn").addEventListener("click", markNoResponse);
    $("markStartBtn").addEventListener("click", () => {
      const row = currentRow();
      if (!row) return;
      const value = formatTime($("video").currentTime || 0);
      row.exercise_start_s = value;
      $("exerciseStart").value = value;
      scheduleSave();
    });
    $("markEndBtn").addEventListener("click", () => {
      const row = currentRow();
      if (!row) return;
      const value = formatTime($("video").currentTime || 0);
      row.exercise_end_s = value;
      $("exerciseEnd").value = value;
      scheduleSave();
    });
    $("clearTimingBtn").addEventListener("click", () => {
      const row = currentRow();
      if (!row) return;
      row.exercise_start_s = "";
      row.exercise_end_s = "";
      row.exercise_timing_note = "";
      $("exerciseStart").value = "";
      $("exerciseEnd").value = "";
      $("exerciseTimingNote").value = "";
      scheduleSave();
    });
    $("mirrorBtn").addEventListener("click", () => {
      const row = currentRow();
      mirrorOn = !mirrorOn;
      if (row) {
        row.hand_order_override = mirrorOn ? "mirror_on" : "mirror_off";
      }
      updateMirrorDisplay();
      renderCurrent();
      scheduleSave();
    });
    $("mirrorAutoBtn").addEventListener("click", () => {
      const row = currentRow();
      if (!row) return;
      row.hand_order_override = "";
      mirrorOn = effectiveMirrorFor(row);
      updateMirrorDisplay();
      renderCurrent();
      scheduleSave();
    });

    document.addEventListener("keydown", (event) => {
      if (event.ctrlKey && event.key.toLowerCase() === "s") {
        event.preventDefault();
        saveNow();
        return;
      }
      if (isTypingTarget(event.target)) return;
      const keyMap = { "1": "correcta", "2": "parcial", "3": "incorrecta", "4": "no_visible" };
      if (keyMap[event.key]) {
        setRepStatus(activeRep, keyMap[event.key]);
      } else if (event.key.toLowerCase() === "n") {
        nextOpen();
      } else if (event.key === "ArrowRight") {
        move(1);
      } else if (event.key === "ArrowLeft") {
        move(-1);
      }
    });

    loadState().catch((err) => {
      console.error(err);
      setSaveStatus("No se pudo cargar: " + err.message, true);
    });
  </script>
</body>
</html>
"""


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    server: "AnnotationHTTPServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self.send_bytes(APP_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self.send_json(self.server.store.state())
            elif parsed.path == "/health":
                self.send_json({"ok": True, "app_version": APP_VERSION})
            elif parsed.path.startswith("/media/video/"):
                sample_id = unquote(parsed.path.removeprefix("/media/video/"))
                self.send_file(self.server.store.media_path(sample_id, "video"))
            elif parsed.path.startswith("/media/sheet/"):
                sample_id = unquote(parsed.path.removeprefix("/media/sheet/"))
                self.send_file(self.server.store.media_path(sample_id, "sheet"))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
        except KeyError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
        except PermissionError as exc:
            self.send_error(HTTPStatus.FORBIDDEN, str(exc))
        except FileNotFoundError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Payload too large")
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json(self.server.store.save(payload))
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except KeyError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(response_json(payload), "application/json; charset=utf-8", status=status)

    def send_bytes(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_file(self, path: Path) -> None:
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        size = path.stat().st_size
        range_header = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
        if match:
            start_text, end_text = match.groups()
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else size - 1
            end = min(end, size - 1)
            if start >= size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", content_type)
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                self._copy_limited(fh, end - start + 1)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with path.open("rb") as fh:
            self._copy_limited(fh, size)

    def _copy_limited(self, fh: Any, remaining: int) -> None:
        while remaining > 0:
            chunk = fh.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            self.wfile.write(chunk)
            remaining -= len(chunk)

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.quiet:
            return
        super().log_message(format, *args)


class AnnotationHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], store: AnnotationStore, quiet: bool = False) -> None:
        super().__init__(server_address, AnnotationRequestHandler)
        self.store = store
        self.quiet = quiet


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local EDM v3 annotation web app.")
    parser.add_argument("--base", default=str(videos_base()), help="Videos base directory.")
    parser.add_argument("--template", default=None, help="Annotation template CSV.")
    parser.add_argument("--annotations", default=None, help="Working annotations CSV to create/update.")
    parser.add_argument("--queue", default=None, help="Optional pilot queue CSV.")
    parser.add_argument("--hand-order", default=None, help="Optional hand-order audit CSV.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base = Path(args.base).resolve()
    out_dir = artifacts_v3_dir(base)
    template_csv = Path(args.template).resolve() if args.template else out_dir / "edm_annotation_template.csv"
    working_csv = Path(args.annotations).resolve() if args.annotations else out_dir / "edm_annotations_working.csv"
    default_queue = out_dir / "edm_annotation_pilot_queue.csv"
    queue_csv = Path(args.queue).resolve() if args.queue else (default_queue if default_queue.exists() else None)
    default_hand_order = out_dir / "edm_hand_order_audit.csv"
    hand_order_csv = Path(args.hand_order).resolve() if args.hand_order else (default_hand_order if default_hand_order.exists() else None)
    store = AnnotationStore(
        base=base,
        template_csv=template_csv,
        working_csv=working_csv,
        queue_csv=queue_csv,
        hand_order_csv=hand_order_csv,
    )
    server = AnnotationHTTPServer((args.host, args.port), store=store, quiet=args.quiet)
    url = f"http://{args.host}:{args.port}/"
    print("=== NEUROLAP EDM V3 ANNOTATION APP ===", flush=True)
    print(f"url: {url}", flush=True)
    print(f"template: {template_csv}", flush=True)
    print(f"working: {working_csv}", flush=True)
    print(f"queue: {queue_csv if queue_csv else 'none'}", flush=True)
    print(f"hand_order: {hand_order_csv if hand_order_csv else 'none'}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation app.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
