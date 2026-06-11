import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


LABELS = ("amarillo", "rojo", "verde")
VALID_VIDEO_RE = re.compile(r"\.\s*(mp4|mov|mkv|avi)$", re.IGNORECASE)


def videos_base() -> Path:
    return Path(__file__).resolve().parents[1]


def artifacts_v2_dir(base: Optional[Path] = None) -> Path:
    root = Path(base) if base else videos_base()
    return root / "artifacts" / "v2"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_label(value: str) -> str:
    return str(value).strip().casefold()


def label_from_path(path: Path) -> Optional[str]:
    for part in path.parts:
        clean = clean_label(part)
        if clean in LABELS:
            return clean
    return None


def is_video_file(path: Path) -> bool:
    return bool(VALID_VIDEO_RE.search(path.name.strip()))


def strip_video_extension(name: str) -> str:
    return VALID_VIDEO_RE.sub("", str(name).strip())


def normalize_stem(name_or_path: str) -> str:
    stem = strip_video_extension(os.path.basename(str(name_or_path)))
    stem = re.sub(r"(?i)[_-]?15fps$", "", stem)
    stem = stem.replace("-", "_").replace(" ", "")
    stem = re.sub(r"[\(\[]\s*\d+\s*[\)\]]$", "", stem)
    return stem.strip().casefold()


def safe_slug(value: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().casefold())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return (slug or "unnamed")[:max_len]


def rel_to(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_source_path(path_value: str, base: Optional[Path] = None) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (Path(base) if base else videos_base()) / path


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_csv_list(values: Optional[Iterable[str]]) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend([v.strip() for v in str(value).split(",") if v.strip()])
    return out

