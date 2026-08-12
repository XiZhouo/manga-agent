from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


SKILL_OUTPUTS_DIRNAME = "skill_outputs"


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_for_json(data), ensure_ascii=False, indent=2), encoding="utf-8")


def slugify_job_stem(name: str, *, fallback: str = "job", max_length: int = 48) -> str:
    stem = Path(str(name or fallback)).stem
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    if not stem:
        stem = fallback
    return stem[:max_length].strip("_") or fallback


def build_job_id(source_file_name: str, *, now: datetime | None = None) -> str:
    dt = (now or datetime.now().astimezone()).replace(microsecond=0)
    slug = slugify_job_stem(source_file_name)
    timestamp = dt.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{slug}_{timestamp}_{short}"


def skill_outputs_dir(job_dir: Path) -> Path:
    return Path(job_dir) / SKILL_OUTPUTS_DIRNAME


def skill_dir(job_dir: Path, skill_name: str) -> Path:
    return skill_outputs_dir(job_dir) / str(skill_name)


def page_skill_dir(job_dir: Path, skill_name: str, page_index: int) -> Path:
    return skill_dir(job_dir, skill_name) / f"page_{int(page_index):04d}"


def manifest_dir(job_dir: Path) -> Path:
    return Path(job_dir) / "manifest"


def resolve_job_dir_from_path(path: Path) -> Path:
    p = Path(path)
    for ancestor in [p.parent, *p.parents]:
        if ancestor.name == SKILL_OUTPUTS_DIRNAME:
            return ancestor.parent

    if p.parent.name in {"pages", "erased", "rendered", "masks"}:
        return p.parent.parent
    if p.parent != Path("."):
        return p.parent
    return p.parent
