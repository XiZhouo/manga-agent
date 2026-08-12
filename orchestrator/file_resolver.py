from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


SUPPORTED_SUFFIXES = {".cbz", ".zip", ".cbr", ".rar", ".pdf", ".png", ".jpg", ".jpeg", ".bmp"}


def extract_path_candidates(text: str) -> List[str]:
    raw = str(text or "")
    found: List[str] = []

    for pattern in (r"`([^`]+)`", r'"([^"]+)"', r"'([^']+)'"):
        for match in re.findall(pattern, raw):
            candidate = str(match).strip()
            if candidate:
                found.append(candidate)

    bare_pattern = r"([~/\.\w\-\(\)\[\] ]+\.(?:cbz|zip|cbr|rar|pdf|png|jpg|jpeg|bmp|json))"
    for match in re.findall(bare_pattern, raw, flags=re.IGNORECASE):
        candidate = str(match).strip()
        if candidate:
            found.append(candidate)

    deduped: List[str] = []
    seen = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def resolve_input_path(raw: str, cwd: Path, *, allowed_suffixes: Optional[set[str]] = None) -> Optional[Path]:
    text = str(raw or "").strip()
    if not text:
        return None

    suffixes = allowed_suffixes or SUPPORTED_SUFFIXES
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (cwd / path).resolve()

    if path.exists() and path.is_file() and path.suffix.lower() in suffixes:
        return path
    return None


def find_by_filename(
    name: str,
    cwd: Path,
    extra_dirs: Optional[List[Path]] = None,
    *,
    allowed_suffixes: Optional[set[str]] = None,
) -> Optional[Path]:
    target = str(name or "").strip()
    if not target:
        return None

    suffixes = allowed_suffixes or SUPPORTED_SUFFIXES
    search_dirs = [cwd, cwd / "mangatest", Path.home() / "Downloads"]
    for item in extra_dirs or []:
        if item not in search_dirs:
            search_dirs.append(item)

    for base in search_dirs:
        if not base.exists() or not base.is_dir():
            continue
        for path in base.glob("*"):
            if not path.is_file():
                continue
            if path.name != target:
                continue
            if path.suffix.lower() not in suffixes:
                continue
            return path.resolve()
    return None
