from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_MD_PATHS = {
    "lama_eraser": REPO_ROOT / "skills" / "lama_erase_server" / "SKILL.md",
    "lettering": REPO_ROOT / "skills" / "lettering_basic_v0" / "SKILL.md",
}


def _extract_json_block(text: str, heading: str = "## Brain Param Spec") -> Optional[Dict[str, Any]]:
    pattern = re.escape(heading) + r".*?```json\s*(\{.*?\})\s*```"
    m = re.search(pattern, text, flags=re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


@lru_cache(maxsize=1)
def load_skill_param_specs() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for skill_name, path in SKILL_MD_PATHS.items():
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        spec = _extract_json_block(text)
        if isinstance(spec, dict):
            out[skill_name] = spec
    return out


def build_skill_context_for_brain() -> Dict[str, Any]:
    specs = load_skill_param_specs()
    summary: Dict[str, Any] = {}
    for skill_name, spec in specs.items():
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        summary[skill_name] = {
            "purpose": spec.get("purpose"),
            "when_to_use": spec.get("when_to_use"),
            "params": params,
        }
    return summary


def _sanitize_param(name: str, value: Any, rule: Dict[str, Any]) -> Any:
    rtype = str(rule.get("type") or "").strip().lower()
    if rtype == "enum":
        values = [str(x) for x in (rule.get("values") or [])]
        if str(value) in values:
            return str(value)
        return None
    if rtype == "int":
        try:
            iv = int(round(float(value)))
        except Exception:
            return None
        min_v = rule.get("min")
        max_v = rule.get("max")
        if min_v is not None and iv < int(min_v):
            iv = int(min_v)
        if max_v is not None and iv > int(max_v):
            iv = int(max_v)
        return iv
    if rtype == "float":
        try:
            fv = float(value)
        except Exception:
            return None
        min_v = rule.get("min")
        max_v = rule.get("max")
        if min_v is not None and fv < float(min_v):
            fv = float(min_v)
        if max_v is not None and fv > float(max_v):
            fv = float(max_v)
        return fv
    if rtype == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
        return None
    return None


def sanitize_skill_param_suggestions(raw: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    specs = load_skill_param_specs()
    out: Dict[str, Dict[str, Any]] = {}
    for skill_name, params in raw.items():
        if skill_name not in specs or not isinstance(params, dict):
            continue
        param_rules = specs[skill_name].get("params") or {}
        if not isinstance(param_rules, dict):
            continue
        clean: Dict[str, Any] = {}
        for param_name, value in params.items():
            rule = param_rules.get(param_name)
            if not isinstance(rule, dict):
                continue
            sanitized = _sanitize_param(param_name, value, rule)
            if sanitized is not None:
                clean[param_name] = sanitized
        if clean:
            out[skill_name] = clean
    return out
