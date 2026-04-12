from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_MD_PATHS = {
    "vlm_page_translate": REPO_ROOT / "skills" / "vlm_page_translate" / "SKILL.md",
    "vlm_translation_revision": REPO_ROOT / "skills" / "vlm_translation_revision" / "SKILL.md",
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


def sanitize_strategy_suggestions(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Any] = {}

    preferred_action = raw.get("preferred_rerun_action")
    allowed_actions = {
        "retry_typeset",
        "retry_erase",
        "retry_vlm_page",
        "retry_translation",
        "retry_ocr",
        "retry_detection",
        "run_revision",
    }
    if str(preferred_action or "").strip().lower() in allowed_actions:
        out["preferred_rerun_action"] = str(preferred_action).strip().lower()

    preferred_correction = raw.get("preferred_correction_action")
    allowed_correction_actions = {
        "run_revision",
        "retry_vlm_page",
        "retry_translation",
        "retry_ocr",
        "retry_detection",
    }
    if str(preferred_correction or "").strip().lower() in allowed_correction_actions:
        out["preferred_correction_action"] = str(preferred_correction).strip().lower()

    post_translation_action = raw.get("preferred_post_translation_action")
    if str(post_translation_action or "").strip().lower() in {"auto_revision_then_erase", "erase_direct"}:
        out["preferred_post_translation_action"] = str(post_translation_action).strip().lower()

    revision_scope = raw.get("revision_scope")
    if str(revision_scope or "").strip().lower() in {"page_only", "global"}:
        out["revision_scope"] = str(revision_scope).strip().lower()

    revision_page_limit = raw.get("revision_page_limit")
    try:
        if revision_page_limit is not None:
            limit = int(revision_page_limit)
            out["revision_page_limit"] = max(1, min(24, limit))
    except Exception:
        pass

    reuse_erased = raw.get("reuse_erased_image")
    if isinstance(reuse_erased, bool):
        out["reuse_erased_image"] = reuse_erased
    elif isinstance(reuse_erased, str):
        text = reuse_erased.strip().lower()
        if text in {"true", "1", "yes"}:
            out["reuse_erased_image"] = True
        elif text in {"false", "0", "no"}:
            out["reuse_erased_image"] = False

    page_profile = raw.get("page_profile")
    if str(page_profile or "").strip().lower() in {"default", "art_integrated", "boxed_dialogue"}:
        out["page_profile"] = str(page_profile).strip().lower()

    recommended_page_action = raw.get("recommended_page_action")
    allowed_page_actions = {
        "continue",
        "ask_user",
        "retry_typeset",
        "retry_erase",
        "retry_vlm_page",
        "retry_translation",
        "retry_ocr",
        "retry_detection",
        "run_revision",
    }
    if str(recommended_page_action or "").strip().lower() in allowed_page_actions:
        out["recommended_page_action"] = str(recommended_page_action).strip().lower()

    ask_user_reason = raw.get("ask_user_reason")
    if isinstance(ask_user_reason, str):
        ask_user_reason = ask_user_reason.strip()
        if ask_user_reason:
            out["ask_user_reason"] = ask_user_reason

    return out


def sanitize_job_strategy_suggestions(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    out: Dict[str, Any] = {}

    recommended_job_action = raw.get("recommended_job_action")
    if str(recommended_job_action or "").strip().lower() in {"continue", "run_vlm_batch", "ask_user"}:
        out["recommended_job_action"] = str(recommended_job_action).strip().lower()

    ask_user_reason = raw.get("ask_user_reason")
    if isinstance(ask_user_reason, str):
        ask_user_reason = ask_user_reason.strip()
        if ask_user_reason:
            out["ask_user_reason"] = ask_user_reason

    batching_mode = raw.get("batching_mode")
    if str(batching_mode or "").strip().lower() in {"contiguous_window"}:
        out["batching_mode"] = str(batching_mode).strip().lower()

    vlm_batch_size = raw.get("vlm_batch_size")
    try:
        if vlm_batch_size is not None:
            out["vlm_batch_size"] = max(1, min(10, int(vlm_batch_size)))
    except Exception:
        pass

    page_indices = raw.get("vlm_batch_page_indices")
    if isinstance(page_indices, list):
        clean_indices = []
        for item in page_indices[:10]:
            try:
                clean_indices.append(int(item))
            except Exception:
                continue
        if clean_indices:
            out["vlm_batch_page_indices"] = clean_indices

    return out
