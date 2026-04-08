from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# NOTE: LangChain dependencies are optional for offline/stub runs.
# We import them lazily only when the judge is actually invoked.


@dataclass(frozen=True)
class VLMJudgeConfig:
    """Configuration for the text-only judge (no images)."""

    enabled: bool = True
    mode: str = "auto"  # off|auto|on

    # Use a cheaper text model for judging.
    model: str = "qwen3-max"
    temperature: float = 0.0

    max_regions: int = 12
    max_text_len: int = 280


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _json_loads_best_effort(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _has_cjk(s: str) -> bool:
    for ch in s:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            return True
    return False


def should_judge(*, mode: str, regions: List[Dict[str, Any]], target_language: str, warnings: List[str]) -> bool:
    m = (mode or "auto").lower().strip()
    if m == "off":
        return False
    if m == "on":
        return True

    n = len(regions)
    if n == 0:
        # No text evidence to judge.
        return False

    # auto: judge when there is some uncertainty signal
    if warnings:
        return True

    if n >= 20:
        return True

    # any empties -> judge
    empty_src = sum(1 for r in regions if not (r.get("src_text") or "").strip())
    empty_tgt = sum(1 for r in regions if not (r.get("tgt_text") or "").strip())
    if empty_src or empty_tgt:
        return True

    if target_language == "zh":
        nonempty = [r for r in regions if (r.get("tgt_text") or "").strip()]
        if nonempty:
            cjk = sum(1 for r in nonempty if _has_cjk(r.get("tgt_text") or ""))
            if (cjk / float(len(nonempty))) < 0.75:
                return True

    return False


def judge_vlm_page_text_only(
    *,
    page_index: int,
    target_language: str,
    reading_direction: str,
    regions: List[Dict[str, Any]],
    metrics: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    config: Optional[VLMJudgeConfig] = None,
) -> Dict[str, Any]:
    """Judge VLM output quality using text-only inputs.

    Returns envelope:
    {
      ok: bool,
      data: { decision, reason_code, explanation, confidence, warnings[] },
      metrics: { usage? },
      error: {code,message,detail?} | null
    }
    """

    cfg = config or VLMJudgeConfig()
    if not cfg.enabled:
        return {"ok": True, "data": {"decision": "proceed", "reason_code": "judge_disabled", "explanation": "judge disabled", "confidence": 1.0, "warnings": []}, "metrics": {}, "error": None}

    # Allow env overrides
    mode = os.getenv("MANGA_AGENT_VLM_JUDGE_MODE", cfg.mode)
    model = os.getenv("MANGA_AGENT_VLM_JUDGE_MODEL", cfg.model)
    temperature = float(os.getenv("MANGA_AGENT_VLM_JUDGE_TEMPERATURE", str(cfg.temperature)))
    max_regions = int(os.getenv("MANGA_AGENT_VLM_JUDGE_MAX_REGIONS", str(cfg.max_regions)))
    max_text_len = int(os.getenv("MANGA_AGENT_VLM_JUDGE_MAX_TEXT_LEN", str(cfg.max_text_len)))

    warn_list = list(warnings or [])

    mode_norm = (mode or "auto").lower().strip()
    if mode_norm == "off":
        return {
            "ok": True,
            "data": {
                "decision": "proceed",
                "reason_code": "mode_off",
                "explanation": "Judge disabled (mode=off).",
                "confidence": 1.0,
                "warnings": warn_list,
            },
            "metrics": {},
            "error": None,
        }

    if not should_judge(mode=mode, regions=regions, target_language=target_language, warnings=warn_list):
        return {
            "ok": True,
            "data": {
                "decision": "proceed",
                "reason_code": "auto_skip",
                "explanation": "Heuristics did not indicate uncertainty; skipping judge.",
                "confidence": 0.7,
                "warnings": warn_list,
            },
            "metrics": {},
            "error": None,
        }

    # Prepare compact evidence
    sample = []
    for r in regions[:max_regions]:
        sample.append(
            {
                "region_id": r.get("region_id"),
                "src_text": _truncate(str(r.get("src_text") or ""), max_text_len),
                "tgt_text": _truncate(str(r.get("tgt_text") or ""), max_text_len),
            }
        )

    payload = {
        "task": "vlm_judge_text_only",
        "page_index": page_index,
        "reading_direction": reading_direction,
        "target_language": target_language,
        "metrics": metrics or {},
        "heuristic_warnings": warn_list,
        "region_sample": sample,
        "output_schema": {
            "decision": "proceed|retry|ask_user",
            "reason_code": "snake_case_short",
            "explanation": "one_sentence",
            "confidence": "float_0_to_1",
            "warnings": ["string"],
        },
    }

    # build_chat_llm reads keys from DASHSCOPE_API_KEY / OPENAI_API_KEY.
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        return {
            "ok": False,
            "data": None,
            "metrics": {},
            "error": {
                "code": "missing_api_key",
                "message": "Missing DASHSCOPE_API_KEY (preferred) or OPENAI_API_KEY for judge call.",
            },
        }

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from orchestrator.llm_adapter import LLMConfig, build_chat_llm
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "metrics": {},
            "error": {
                "code": "missing_dependency",
                "message": f"Missing judge dependencies (langchain). Install requirements to enable judge. Detail: {e}",
            },
        }

    system = SystemMessage(
        content=(
            "You are a QA gate for a comic translation pipeline. "
            "You will receive text-only OCR+translation outputs per region (no image). "
            "Decide whether to PROCEED to lettering, RETRY the VLM core once, or ASK_USER for review. "
            "Only choose RETRY if there is strong evidence the output is broken (e.g., many empty texts, wrong language, obvious gibberish). "
            "Do NOT request retry for style improvements. "
            "Return STRICT JSON only (no markdown)."
        )
    )

    human = HumanMessage(content=json.dumps(payload, ensure_ascii=False))

    llm = build_chat_llm(
        LLMConfig(
            model=model,
            temperature=temperature,
        )
    )

    try:
        resp = llm.invoke([system, human])
    except Exception as e:
        return {"ok": False, "data": None, "metrics": {}, "error": {"code": "judge_request_failed", "message": str(e)}}

    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    try:
        data = _json_loads_best_effort(text)
    except Exception as e:
        return {"ok": False, "data": None, "metrics": {}, "error": {"code": "judge_bad_json", "message": str(e), "detail": text}}

    decision = str(data.get("decision") or "proceed").strip().lower()
    if decision not in {"proceed", "retry", "ask_user"}:
        decision = "proceed"

    out = {
        "decision": decision,
        "reason_code": str(data.get("reason_code") or "").strip() or None,
        "explanation": str(data.get("explanation") or "").strip() or None,
        "confidence": None,
        "warnings": [str(x) for x in (data.get("warnings") or []) if str(x).strip()],
    }

    try:
        if data.get("confidence") is not None:
            out["confidence"] = float(data.get("confidence"))
    except Exception:
        out["confidence"] = None

    # Best-effort usage extraction (LangChain may vary by provider)
    usage_out = None
    try:
        meta = getattr(resp, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage")
        if isinstance(usage, dict):
            usage_out = usage
    except Exception:
        usage_out = None

    return {"ok": True, "data": out, "metrics": {"usage": usage_out}, "error": None}
