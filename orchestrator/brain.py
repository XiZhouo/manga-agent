from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.llm_adapter import LLMConfig, build_chat_llm


@dataclass(frozen=True)
class DecisionBrainConfig:
    """Configuration for the decision LLM (agent brain).

    This LLM is responsible for interpreting user feedback and producing a structured plan.
    """

    enabled: bool = True
    model: str = "qwen3-max"
    temperature: float = 0.0


def _json_loads_best_effort(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _extract_page_number_heuristic(msg: str) -> Optional[int]:
    """Best-effort: extract a human page number like '第12页' -> 12."""

    m = re.search(r"第\s*(\d+)\s*页", msg)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    # fallback: first integer
    digits = []
    reading = False
    for ch in msg:
        if ch.isdigit():
            digits.append(ch)
            reading = True
        elif reading:
            break
    if digits:
        try:
            return int("".join(digits))
        except Exception:
            return None
    return None


class DecisionBrain:
    """LLM-based agent brain.

    v0 scope:
    - Parse user feedback into a structured plan for the orchestrator.
    - Decide whether to ask a clarification question (reference example / scope).

    NOTE: This brain does NOT execute skills. It only proposes a plan.
    """

    def __init__(self, config: DecisionBrainConfig | None = None) -> None:
        self.config = config or DecisionBrainConfig()

    def plan_user_message(
        self,
        *,
        user_message: str,
        page_count: int,
        pipeline_mode: str,
        target_language: str,
    ) -> Dict[str, Any]:
        """Return a structured plan for a user message.

        Output schema:
        {
          intent: correction|rerun_page|preference|unknown,
          needs_clarification: bool,
          clarifying_question: str|null,
          reference_page_number: int|null,   # human-facing number (usually 1-based)
          scope: page_only|global,
          from_side: src|tgt|auto,
          from_text: str|null,
          to_text: str|null,
          glossary_type: term|quote|name|phrase|null,
          apply_glossary: bool,
          confidence: float|null
        }
        """

        cfg = self.config
        if not cfg.enabled:
            return {
                "intent": "unknown",
                "needs_clarification": False,
                "clarifying_question": None,
                "reference_page_number": None,
                "scope": "page_only",
                "from_side": "auto",
                "from_text": None,
                "to_text": None,
                "glossary_type": None,
                "apply_glossary": False,
                "confidence": None,
            }

        # Hard-disable via env
        mode = (os.getenv("MANGA_AGENT_BRAIN_MODE", "auto") or "auto").lower().strip()
        if mode == "off":
            return {
                "intent": "unknown",
                "needs_clarification": False,
                "clarifying_question": None,
                "reference_page_number": _extract_page_number_heuristic(user_message),
                "scope": "page_only",
                "from_side": "auto",
                "from_text": None,
                "to_text": None,
                "glossary_type": None,
                "apply_glossary": False,
                "confidence": None,
            }

        # Requires API key (DashScope/OpenAI compat)
        if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
            # fallback heuristic plan (no LLM)
            return {
                "intent": "unknown",
                "needs_clarification": False,
                "clarifying_question": None,
                "reference_page_number": _extract_page_number_heuristic(user_message),
                "scope": "global" if any(k in user_message for k in ("所有", "全部", "全书", "全本", "都")) else "page_only",
                "from_side": "auto",
                "from_text": None,
                "to_text": None,
                "glossary_type": None,
                "apply_glossary": False,
                "confidence": None,
            }

        model = os.getenv("MANGA_AGENT_BRAIN_MODEL", cfg.model)
        temperature = float(os.getenv("MANGA_AGENT_BRAIN_TEMPERATURE", str(cfg.temperature)))

        system = SystemMessage(
            content=(
                "You are the decision brain of a comic translation agent. "
                "You interpret user feedback and produce a STRICT JSON plan for the orchestrator. "
                "Do NOT ask the user whether 'xxx is source or target'; infer it yourself. "
                "Only ask clarification when required: reference example (page number) and scope (page-only vs global). "
                "Target language is provided. "
                "Return STRICT JSON only (no markdown)."
            )
        )

        payload = {
            "task": "plan_user_message",
            "user_message": user_message,
            "context": {
                "page_count": page_count,
                "pipeline_mode": pipeline_mode,
                "target_language": target_language,
                "note": "User page numbers are human-facing; usually 1-based.",
            },
            "output_schema": {
                "intent": "correction|rerun_page|preference|unknown",
                "needs_clarification": "bool",
                "clarifying_question": "string|null",
                "reference_page_number": "int|null",
                "scope": "page_only|global",
                "from_side": "src|tgt|auto",
                "from_text": "string|null",
                "to_text": "string|null",
                "glossary_type": "term|quote|name|phrase|null",
                "apply_glossary": "bool",
                "confidence": "float_0_to_1|null",
            },
        }

        llm = build_chat_llm(
            LLMConfig(
                model=model,
                temperature=temperature,
            )
        )

        resp = llm.invoke([system, HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = _json_loads_best_effort(text)

        # Normalize
        intent = str(data.get("intent") or "unknown").strip().lower()
        if intent not in {"correction", "rerun_page", "preference", "unknown"}:
            intent = "unknown"

        scope = str(data.get("scope") or "page_only").strip().lower()
        if scope not in {"page_only", "global"}:
            scope = "page_only"

        from_side = str(data.get("from_side") or "auto").strip().lower()
        if from_side not in {"src", "tgt", "auto"}:
            from_side = "auto"

        glossary_type = data.get("glossary_type")
        if glossary_type is not None:
            glossary_type = str(glossary_type).strip().lower() or None
            if glossary_type not in {"term", "quote", "name", "phrase"}:
                glossary_type = None

        ref_num = data.get("reference_page_number")
        try:
            ref_num = int(ref_num) if ref_num is not None else None
        except Exception:
            ref_num = None

        needs_clarification = bool(data.get("needs_clarification"))
        clarifying_question = data.get("clarifying_question")
        if isinstance(clarifying_question, str):
            clarifying_question = clarifying_question.strip() or None
        else:
            clarifying_question = None

        from_text = data.get("from_text")
        if isinstance(from_text, str):
            from_text = from_text.strip() or None
        else:
            from_text = None

        to_text = data.get("to_text")
        if isinstance(to_text, str):
            to_text = to_text.strip() or None
        else:
            to_text = None

        apply_glossary = bool(data.get("apply_glossary"))

        conf = data.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None

        return {
            "intent": intent,
            "needs_clarification": needs_clarification,
            "clarifying_question": clarifying_question,
            "reference_page_number": ref_num,
            "scope": scope,
            "from_side": from_side,
            "from_text": from_text,
            "to_text": to_text,
            "glossary_type": glossary_type,
            "apply_glossary": apply_glossary,
            "confidence": conf,
        }
