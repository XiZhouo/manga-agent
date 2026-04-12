from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

# NOTE: LangChain dependencies are optional for offline/stub runs.
# We import them lazily only when the brain is actually invoked.


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

    def suggest_job_strategy(
        self,
        *,
        job_summary: Dict[str, Any],
        pipeline_mode: str,
        target_language: str,
        skill_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return proactive job-level strategy suggestions.

        This sits above page-level strategy and is mainly responsible for
        batching / queue management, especially for VLM page windows.
        """

        def _heuristic() -> Dict[str, Any]:
            next_not_started = [int(x) for x in (job_summary.get("next_not_started_page_indices") or []) if isinstance(x, int)]
            active_pages = int(job_summary.get("active_page_count") or 0)
            preferred_batch_size = int(job_summary.get("preferred_vlm_batch_size") or 10)
            preferred_batch_size = max(1, min(10, preferred_batch_size))

            strategy: Dict[str, Any] = {}
            summary_parts = []

            if pipeline_mode == "vlm" and active_pages == 0 and next_not_started:
                chosen = next_not_started[:preferred_batch_size]
                strategy["recommended_job_action"] = "run_vlm_batch"
                strategy["vlm_batch_size"] = len(chosen)
                strategy["vlm_batch_page_indices"] = chosen
                strategy["batching_mode"] = "contiguous_window"
                summary_parts.append(f"schedule vlm batch for pages {chosen[0]}-{chosen[-1]}")
            else:
                strategy["recommended_job_action"] = "continue"
                if active_pages > 0:
                    summary_parts.append("finish active pages before opening another vlm batch")
                elif not next_not_started:
                    summary_parts.append("no pending vlm pages left to batch")

            return {
                "strategy_suggestions": strategy,
                "summary": "; ".join(summary_parts) if summary_parts else "default job strategy",
                "confidence": 0.75,
            }

        cfg = self.config
        mode = (os.getenv("MANGA_AGENT_BRAIN_MODE", "auto") or "auto").lower().strip()
        if (not cfg.enabled) or mode == "off":
            return _heuristic()

        if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
            return _heuristic()

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from orchestrator.llm_adapter import LLMConfig, build_chat_llm
        except Exception:
            return _heuristic()

        system = SystemMessage(
            content=(
                "You are the job-strategy brain of a comic translation agent. "
                "You receive a structured job_summary with queue state and page-stage counts. "
                "Recommend SAFE, minimal job-level strategy updates. "
                "Focus on queueing and VLM batching, not page-local typography decisions. "
                "If VLM mode is active, prefer grouping the next contiguous pages into one small batch window rather than sending the whole book. "
                "Return STRICT JSON only (no markdown)."
            )
        )

        payload = {
            "task": "suggest_job_strategy",
            "job_summary": job_summary,
            "context": {
                "pipeline_mode": pipeline_mode,
                "target_language": target_language,
            },
            "skill_context": skill_context or {},
            "output_schema": {
                "strategy_suggestions": {
                    "recommended_job_action": "continue|run_vlm_batch|ask_user",
                    "ask_user_reason": "string|null",
                    "batching_mode": "contiguous_window",
                    "vlm_batch_size": "int_1_to_10",
                    "vlm_batch_page_indices": "int[]_len_1_to_10",
                },
                "summary": "string",
                "confidence": "float_0_to_1|null",
            },
        }

        llm = build_chat_llm(
            LLMConfig(
                model=os.getenv("MANGA_AGENT_BRAIN_MODEL", cfg.model),
                temperature=float(os.getenv("MANGA_AGENT_BRAIN_TEMPERATURE", str(cfg.temperature))),
            )
        )

        try:
            resp = llm.invoke([system, HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = _json_loads_best_effort(text)
        except Exception:
            return _heuristic()

        strategy = data.get("strategy_suggestions")
        if not isinstance(strategy, dict):
            strategy = {}
        summary = data.get("summary")
        if not isinstance(summary, str):
            summary = None
        confidence = data.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None

        out = {
            "strategy_suggestions": strategy,
            "summary": summary,
            "confidence": confidence,
        }
        if not out["strategy_suggestions"] and not out["summary"]:
            return _heuristic()
        return out

    def suggest_page_strategy(
        self,
        *,
        page_summary: Dict[str, Any],
        pipeline_mode: str,
        target_language: str,
        skill_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return proactive page-level strategy suggestions.

        Used after page-level execution steps so the orchestrator can choose the
        next safe action and tune skill params for the current page.
        """

        def _heuristic() -> Dict[str, Any]:
            stage = str(page_summary.get("stage") or "").strip().lower()
            page_type = str(page_summary.get("page_type") or "content").strip().lower()
            embedded_ratio = float(page_summary.get("embedded_ratio") or 0.0)
            empty_tgt_regions = int(page_summary.get("empty_tgt_regions") or 0)
            overflow_regions = int(page_summary.get("overflow_regions") or 0)
            bad_render_regions = int(page_summary.get("bad_render_regions") or 0)
            has_erased_image = bool(page_summary.get("has_erased_image"))
            erase_backend = str(page_summary.get("erase_backend") or "").strip().lower()
            erase_mask_quality = str(page_summary.get("erase_mask_quality") or "").strip().lower()
            erase_mask_reason = str(page_summary.get("erase_mask_quality_reason") or "").strip()
            erase_mask_retry_hint = page_summary.get("erase_mask_retry_hint") or {}
            auto_revision_done = bool(page_summary.get("auto_revision_done"))
            vlm_quality_ok = page_summary.get("vlm_quality_ok")
            vlm_quality_reason = str(page_summary.get("vlm_quality_reason") or "").strip().lower()
            vlm_judge_decision = str(page_summary.get("vlm_judge_decision") or "").strip().lower()
            vlm_judge_reason = str(page_summary.get("vlm_judge_reason_code") or "").strip().lower()

            strategy: Dict[str, Any] = {}
            params: Dict[str, Dict[str, Any]] = {}
            summary_parts = []

            if page_type in {"cover", "info"}:
                strategy["page_profile"] = page_type
                strategy["recommended_page_action"] = "continue"
                return {
                    "strategy_suggestions": strategy,
                    "skill_param_suggestions": params,
                    "summary": f"non-content page ({page_type}); skip text pipeline",
                    "confidence": 0.95,
                }

            if embedded_ratio >= 0.5:
                strategy["page_profile"] = "art_integrated"
                params["lama_eraser"] = {"profile": "aggressive", "mask_dilate_px": 4}
                params["lettering"] = {"render_mode": "art_embed", "padding_ratio": 0.06, "stroke_width": 3}
                summary_parts.append("art-integrated page")
            else:
                strategy["page_profile"] = "boxed_dialogue"
                summary_parts.append("boxed-dialogue page")

            if stage == "translated":
                if vlm_judge_decision == "ask_user":
                    strategy["recommended_page_action"] = "ask_user"
                    strategy["ask_user_reason"] = "Page review is needed because the VLM judge flagged this page."
                    summary_parts.append("judge requested user review")
                elif vlm_judge_decision == "retry":
                    strategy["recommended_page_action"] = "retry_vlm_page"
                    summary_parts.append("judge requested a VLM retry")
                elif vlm_quality_ok is False:
                    retriable = {
                        "vlm_request_failed",
                        "bad_json",
                        "target_language_mismatch",
                        "judge_retry",
                    }
                    if vlm_quality_reason in retriable or vlm_quality_reason.startswith("vlm_zero_regions") or vlm_quality_reason.startswith("too_many_empty"):
                        strategy["recommended_page_action"] = "retry_vlm_page"
                        summary_parts.append("vlm output looks retriable")
                    else:
                        strategy["recommended_page_action"] = "ask_user"
                        strategy["ask_user_reason"] = (
                            f"Page review is needed because VLM quality is still not acceptable ({vlm_quality_reason or vlm_judge_reason or 'unknown reason'})."
                        )
                        summary_parts.append("vlm quality needs review")
                elif empty_tgt_regions > 0:
                    strategy["recommended_page_action"] = "retry_vlm_page" if pipeline_mode == "vlm" else "retry_translation"
                    summary_parts.append("page still has empty translated regions")

                strategy["preferred_post_translation_action"] = "erase_direct" if embedded_ratio >= 0.8 else "auto_revision_then_erase"
                if auto_revision_done:
                    summary_parts.append("translation already revised")
                elif strategy.get("preferred_post_translation_action") != "erase_direct":
                    strategy["recommended_page_action"] = strategy.get("recommended_page_action") or "run_revision"
                    summary_parts.append("recommend revision before erase")

            if stage == "typesetting" and (overflow_regions > 0 or bad_render_regions > 0):
                if erase_mask_quality in {"under_masked", "over_masked"} and has_erased_image:
                    strategy["recommended_page_action"] = "retry_erase"
                    strategy["preferred_rerun_action"] = "retry_erase"
                    params.setdefault("lama_eraser", {})
                    if isinstance(erase_mask_retry_hint, dict):
                        params["lama_eraser"].update(erase_mask_retry_hint)
                    params.setdefault("lettering", {})
                    params["lettering"].setdefault("render_mode", "art_embed" if embedded_ratio >= 0.5 else "bubble_fit")
                    params["lettering"].setdefault("padding_ratio", 0.05 if embedded_ratio >= 0.5 else 0.04)
                    summary_parts.append(f"typeset issue likely rooted in erase/mask quality ({erase_mask_reason or erase_mask_quality})")
                elif embedded_ratio >= 0.5 and has_erased_image:
                    strategy["recommended_page_action"] = "retry_erase"
                    strategy["preferred_rerun_action"] = "retry_erase"
                    params.setdefault("lama_eraser", {})
                    params["lama_eraser"].setdefault("profile", "aggressive")
                    params["lama_eraser"].setdefault("mask_dilate_px", 4)
                    params.setdefault("lettering", {})
                    params["lettering"].setdefault("render_mode", "art_embed")
                    params["lettering"].setdefault("padding_ratio", 0.05)
                    summary_parts.append("typeset issue likely rooted in erase/mask quality")
                else:
                    strategy["recommended_page_action"] = "retry_typeset"
                    strategy["preferred_rerun_action"] = "retry_typeset"
                    params.setdefault("lettering", {})
                    params["lettering"].setdefault("render_mode", "bubble_fit")
                    params["lettering"].setdefault("padding_ratio", 0.04)
                    params["lettering"].setdefault("line_spacing", 1.0)
                    summary_parts.append("typeset issue looks layout-driven")

            if stage == "erased" and erase_mask_quality in {"under_masked", "over_masked"} and has_erased_image:
                strategy["recommended_page_action"] = "retry_erase"
                strategy["preferred_rerun_action"] = "retry_erase"
                params.setdefault("lama_eraser", {})
                if isinstance(erase_mask_retry_hint, dict):
                    params["lama_eraser"].update(erase_mask_retry_hint)
                summary_parts.append(f"erase mask quality suggests retry ({erase_mask_reason or erase_mask_quality})")

            if stage == "erased" and erase_backend == "mock":
                summary_parts.append("using mock erase backend")

            summary = "; ".join(summary_parts) if summary_parts else "default page strategy"
            return {
                "strategy_suggestions": strategy,
                "skill_param_suggestions": params,
                "summary": summary,
                "confidence": 0.7,
            }

        cfg = self.config
        mode = (os.getenv("MANGA_AGENT_BRAIN_MODE", "auto") or "auto").lower().strip()
        if (not cfg.enabled) or mode == "off":
            return _heuristic()

        if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
            return _heuristic()

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from orchestrator.llm_adapter import LLMConfig, build_chat_llm
        except Exception:
            return _heuristic()

        system = SystemMessage(
            content=(
                "You are the page-strategy brain of a comic translation agent. "
                "You receive a structured page_summary describing the current page state, quality signals, and recent execution outputs. "
                "Recommend SAFE, minimal strategy updates for the next step. "
                "Do not invent observations not present in page_summary. "
                "Use page_profile to distinguish art-integrated pages from boxed-dialogue pages when strongly supported. "
                "Use preferred_post_translation_action to choose between auto_revision_then_erase and erase_direct. "
                "Use preferred_rerun_action only when the page is already in a rerun-worthy state such as bad typesetting. "
                "You may also propose skill_param_suggestions, but only for supported skills/params from skill_context. "
                "Return STRICT JSON only (no markdown)."
            )
        )

        payload = {
            "task": "suggest_page_strategy",
            "page_summary": page_summary,
            "context": {
                "pipeline_mode": pipeline_mode,
                "target_language": target_language,
            },
            "skill_context": skill_context or {},
            "output_schema": {
                "strategy_suggestions": {
                    "recommended_page_action": "continue|ask_user|retry_typeset|retry_erase|retry_vlm_page|retry_translation|retry_ocr|retry_detection|run_revision",
                    "ask_user_reason": "string|null",
                    "preferred_rerun_action": "retry_typeset|retry_erase|retry_vlm_page|retry_translation|retry_ocr|retry_detection|run_revision",
                    "preferred_correction_action": "run_revision|retry_vlm_page|retry_translation|retry_ocr|retry_detection",
                    "preferred_post_translation_action": "auto_revision_then_erase|erase_direct",
                    "revision_scope": "page_only|global",
                    "revision_page_limit": "int_1_to_24",
                    "reuse_erased_image": "bool",
                    "page_profile": "default|art_integrated|boxed_dialogue"
                },
                "skill_param_suggestions": {
                    "lama_eraser": {
                        "profile": "enum?",
                        "mask_dilate_px": "int?",
                        "mask_min_coverage": "float?",
                        "mask_max_coverage": "float?",
                        "embedded_box_expand_px": "int?",
                        "embedded_complete_mask_close_px": "int?"
                    },
                    "lettering": {
                        "render_mode": "enum?",
                        "padding_ratio": "float?",
                        "line_spacing": "float?",
                        "stroke_width": "int?",
                        "max_font_size_ratio": "float?"
                    }
                },
                "summary": "string",
                "confidence": "float_0_to_1|null"
            },
        }

        llm = build_chat_llm(
            LLMConfig(
                model=os.getenv("MANGA_AGENT_BRAIN_MODEL", cfg.model),
                temperature=float(os.getenv("MANGA_AGENT_BRAIN_TEMPERATURE", str(cfg.temperature))),
            )
        )

        try:
            resp = llm.invoke([system, HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = _json_loads_best_effort(text)
        except Exception:
            return _heuristic()

        strategy = data.get("strategy_suggestions")
        if not isinstance(strategy, dict):
            strategy = {}
        skill_params = data.get("skill_param_suggestions")
        if not isinstance(skill_params, dict):
            skill_params = {}
        summary = data.get("summary")
        if not isinstance(summary, str):
            summary = None
        confidence = data.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None

        out = {
            "strategy_suggestions": strategy,
            "skill_param_suggestions": skill_params,
            "summary": summary,
            "confidence": confidence,
        }
        if not out["strategy_suggestions"] and not out["skill_param_suggestions"] and not out["summary"]:
            return _heuristic()
        return out

    def plan_user_message(
        self,
        *,
        user_message: str,
        page_count: int,
        pipeline_mode: str,
        target_language: str,
        brain_context: Optional[Dict[str, Any]] = None,
        skill_context: Optional[Dict[str, Any]] = None,
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

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from orchestrator.llm_adapter import LLMConfig, build_chat_llm
        except Exception:
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

        system = SystemMessage(
            content=(
                "You are the decision brain of a comic translation agent. "
            "You interpret user feedback and produce a STRICT JSON plan for the orchestrator. "
            "Do NOT ask the user whether 'xxx is source or target'; infer it yourself. "
            "Only ask clarification when required: reference example (page number) and scope (page-only vs global). "
            "You may receive brain_context summarizing prior user preferences, recent corrections, relevant terminology, and project rules. "
            "Use it to stay consistent with past decisions, but prioritize the current user message if there is any conflict. "
            "You may also receive skill_context describing supported skill parameters. "
            "If the user seems to want quality/style changes, you may propose skill_param_suggestions, but only for supported skills/params from skill_context. "
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
            "brain_context": brain_context or {},
            "skill_context": skill_context or {},
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
                "strategy_suggestions": {
                    "preferred_rerun_action": "retry_typeset|retry_erase|retry_vlm_page|retry_translation|retry_ocr|retry_detection|run_revision",
                    "preferred_correction_action": "run_revision|retry_vlm_page|retry_translation|retry_ocr|retry_detection",
                    "preferred_post_translation_action": "auto_revision_then_erase|erase_direct",
                    "revision_scope": "page_only|global",
                    "revision_page_limit": "int_1_to_24",
                    "reuse_erased_image": "bool",
                    "page_profile": "default|art_integrated|boxed_dialogue"
                },
                "skill_param_suggestions": {
                    "lama_eraser": {
                        "profile": "enum?",
                        "mask_dilate_px": "int?",
                        "mask_min_coverage": "float?",
                        "mask_max_coverage": "float?",
                        "embedded_box_expand_px": "int?",
                        "embedded_complete_mask_close_px": "int?"
                    },
                    "lettering": {
                        "render_mode": "enum?",
                        "padding_ratio": "float?",
                        "line_spacing": "float?",
                        "stroke_width": "int?",
                        "max_font_size_ratio": "float?"
                    }
                },
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
        strategy_suggestions = data.get("strategy_suggestions")
        if not isinstance(strategy_suggestions, dict):
            strategy_suggestions = {}
        skill_param_suggestions = data.get("skill_param_suggestions")
        if not isinstance(skill_param_suggestions, dict):
            skill_param_suggestions = {}

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
            "strategy_suggestions": strategy_suggestions,
            "skill_param_suggestions": skill_param_suggestions,
            "confidence": conf,
        }
