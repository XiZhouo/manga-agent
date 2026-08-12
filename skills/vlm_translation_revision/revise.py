from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# NOTE: LangChain dependencies are optional for offline/stub runs.
# We import them lazily only when the revision LLM is actually invoked.


@dataclass(frozen=True)
class VLMTranslationRevisionConfig:
    model: str = "qwen3-max"
    temperature: float = 0.0

    max_candidate_pages: int = 12
    max_patches: int = 64


def _json_loads_best_effort(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())


def _extract_from_to_heuristic(user_request: str) -> Tuple[Optional[str], Optional[str]]:
    """Heuristic extraction for patterns like 'xxx应该翻译成yyy' or 'xxx改为yyy'."""

    msg = _norm(user_request)
    markers = ["应该翻译成", "应翻译成", "应该译为", "应译为", "改成", "改为", "替换为"]
    for m in markers:
        if m in msg:
            left, right = msg.split(m, 1)
            from_text = left.split("的")[-1].strip()
            to_text = right.strip()

            for cut in ["，", ",", "。", ".", "；", ";", "请", "并", "然后"]:
                if cut in to_text:
                    to_text = to_text.split(cut, 1)[0].strip()
            for cut in ["，", ",", "。", ".", "；", ";"]:
                if cut in from_text:
                    from_text = from_text.split(cut, 1)[-1].strip()

            return (from_text or None, to_text or None)

    return (None, None)


def _resolve_reference_page_index(pages: List[Dict[str, Any]], reference_page_number: Optional[int]) -> Optional[int]:
    if reference_page_number is None:
        return None

    page_indices = {int(p.get("page_index")) for p in pages if str(p.get("page_index")).strip().isdigit()}

    if reference_page_number == 0 and 0 in page_indices:
        return 0
    if (reference_page_number - 1) in page_indices:
        return reference_page_number - 1
    if reference_page_number in page_indices:
        return reference_page_number

    return None


def _page_has_query(page: Dict[str, Any], *, query: str, from_side: str) -> bool:
    q = _norm(query)
    if not q:
        return False

    for r in page.get("regions", []) or []:
        if not isinstance(r, dict):
            continue
        src = _norm(str(r.get("src_text") or ""))
        tgt = _norm(str(r.get("tgt_text") or ""))

        if from_side == "src" and q and q in src:
            return True
        if from_side == "tgt" and q and q in tgt:
            return True
        if from_side == "auto" and (q in src or q in tgt):
            return True

    return False


def _extract_usage(resp: Any) -> Optional[Dict[str, Any]]:
    try:
        meta = getattr(resp, "response_metadata", None) or {}
        usage = meta.get("token_usage") or meta.get("usage")
        if isinstance(usage, dict):
            return usage
    except Exception:
        return None
    return None


def _apply_patches_to_pages(
    *,
    pages: List[Dict[str, Any]],
    document_range: Optional[Dict[str, Any]],
    patches: List[Dict[str, Any]],
    glossary_updates: List[Dict[str, Any]],
    warnings: List[str],
    decision: str,
    ask_user: Optional[str],
    candidate_pages_count: int,
    truncated: bool,
    usage_out: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    revised_pages = json.loads(json.dumps(pages))
    by_pi = {int(p.get("page_index")): p for p in revised_pages if str(p.get("page_index")).strip().isdigit()}

    changed_pages: set[int] = set()
    applied = 0
    for patch in patches:
        pi = patch["page_index"]
        rid = patch["region_id"]
        page = by_pi.get(pi)
        if not page:
            continue
        regions = page.get("regions") or []
        if not isinstance(regions, list):
            continue
        for r in regions:
            if not isinstance(r, dict):
                continue
            try:
                if int(r.get("region_id")) != rid:
                    continue
            except Exception:
                continue
            old_tgt = str(r.get("tgt_text") or "")
            exp = patch.get("expected_old_tgt")
            if isinstance(exp, str) and exp.strip() and (old_tgt.strip() != exp.strip()):
                continue
            if old_tgt != patch["new_tgt"]:
                r["tgt_text"] = patch["new_tgt"]
                applied += 1
                changed_pages.add(pi)
            break

    return {
        "ok": True,
        "data": {
            "decision": decision,
            "ask_user": ask_user,
            "document_range": document_range,
            "pages": revised_pages,
            "changed_pages": sorted(changed_pages),
            "patches": patches,
            "glossary_updates": [x for x in glossary_updates if isinstance(x, dict)],
            "warnings": warnings,
        },
        "metrics": {
            "candidate_pages": candidate_pages_count,
            "truncated": truncated,
            "patches": len(patches),
            "applied": applied,
            "usage": usage_out,
        },
        "error": None,
    }


def _load_revision_runtime(config: VLMTranslationRevisionConfig) -> Tuple[Any, Any, Any, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage
    from orchestrator.llm_adapter import LLMConfig, build_chat_llm

    llm = build_chat_llm(
        LLMConfig(
            model=os.getenv("MANGA_AGENT_REVISION_MODEL", config.model),
            temperature=float(os.getenv("MANGA_AGENT_REVISION_TEMPERATURE", str(config.temperature))),
        )
    )
    return HumanMessage, SystemMessage, LLMConfig, llm


def _validate_patches(raw_patches: Any, max_patches: int) -> List[Dict[str, Any]]:
    patches_in = raw_patches or []
    if not isinstance(patches_in, list):
        patches_in = []

    patches: List[Dict[str, Any]] = []
    for p in patches_in[:max_patches]:
        if not isinstance(p, dict):
            continue
        try:
            pi = int(p.get("page_index"))
            rid = int(p.get("region_id"))
        except Exception:
            continue
        new_tgt = p.get("new_tgt")
        if not isinstance(new_tgt, str) or not new_tgt.strip():
            continue

        expected_old = p.get("expected_old_tgt")
        if expected_old is not None and not isinstance(expected_old, str):
            expected_old = None

        patches.append(
            {
                "page_index": pi,
                "region_id": rid,
                "expected_old_tgt": expected_old,
                "new_tgt": new_tgt.strip(),
                "confidence": p.get("confidence"),
                "why": p.get("why"),
            }
        )
    return patches


def _run_auto_rag_revision(
    *,
    pages: List[Dict[str, Any]],
    document_range: Optional[Dict[str, Any]],
    user_request: str,
    reading_direction: str,
    target_language: str,
    glossary: Optional[Dict[str, Any]],
    retrieval_context: Optional[Dict[str, Any]],
    config: VLMTranslationRevisionConfig,
) -> Dict[str, Any]:
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        return _apply_patches_to_pages(
            pages=pages,
            document_range=document_range,
            patches=[],
            glossary_updates=[],
            warnings=["missing_api_key_auto_revision_skipped"],
            decision="apply",
            ask_user=None,
            candidate_pages_count=len(pages),
            truncated=False,
            usage_out=None,
        )

    try:
        HumanMessage, SystemMessage, _, llm = _load_revision_runtime(config)
    except Exception as e:
        return _apply_patches_to_pages(
            pages=pages,
            document_range=document_range,
            patches=[],
            glossary_updates=[],
            warnings=[f"auto_revision_runtime_unavailable: {e}"],
            decision="apply",
            ask_user=None,
            candidate_pages_count=len(pages),
            truncated=False,
            usage_out=None,
        )

    glossary_entries = []
    if isinstance(glossary, dict):
        ge = glossary.get("entries")
        if isinstance(ge, list):
            glossary_entries = [x for x in ge if isinstance(x, dict)][:50]

    system = SystemMessage(
        content=(
            "You are a comic translation self-revision engine (text-only). "
            "You receive current page translations plus a retrieval_context that contains glossary hits, translation memory hits, style hints, and project rules. "
            "Your job is to apply a minimal consistency pass. Only patch tgt_text when retrieval_context gives strong evidence. "
            "Prioritize locked glossary and locked translation memory entries over everything else. "
            "Do not rewrite the whole page. Do not change region_id, bbox_xyxy, src_text, or ordering. "
            "If nothing clearly needs fixing, return an empty patches list. "
            "Return STRICT JSON only (no markdown)."
        )
    )

    payload = {
        "task": "vlm_translation_auto_rag_revision",
        "document_range": document_range,
        "user_request": user_request,
        "target_language": target_language,
        "reading_direction": reading_direction,
        "pages": pages,
        "glossary_entries": glossary_entries,
        "retrieval_context": retrieval_context or {},
        "output_schema": {
            "decision": "apply",
            "patches": [
                {
                    "page_index": "int",
                    "region_id": "int",
                    "expected_old_tgt": "string|null",
                    "new_tgt": "string",
                    "confidence": "float_0_to_1|null",
                    "why": "string|null",
                }
            ],
            "glossary_updates": [
                {
                    "type": "term|quote|name|phrase",
                    "src_text": "string",
                    "tgt_text": "string",
                    "locked": "bool",
                    "aliases_src": "string[]",
                    "aliases_tgt": "string[]",
                }
            ],
            "warnings": "string[]",
        },
    }

    resp = llm.invoke([system, HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        out = _json_loads_best_effort(text)
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "metrics": {"candidate_pages": len(pages), "truncated": False},
            "error": {"code": "bad_json", "message": str(e), "detail": text},
        }

    patches = _validate_patches(out.get("patches"), config.max_patches)
    glossary_updates = out.get("glossary_updates") or []
    if not isinstance(glossary_updates, list):
        glossary_updates = []
    warnings = out.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = []
    warnings = [str(w) for w in warnings if str(w).strip()]

    return _apply_patches_to_pages(
        pages=pages,
        document_range=document_range,
        patches=patches,
        glossary_updates=[x for x in glossary_updates if isinstance(x, dict)],
        warnings=warnings,
        decision="apply",
        ask_user=None,
        candidate_pages_count=len(pages),
        truncated=False,
        usage_out=_extract_usage(resp),
    )


def _run_user_feedback_revision(
    *,
    pages: List[Dict[str, Any]],
    document_range: Optional[Dict[str, Any]],
    user_request: str,
    reference_page_number: Optional[int],
    scope: str,
    apply_page_indices: Optional[List[int]],
    from_side: str,
    from_text: Optional[str],
    to_text: Optional[str],
    reading_direction: str,
    target_language: str,
    glossary: Optional[Dict[str, Any]],
    retrieval_context: Optional[Dict[str, Any]],
    config: VLMTranslationRevisionConfig,
) -> Dict[str, Any]:
    if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        return {
            "ok": False,
            "data": None,
            "metrics": {},
            "error": {
                "code": "missing_api_key",
                "message": "Missing DASHSCOPE_API_KEY (preferred) or OPENAI_API_KEY (fallback).",
            },
        }

    try:
        HumanMessage, SystemMessage, _, llm = _load_revision_runtime(config)
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "metrics": {},
            "error": {
                "code": "missing_dependency",
                "message": f"Missing revision dependencies (langchain). Install requirements to enable revision. Detail: {e}",
            },
        }

    scope_norm = (scope or "page_only").lower().strip()
    if scope_norm not in {"page_only", "global"}:
        scope_norm = "page_only"

    from_side_norm = (from_side or "auto").lower().strip()
    if from_side_norm not in {"src", "tgt", "auto"}:
        from_side_norm = "auto"

    ref_index = _resolve_reference_page_index(pages, reference_page_number)
    if ref_index is None:
        m = re.search(r"第\s*(\d+)\s*页", user_request or "")
        if m:
            try:
                ref_index = _resolve_reference_page_index(pages, int(m.group(1)))
            except Exception:
                ref_index = None

    if ref_index is None:
        return {
            "ok": True,
            "data": {
                "decision": "ask_user",
                "ask_user": "请提供一个参照示例页码（例如：第12页），我才能对照判断并批量修正。",
                "document_range": document_range,
                "pages": pages,
                "changed_pages": [],
                "glossary_updates": [],
                "patches": [],
                "warnings": ["missing_reference_page"],
            },
            "metrics": {"candidate_pages": 0, "patches": 0},
            "error": None,
        }

    from_text_norm = _norm(from_text or "")
    to_text_norm = _norm(to_text or "")
    if (not from_text_norm) or (not to_text_norm):
        f2, t2 = _extract_from_to_heuristic(user_request)
        if not from_text_norm:
            from_text_norm = _norm(f2 or "")
        if not to_text_norm:
            to_text_norm = _norm(t2 or "")

    if (not from_text_norm) or (not to_text_norm):
        return {
            "ok": True,
            "data": {
                "decision": "ask_user",
                "ask_user": "我理解你想纠正翻译，但缺少明确的‘原说法/错误译法’和‘期望译法’。请用‘X 应该翻译成 Y’的形式再发一次。",
                "document_range": document_range,
                "pages": pages,
                "changed_pages": [],
                "glossary_updates": [],
                "patches": [],
                "warnings": ["missing_from_to"],
            },
            "metrics": {"candidate_pages": 0, "patches": 0},
            "error": None,
        }

    pages_by_index = {int(p.get("page_index")): p for p in pages if str(p.get("page_index")).strip().isdigit()}
    ref_page = pages_by_index.get(ref_index)
    if not ref_page:
        return {
            "ok": True,
            "data": {
                "decision": "ask_user",
                "ask_user": "参照页不存在（可能页码超出范围）。请提供有效页码。",
                "document_range": document_range,
                "pages": pages,
                "changed_pages": [],
                "glossary_updates": [],
                "patches": [],
                "warnings": ["reference_page_not_found"],
            },
            "metrics": {"candidate_pages": 0, "patches": 0},
            "error": None,
        }

    candidate_page_indices: List[int] = [ref_index]
    if scope_norm == "global":
        for pi in sorted(pages_by_index.keys()):
            if pi == ref_index:
                continue
            if _page_has_query(pages_by_index[pi], query=from_text_norm, from_side=from_side_norm):
                candidate_page_indices.append(pi)

    candidate_page_indices = sorted(set(candidate_page_indices))
    truncated = False
    if len(candidate_page_indices) > config.max_candidate_pages:
        truncated = True
        rest = [pi for pi in candidate_page_indices if pi != ref_index]
        candidate_page_indices = [ref_index] + rest[: config.max_candidate_pages - 1]

    candidate_pages = [pages_by_index[pi] for pi in candidate_page_indices if pi in pages_by_index]

    glossary_entries = []
    if isinstance(glossary, dict):
        ge = glossary.get("entries")
        if isinstance(ge, list):
            glossary_entries = [x for x in ge if isinstance(x, dict)][:50]

    system = SystemMessage(
        content=(
            "You are a comic translation correction engine (text-only). "
            "You receive one REFERENCE page (with full region list) and several CANDIDATE pages (full region lists). "
            "The user described a translation correction. "
            "Your job: decide which regions across candidate pages correspond to the same translation issue, and propose minimal patches that update tgt_text. "
            "You may use pages outside the allowed patch set only as context. "
            "You MUST propose patches only for allowed_patch_page_indices. "
            "You MUST NOT change region_id or bbox_xyxy. Only change tgt_text. "
            "If uncertain, choose decision=ask_user and ask one short question. "
            "Return STRICT JSON only (no markdown)."
        )
    )

    payload = {
        "task": "vlm_translation_revision",
        "document_range": document_range,
        "user_request": user_request,
        "reference_page_number": reference_page_number,
        "reference_page_index": ref_index,
        "scope": scope_norm,
        "allowed_patch_page_indices": [int(x) for x in (apply_page_indices or []) if isinstance(x, int)],
        "from_side": from_side_norm,
        "from_text": from_text_norm,
        "to_text": to_text_norm,
        "target_language": target_language,
        "reading_direction": reading_direction,
        "glossary_entries": glossary_entries,
        "retrieval_context": retrieval_context or {},
        "reference_page": {"page_index": ref_index, "regions": ref_page.get("regions") or []},
        "candidate_pages": [{"page_index": int(p.get("page_index")), "regions": p.get("regions") or []} for p in candidate_pages],
        "output_schema": {
            "decision": "apply|ask_user",
            "ask_user": "string|null",
            "patches": [
                {
                    "page_index": "int",
                    "region_id": "int",
                    "expected_old_tgt": "string|null",
                    "new_tgt": "string",
                    "confidence": "float_0_to_1|null",
                    "why": "string|null",
                }
            ],
            "glossary_updates": [
                {
                    "type": "term|quote|name|phrase",
                    "src_text": "string",
                    "tgt_text": "string",
                    "locked": "bool",
                    "aliases_src": "string[]",
                    "aliases_tgt": "string[]",
                }
            ],
            "warnings": "string[]",
        },
    }

    resp = llm.invoke([system, HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        out = _json_loads_best_effort(text)
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "metrics": {"candidate_pages": len(candidate_pages), "truncated": truncated},
            "error": {"code": "bad_json", "message": str(e), "detail": text},
        }

    decision = str(out.get("decision") or "apply").strip().lower()
    if decision not in {"apply", "ask_user"}:
        decision = "apply"

    ask_user = out.get("ask_user")
    if isinstance(ask_user, str):
        ask_user = ask_user.strip() or None
    else:
        ask_user = None

    patches = _validate_patches(out.get("patches"), config.max_patches)
    allowed_apply = {int(x) for x in (apply_page_indices or []) if isinstance(x, int)}
    if allowed_apply:
        patches = [x for x in patches if int(x.get("page_index")) in allowed_apply]
    glossary_updates = out.get("glossary_updates") or []
    if not isinstance(glossary_updates, list):
        glossary_updates = []
    warnings = out.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = []
    warnings = [str(w) for w in warnings if str(w).strip()]
    if truncated:
        warnings.append(f"candidate_pages_truncated_to_{config.max_candidate_pages}")

    return _apply_patches_to_pages(
        pages=pages,
        document_range=document_range,
        patches=patches,
        glossary_updates=[x for x in glossary_updates if isinstance(x, dict)],
        warnings=warnings,
        decision=decision,
        ask_user=ask_user,
        candidate_pages_count=len(candidate_pages),
        truncated=truncated,
        usage_out=_extract_usage(resp),
    )


def revise_vlm_translation(
    *,
    pages: List[Dict[str, Any]],
    document_range: Optional[Dict[str, Any]] = None,
    revision_mode: str = "user_feedback_revision",
    user_request: str,
    reference_page_number: Optional[int],
    scope: str = "page_only",
    apply_page_indices: Optional[List[int]] = None,
    from_side: str = "auto",
    from_text: Optional[str] = None,
    to_text: Optional[str] = None,
    reading_direction: str = "ltr",
    target_language: str = "zh",
    glossary: Optional[Dict[str, Any]] = None,
    retrieval_context: Optional[Dict[str, Any]] = None,
    config: Optional[VLMTranslationRevisionConfig] = None,
) -> Dict[str, Any]:
    """Revise VLM translations using text-only inputs.

    Supported modes:
    - user_feedback_revision: explicit user correction, may ask for clarification.
    - auto_rag_revision: pipeline self-correction using retrieval_context.
    """

    cfg = config or VLMTranslationRevisionConfig()
    mode = str(revision_mode or "user_feedback_revision").strip().lower()

    if mode == "auto_rag_revision":
        return _run_auto_rag_revision(
            pages=pages,
            document_range=document_range,
            user_request=user_request,
            reading_direction=reading_direction,
            target_language=target_language,
            glossary=glossary,
            retrieval_context=retrieval_context,
            config=cfg,
        )

    return _run_user_feedback_revision(
        pages=pages,
        document_range=document_range,
        user_request=user_request,
        reference_page_number=reference_page_number,
        scope=scope,
        apply_page_indices=apply_page_indices,
        from_side=from_side,
        from_text=from_text,
        to_text=to_text,
        reading_direction=reading_direction,
        target_language=target_language,
        glossary=glossary,
        retrieval_context=retrieval_context,
        config=cfg,
    )
