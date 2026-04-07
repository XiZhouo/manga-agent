from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from orchestrator.llm_adapter import LLMConfig, build_chat_llm


@dataclass(frozen=True)
class GlossaryEntry:
    entry_id: str
    src_text: str
    tgt_text: str
    locked: bool = False
    aliases_src: Optional[List[str]] = None
    aliases_tgt: Optional[List[str]] = None
    type: str = "term"  # term|quote|name|phrase


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def build_glossary_context(utterances: List[Dict[str, Any]], glossary: List[GlossaryEntry]) -> List[Dict[str, Any]]:
    """Pick glossary entries relevant to current utterances (v0: exact/substring match)."""

    if not glossary:
        return []

    texts = [_normalize(u.get("clean_text") or u.get("src_text") or "") for u in utterances]
    out: List[Dict[str, Any]] = []

    for entry in glossary:
        keys = [_normalize(entry.src_text)]
        if entry.aliases_src:
            keys.extend(_normalize(x) for x in entry.aliases_src)

        if any(k and any(k in t for t in texts) for k in keys):
            out.append(
                {
                    "entry_id": entry.entry_id,
                    "type": entry.type,
                    "src_text": entry.src_text,
                    "tgt_text": entry.tgt_text,
                    "locked": bool(entry.locked),
                    "aliases_src": entry.aliases_src or [],
                    "aliases_tgt": entry.aliases_tgt or [],
                }
            )

    # stable ordering: locked first
    out.sort(key=lambda x: (not x.get("locked", False), x.get("type", ""), x.get("entry_id", "")))
    return out


def translate_utterances_en2zh(
    *,
    utterances: List[Dict[str, Any]],
    page_context: str = "",
    glossary: Optional[List[GlossaryEntry]] = None,
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, Any]:
    """Translate a page's utterances EN->ZH.

    Contract (v0):
    - Input utterances must include `region_id` and `clean_text`.
    - Output is a dict:
      {
        ok: bool,
        results: {region_id: {translated_text: str}},
        warnings: [...]
      }

    Notes:
    - This function uses a single LLM call for the whole page.
    - Glossary is enforced by instruction; `locked=true` entries should be followed.
    """

    glossary_ctx = build_glossary_context(utterances, glossary or [])

    system = SystemMessage(
        content=(
            "You are a translation engine for English->Simplified Chinese. "
            "Return STRICT JSON only (no markdown, no code fences). "
            "Follow glossary constraints: if an entry is locked=true, you MUST use its tgt_text when the src_text (or aliases_src) matches. "
            "Keep punctuation natural for Chinese."
        )
    )

    payload = {
        "task": "translate_utterances_en2zh",
        "page_context": page_context,
        "glossary": glossary_ctx,
        "utterances": [
            {
                "region_id": u.get("region_id"),
                "text": u.get("clean_text") or "",
                "speaker_id": u.get("speaker_id"),
                "utterance_type": u.get("utterance_type", "unknown"),
            }
            for u in utterances
        ],
        "output_schema": {
            "results": {"<region_id:int>": {"translated_text": "<str>"}},
            "warnings": ["<str>"]
        },
    }

    human = HumanMessage(content=json.dumps(payload, ensure_ascii=False))

    llm = build_chat_llm(llm_config)
    resp = llm.invoke([system, human])

    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # best-effort extraction
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            return {"ok": False, "error": {"code": "bad_json", "message": "LLM did not return JSON", "detail": text}}

    results = data.get("results") or {}
    warnings = data.get("warnings") or []

    # normalize keys to int-like strings
    normalized: Dict[int, Dict[str, Any]] = {}
    for k, v in results.items():
        try:
            region_id = int(k)
        except Exception:
            continue
        translated_text = (v or {}).get("translated_text")
        if isinstance(translated_text, str):
            normalized[region_id] = {"translated_text": translated_text}

    return {"ok": True, "results": normalized, "warnings": warnings}
