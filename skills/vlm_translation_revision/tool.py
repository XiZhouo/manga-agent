from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from skills.vlm_translation_revision.revise import VLMTranslationRevisionConfig, revise_vlm_translation


@tool
def vlm_translation_revision(
    pages: List[Dict[str, Any]],
    user_request: str,
    reference_page_number: Optional[int] = None,
    scope: str = "page_only",
    from_side: str = "auto",
    from_text: Optional[str] = None,
    to_text: Optional[str] = None,
    reading_direction: str = "ltr",
    target_language: str = "zh",
    glossary: Optional[Dict[str, Any]] = None,
    model: str = "qwen3-max",
    temperature: float = 0.0,
    max_candidate_pages: int = 12,
    max_patches: int = 64,
) -> Dict[str, Any]:
    """Revise VLM translations using text-only correction instructions.

    Notes:
    - Does NOT call VLM; uses a text LLM to plan patches.
    - Input/output page schema matches VLM-style {page_index, regions[{region_id,bbox_xywh,src_text,tgt_text}]}.
    - Reads API key from env: `DASHSCOPE_API_KEY` (preferred) or `OPENAI_API_KEY` (fallback).
    """

    cfg = VLMTranslationRevisionConfig(
        model=model,
        temperature=temperature,
        max_candidate_pages=max_candidate_pages,
        max_patches=max_patches,
    )

    return revise_vlm_translation(
        pages=pages,
        user_request=user_request,
        reference_page_number=reference_page_number,
        scope=scope,
        from_side=from_side,
        from_text=from_text,
        to_text=to_text,
        reading_direction=reading_direction,
        target_language=target_language,
        glossary=glossary,
        config=cfg,
    )
