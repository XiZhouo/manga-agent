from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from skills.vlm_translation_revision.revise import VLMTranslationRevisionConfig, revise_vlm_translation

mcp = FastMCP("VLMTranslationRevision")


@mcp.tool()
def revise(
    pages: List[Dict[str, Any]],
    user_request: str,
    document_range: Optional[Dict[str, Any]] = None,
    reference_page_number: Optional[int] = None,
    scope: str = "page_only",
    apply_page_indices: Optional[List[int]] = None,
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
):
    """MCP tool: revise VLM translations (text-only) with a user correction request."""

    cfg = VLMTranslationRevisionConfig(
        model=model,
        temperature=temperature,
        max_candidate_pages=max_candidate_pages,
        max_patches=max_patches,
    )

    return revise_vlm_translation(
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
        config=cfg,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
