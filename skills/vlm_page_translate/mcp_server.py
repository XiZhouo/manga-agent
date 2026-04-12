from __future__ import annotations

from typing import Any, Dict, Optional

from fastmcp import FastMCP

from skills.vlm_page_translate.vlm import VLMPageTranslateConfig, vlm_extract_translate_page, vlm_extract_translate_pages

mcp = FastMCP("VLMPageTranslate")


@mcp.tool()
def translate_page(
    page_index: int,
    image_bytes: bytes,
    reading_direction: str = "ltr",
    source_language: str = "en",
    target_language: str = "zh",
    retrieval_context: Optional[Dict[str, Any]] = None,
    model: str = "qwen3.6-plus",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature: float = 0.0,
    max_width: int = 0,
    max_height: int = 3057,
    max_long_side: int = 0,
    timeout_seconds: Optional[float] = None,
):
    """MCP tool: VLM core (detect+OCR+translate) for a single manga page."""

    cfg = VLMPageTranslateConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_width=max_width,
        max_height=max_height,
        max_long_side=max_long_side,
        timeout_seconds=timeout_seconds,
    )

    return vlm_extract_translate_page(
        page_index=page_index,
        image_bytes=image_bytes,
        reading_direction=reading_direction,
        source_language=source_language,
        target_language=target_language,
        retrieval_context=retrieval_context,
        config=cfg,
    )


@mcp.tool()
def translate_page_batch(
    pages: list[dict],
    reading_direction: str = "ltr",
    source_language: str = "en",
    target_language: str = "zh",
    model: str = "qwen3.6-plus",
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature: float = 0.0,
    max_width: int = 0,
    max_height: int = 3057,
    max_long_side: int = 0,
    timeout_seconds: Optional[float] = None,
):
    """MCP tool: VLM core (detect+OCR+translate) for a small page batch."""

    cfg = VLMPageTranslateConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_width=max_width,
        max_height=max_height,
        max_long_side=max_long_side,
        timeout_seconds=timeout_seconds,
    )

    return vlm_extract_translate_pages(
        pages=pages,
        reading_direction=reading_direction,
        source_language=source_language,
        target_language=target_language,
        config=cfg,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
