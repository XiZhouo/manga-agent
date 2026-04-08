from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.tools import tool

from skills.vlm_page_translate.vlm import VLMPageTranslateConfig, vlm_extract_translate_page


@tool
def vlm_page_translate(
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
) -> dict:
    """Detect + OCR + translate a manga page using a multimodal VLM (OpenAI-compatible).

    Intended model: DashScope `qwen3.6-plus`.

    Notes:
    - Reads API key from env: `DASHSCOPE_API_KEY` (preferred) or `OPENAI_API_KEY` (fallback).
    - Returns STRICT JSON envelope.
    """

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
