from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class LLMConfig:
    """LLM config (OpenAI-compatible).

    Defaults are aligned with `test.py` (DashScope compatible-mode + Qwen).
    """

    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3-max"
    temperature: float = 0.2

    # DashScope docs recommend DASHSCOPE_API_KEY; many OpenAI-compatible setups use OPENAI_API_KEY.
    api_key_env_primary: str = "DASHSCOPE_API_KEY"
    api_key_env_fallback: str = "OPENAI_API_KEY"


def build_chat_llm(config: LLMConfig | None = None) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI client configured for DashScope/Qwen.

    This follows the same configuration style as `test.py`:
    - OpenAI-compatible `base_url`
    - Model name like `qwen3-max`
    - API key is read from environment by default
    """

    cfg = config or LLMConfig()
    api_key = os.getenv(cfg.api_key_env_primary) or os.getenv(cfg.api_key_env_fallback)

    return ChatOpenAI(
        model=cfg.model,
        base_url=cfg.base_url,
        api_key=api_key,
        temperature=cfg.temperature,
    )
