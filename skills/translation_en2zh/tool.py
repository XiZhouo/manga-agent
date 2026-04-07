from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from skills.translation_en2zh.translate import GlossaryEntry, translate_utterances_en2zh


@tool
def translate_script_en2zh(
    utterances: List[Dict[str, Any]],
    page_context: str = "",
    glossary: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """Translate a page script from English to Chinese.

    - `utterances` items should include: region_id, clean_text
    - `glossary` entries are optional; locked entries are enforced by instruction.
    """

    glossary_entries = [
        GlossaryEntry(
            entry_id=e.get("entry_id", ""),
            type=e.get("type", "term"),
            src_text=e.get("src_text", ""),
            tgt_text=e.get("tgt_text", ""),
            locked=bool(e.get("locked", False)),
            aliases_src=e.get("aliases_src"),
            aliases_tgt=e.get("aliases_tgt"),
        )
        for e in (glossary or [])
        if isinstance(e, dict)
    ]

    return translate_utterances_en2zh(
        utterances=utterances,
        page_context=page_context,
        glossary=glossary_entries,
    )
