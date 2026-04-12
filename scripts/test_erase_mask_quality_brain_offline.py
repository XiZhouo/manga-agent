from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.brain import DecisionBrain


def main() -> None:
    brain = DecisionBrain()
    result = brain.suggest_page_strategy(
        page_summary={
            "page_index": 3,
            "stage": "typesetting",
            "embedded_ratio": 1.0,
            "empty_tgt_regions": 0,
            "overflow_regions": 1,
            "bad_render_regions": 1,
            "has_erased_image": True,
            "erase_backend": "lama:saicinpainting_cli",
            "erase_mask_quality": "under_masked",
            "erase_mask_quality_reason": "mask likely misses text strokes or edge pixels",
            "erase_mask_retry_hint": {
                "profile": "aggressive",
                "mask_dilate_px": 5,
                "embedded_box_expand_px": 8,
                "embedded_complete_mask_close_px": 9,
            },
            "auto_revision_done": True,
            "vlm_quality_ok": True,
            "vlm_quality_reason": None,
            "vlm_judge_decision": None,
            "vlm_judge_reason_code": None,
        },
        pipeline_mode="vlm",
        target_language="zh",
        skill_context={},
    )

    print("strategy=", result.get("strategy_suggestions"))
    print("params=", result.get("skill_param_suggestions"))
    assert result.get("strategy_suggestions", {}).get("recommended_page_action") == "retry_erase"
    assert result.get("strategy_suggestions", {}).get("preferred_rerun_action") == "retry_erase"
    assert result.get("skill_param_suggestions", {}).get("lama_eraser") == {
        "profile": "aggressive",
        "mask_dilate_px": 5,
        "embedded_box_expand_px": 8,
        "embedded_complete_mask_close_px": 9,
    }


if __name__ == "__main__":
    main()
