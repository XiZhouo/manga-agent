from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class StubBrain:
    def plan_user_message(self, *, user_message: str, page_count: int, pipeline_mode: str, target_language: str, brain_context=None, skill_context=None):
        return {
            "intent": "correction",
            "needs_clarification": False,
            "clarifying_question": None,
            "reference_page_number": 1,
            "scope": "page_only",
            "from_side": "src",
            "from_text": "boom",
            "to_text": "轰",
            "glossary_type": "phrase",
            "apply_glossary": False,
            "skill_param_suggestions": {
                "lama_eraser": {"profile": "aggressive", "mask_dilate_px": 99},
                "lettering": {"render_mode": "art_embed", "padding_ratio": 0.5, "stroke_width": 4},
                "unknown_skill": {"x": 1},
            },
            "confidence": 0.88,
        }


class SuggestionTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.erase_params = None
        self.typeset_params = None

    def run_vlm_page(self, page, **kw):
        return {
            "ok": True,
            "regions": [
                {
                    "region_id": 1,
                    "bbox": [10, 10, 100, 40],
                    "region_type": "text",
                    "has_text_box": False,
                    "src_text": "boom",
                    "tgt_text": "砰",
                }
            ],
            "bbox_error": False,
            "vlm_metrics": {
                "orig": (100, 100),
                "sent": (100, 100),
                "downscaled": False,
                "usage": {"total_tokens": 1},
            },
            "raw": {"ok": True, "metrics": {"regions": 1}},
        }

    def run_vlm_translation_revision(self, **kwargs):
        pages = kwargs["pages"]
        revised = [dict(p) for p in pages]
        revised[0] = {"page_index": 0, "regions": [dict(revised[0]["regions"][0], tgt_text="轰")]}
        return {
            "ok": True,
            "data": {
                "decision": "apply",
                "pages": revised,
                "changed_pages": [0],
                "patches": [
                    {
                        "page_index": 0,
                        "region_id": 1,
                        "expected_old_tgt": "砰",
                        "new_tgt": "轰",
                    }
                ],
                "glossary_updates": [],
                "warnings": [],
            },
            "metrics": {"candidate_pages": 1, "patches": 1},
            "error": None,
        }

    def run_erase(self, page, *, retry: bool = False, skill_params=None):
        self.erase_params = dict(skill_params or {})
        return {"ok": True, "erase_backend": "mock", "results": {}, "metrics": {"erase_backend": "mock"}}

    def run_typeset(self, page, *, skill_params=None):
        self.typeset_params = dict(skill_params or {})
        return {"ok": True, "results": {1: {"render_ok": True, "overflow": False}}, "metrics": {}}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent1_brain_params_") as td:
        img_path = Path(td) / "page.png"
        Image.new("RGB", (100, 100), (255, 255, 255)).save(img_path)

        store = InMemoryStateStore()
        tools = SuggestionTools()
        agent = ComicAgent(store=store, tools=tools, decision_engine=DecisionEngine(), brain=StubBrain())

        job_id = agent.create_job(
            user_id="test",
            source_file_name="dummy",
            target_language="zh",
            pages=[(0, str(img_path))],
        )
        job = store.get_job(job_id)
        job.user_preferences["pipeline_mode"] = "vlm"
        store.save_job(job)

        job = agent.run_until_waiting_or_done(job_id, max_steps=10)
        assert job.status.value == "completed"

        agent.handle_user_message(job_id, "第1页这个嵌在背景里的字效果不好，帮我改一下")
        job = agent.run_until_waiting_or_done(job_id, max_steps=20)
        assert job.status.value == "completed"

        print("erase_params=", tools.erase_params)
        print("typeset_params=", tools.typeset_params)
        assert tools.erase_params == {"profile": "aggressive", "mask_dilate_px": 5}
        assert tools.typeset_params == {"render_mode": "art_embed", "padding_ratio": 0.2, "stroke_width": 4}


if __name__ == "__main__":
    main()
