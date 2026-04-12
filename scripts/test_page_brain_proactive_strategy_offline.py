from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class ProactiveBrain:
    def plan_user_message(self, *, user_message: str, page_count: int, pipeline_mode: str, target_language: str, brain_context=None, skill_context=None):
        return {
            "intent": "unknown",
            "needs_clarification": False,
            "clarifying_question": None,
            "reference_page_number": None,
            "scope": "page_only",
            "from_side": "auto",
            "from_text": None,
            "to_text": None,
            "glossary_type": None,
            "apply_glossary": False,
            "strategy_suggestions": {},
            "skill_param_suggestions": {},
            "confidence": 0.5,
        }

    def suggest_page_strategy(self, *, page_summary, pipeline_mode: str, target_language: str, skill_context=None):
        stage = str(page_summary.get("stage") or "")
        embedded_ratio = float(page_summary.get("embedded_ratio") or 0.0)
        bad_render_regions = int(page_summary.get("bad_render_regions") or 0)

        if stage == "translated" and embedded_ratio >= 0.5:
            return {
                "strategy_suggestions": {
                    "page_profile": "art_integrated",
                    "preferred_post_translation_action": "erase_direct",
                    "recommended_page_action": "continue",
                },
                "skill_param_suggestions": {
                    "lama_eraser": {"profile": "aggressive", "mask_dilate_px": 4},
                    "lettering": {"render_mode": "art_embed", "padding_ratio": 0.06, "stroke_width": 3},
                },
                "summary": "embedded text page; skip auto revision and use art-integrated rendering",
                "confidence": 0.95,
            }

        if stage == "typesetting" and bad_render_regions > 0:
            return {
                "strategy_suggestions": {
                    "recommended_page_action": "retry_erase",
                    "preferred_rerun_action": "retry_erase",
                },
                "skill_param_suggestions": {
                    "lama_eraser": {"profile": "aggressive", "mask_dilate_px": 5},
                },
                "summary": "overflow likely caused by poor erase/mask fit; retry erase",
                "confidence": 0.9,
            }

        return {
            "strategy_suggestions": {},
            "skill_param_suggestions": {},
            "summary": "no extra page strategy",
            "confidence": 0.5,
        }


class ProactiveTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.auto_revision_calls = 0
        self.erase_params_history = []
        self.typeset_params_history = []
        self.typeset_calls = 0

    def run_vlm_page(self, page, **kw):
        return {
            "ok": True,
            "regions": [
                {
                    "region_id": 1,
                    "bbox": [10, 10, 100, 40],
                    "region_type": "text",
                    "has_text_box": False,
                    "src_text": "BOOM",
                    "tgt_text": "轰",
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
        if kwargs.get("revision_mode") == "auto_rag_revision":
            self.auto_revision_calls += 1
        return {
            "ok": True,
            "data": {
                "decision": "apply",
                "pages": kwargs["pages"],
                "changed_pages": [],
                "patches": [],
                "glossary_updates": [],
                "warnings": [],
            },
            "metrics": {"candidate_pages": len(kwargs["pages"]), "patches": 0},
            "error": None,
        }

    def run_erase(self, page, *, retry: bool = False, skill_params=None):
        self.erase_params_history.append(dict(skill_params or {}))
        return {"ok": True, "erase_backend": "mock", "results": {}, "metrics": {"erase_backend": "mock", "retry": retry}}

    def run_typeset(self, page, *, skill_params=None):
        self.typeset_calls += 1
        self.typeset_params_history.append(dict(skill_params or {}))
        overflow = self.typeset_calls == 1
        return {
            "ok": True,
            "results": {1: {"render_ok": True, "overflow": overflow}},
            "metrics": {"pass": self.typeset_calls},
        }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent1_page_brain_") as td:
        img_path = Path(td) / "page.png"
        img_path.write_bytes(b"stub")

        store = InMemoryStateStore()
        tools = ProactiveTools()
        agent = ComicAgent(store=store, tools=tools, decision_engine=DecisionEngine(), brain=ProactiveBrain())

        job_id = agent.create_job(
            user_id="test",
            source_file_name="dummy",
            target_language="zh",
            pages=[(0, str(img_path))],
        )
        job = store.get_job(job_id)
        job.user_preferences["pipeline_mode"] = "vlm"
        store.save_job(job)

        job = agent.run_until_waiting_or_done(job_id, max_steps=20)
        assert job.status.value == "completed", job.status

        page = job.pages[0]
        print("auto_revision_calls=", tools.auto_revision_calls)
        print("erase_params_history=", tools.erase_params_history)
        print("typeset_params_history=", tools.typeset_params_history)
        print("page_strategy=", page.strategy_suggestions)
        print("page_brain_summary=", page.page_brain_summary)
        print("page_brain_last_actionable_summary=", page.page_brain_last_actionable_summary)

        assert tools.auto_revision_calls == 0
        assert tools.erase_params_history[0] == {"profile": "aggressive", "mask_dilate_px": 4}
        assert tools.erase_params_history[-1] == {"profile": "aggressive", "mask_dilate_px": 5}
        assert tools.typeset_params_history[0] == {"render_mode": "art_embed", "padding_ratio": 0.06, "stroke_width": 3}
        assert page.strategy_suggestions.get("page_profile") == "art_integrated"
        assert page.strategy_suggestions.get("preferred_post_translation_action") == "erase_direct"
        assert page.strategy_suggestions.get("recommended_page_action") == "retry_erase"
        assert page.strategy_suggestions.get("preferred_rerun_action") == "retry_erase"
        assert page.page_brain_stage_last == "after_typeset"
        assert page.page_brain_summary is not None
        assert page.page_brain_last_actionable_summary is not None


if __name__ == "__main__":
    main()
