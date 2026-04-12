from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class RecommendedActionBrain:
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
        if str(page_summary.get("stage") or "") == "translated":
            return {
                "strategy_suggestions": {
                    "preferred_post_translation_action": "erase_direct",
                    "recommended_page_action": "run_revision",
                },
                "skill_param_suggestions": {},
                "summary": "run a text-only revision before erase even though direct erase would otherwise be allowed",
                "confidence": 0.93,
            }

        return {
            "strategy_suggestions": {},
            "skill_param_suggestions": {},
            "summary": "no extra page strategy",
            "confidence": 0.5,
        }


class RecommendedActionTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.auto_revision_calls = 0
        self.erase_calls = 0
        self.typeset_calls = 0

    def run_vlm_page(self, page, **kw):
        return {
            "ok": True,
            "regions": [
                {
                    "region_id": 1,
                    "bbox": [10, 10, 120, 40],
                    "region_type": "text",
                    "has_text_box": True,
                    "src_text": "echo",
                    "tgt_text": "回音",
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
        pages_out = []
        for page in kwargs["pages"]:
            pages_out.append(
                {
                    "page_index": int(page.get("page_index")),
                    "regions": [
                        dict(region, tgt_text="回声")
                        for region in (page.get("regions") or [])
                    ],
                }
            )
        return {
            "ok": True,
            "data": {
                "decision": "apply",
                "pages": pages_out,
                "changed_pages": [int(page.get("page_index")) for page in kwargs["pages"]],
                "patches": [
                    {
                        "page_index": int(page.get("page_index")),
                        "region_id": 1,
                        "expected_old_tgt": "回音",
                        "new_tgt": "回声",
                    }
                    for page in kwargs["pages"]
                ],
                "glossary_updates": [],
                "warnings": [],
            },
            "metrics": {"candidate_pages": len(kwargs["pages"]), "patches": len(kwargs["pages"])},
            "error": None,
        }

    def run_erase(self, page, *, retry: bool = False, skill_params=None):
        self.erase_calls += 1
        return {"ok": True, "erase_backend": "mock", "results": {}, "metrics": {"erase_backend": "mock"}}

    def run_typeset(self, page, *, skill_params=None):
        self.typeset_calls += 1
        return {"ok": True, "results": {1: {"render_ok": True, "overflow": False}}, "metrics": {"pass": self.typeset_calls}}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent1_page_brain_actions_") as td:
        img_path = Path(td) / "page.png"
        img_path.write_bytes(b"stub")

        store = InMemoryStateStore()
        tools = RecommendedActionTools()
        agent = ComicAgent(store=store, tools=tools, decision_engine=DecisionEngine(), brain=RecommendedActionBrain())

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
        print("erase_calls=", tools.erase_calls)
        print("typeset_calls=", tools.typeset_calls)
        print("final_text=", page.regions[1].translated_text)
        print("page_strategy=", page.strategy_suggestions)

        assert tools.auto_revision_calls == 1
        assert tools.erase_calls == 1
        assert tools.typeset_calls == 1
        assert page.regions[1].translated_text == "回声"
        assert page.strategy_suggestions.get("recommended_page_action") == "run_revision"
        assert page.page_brain_last_actionable_summary is not None


if __name__ == "__main__":
    main()
