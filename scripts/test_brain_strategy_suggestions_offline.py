from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore, PageStage, RegionStage, RegionState
from orchestrator.tools import Tools


class StrategyBrain:
    def plan_user_message(self, *, user_message: str, page_count: int, pipeline_mode: str, target_language: str, brain_context=None, skill_context=None):
        if "艺术字" in user_message:
            return {
                "intent": "preference",
                "needs_clarification": False,
                "clarifying_question": None,
                "reference_page_number": 1,
                "scope": "page_only",
                "from_side": "auto",
                "from_text": None,
                "to_text": None,
                "glossary_type": None,
                "apply_glossary": False,
                "strategy_suggestions": {
                    "page_profile": "art_integrated",
                    "preferred_post_translation_action": "erase_direct",
                },
                "confidence": 0.9,
            }
        if "重跑" in user_message:
            return {
                "intent": "rerun_page",
                "needs_clarification": False,
                "clarifying_question": None,
                "reference_page_number": 1,
                "scope": "page_only",
                "from_side": "auto",
                "from_text": None,
                "to_text": None,
                "glossary_type": None,
                "apply_glossary": False,
                "strategy_suggestions": {
                    "preferred_rerun_action": "retry_typeset",
                    "reuse_erased_image": True,
                },
                "confidence": 0.91,
            }
        if "整页重翻" in user_message:
            return {
                "intent": "correction",
                "needs_clarification": False,
                "clarifying_question": None,
                "reference_page_number": 1,
                "scope": "page_only",
                "from_side": "auto",
                "from_text": None,
                "to_text": None,
                "glossary_type": None,
                "apply_glossary": False,
                "strategy_suggestions": {
                    "preferred_correction_action": "retry_vlm_page",
                },
                "confidence": 0.89,
            }

        return {
            "intent": "correction",
            "needs_clarification": False,
            "clarifying_question": None,
            "reference_page_number": 1,
            "scope": "global",
            "from_side": "src",
            "from_text": "echo",
            "to_text": "回声",
            "glossary_type": "term",
            "apply_glossary": True,
            "strategy_suggestions": {
                "revision_scope": "page_only",
                "revision_page_limit": 1,
            },
            "confidence": 0.93,
        }


class StrategyTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.last_revision_pages = None
        self.auto_revision_calls = 0

    def run_vlm_page(self, page, **kw):
        return {
            "ok": True,
            "regions": [
                {
                    "region_id": 1,
                    "bbox": [10, 10, 100, 40],
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
        self.last_revision_pages = len(kwargs["pages"])
        pages = kwargs["pages"]
        revised = []
        for p in pages:
            p2 = {"page_index": int(p.get("page_index")), "regions": []}
            for r in p.get("regions") or []:
                p2["regions"].append(dict(r, tgt_text="回声"))
            revised.append(p2)
        return {
            "ok": True,
            "data": {
                "decision": "apply",
                "pages": revised,
                "changed_pages": [int(p.get("page_index")) for p in pages],
                "patches": [
                    {
                        "page_index": int(p.get("page_index")),
                        "region_id": 1,
                        "expected_old_tgt": "回音",
                        "new_tgt": "回声",
                    }
                    for p in pages
                ],
                "glossary_updates": [],
                "warnings": [],
            },
            "metrics": {"candidate_pages": len(pages), "patches": len(pages)},
            "error": None,
        }

    def run_erase(self, page, *, retry: bool = False, skill_params=None):
        return {"ok": True, "erase_backend": "mock", "results": {}, "metrics": {"erase_backend": "mock"}}

    def run_typeset(self, page, *, skill_params=None):
        return {"ok": True, "results": {1: {"render_ok": True, "overflow": False}}, "metrics": {}}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent1_brain_strategy_") as td:
        img_dir = Path(td)
        pages = []
        for idx in range(3):
            p = img_dir / f"page_{idx}.png"
            p.write_bytes(b"stub")
            pages.append((idx, str(p)))

        store = InMemoryStateStore()
        tools = StrategyTools()
        engine = DecisionEngine()
        agent = ComicAgent(store=store, tools=tools, decision_engine=engine, brain=StrategyBrain())

        job_id = agent.create_job(
            user_id="test",
            source_file_name="dummy",
            target_language="zh",
            pages=pages,
        )
        job = store.get_job(job_id)
        job.user_preferences["pipeline_mode"] = "vlm"
        store.save_job(job)

        agent.handle_user_message(job_id, "第1页是艺术字页，先直接擦除再嵌字")
        job = store.get_job(job_id)
        plan = job.pending_user_plan
        print("pending_after_preference=", plan)
        assert plan is None
        assert job.pages[0].strategy_suggestions.get("page_profile") == "art_integrated"
        assert job.pages[0].strategy_suggestions.get("preferred_post_translation_action") == "erase_direct"
        assert job.pages[0].skill_param_suggestions.get("lama_eraser", {}).get("profile") == "aggressive"
        assert job.pages[0].skill_param_suggestions.get("lettering", {}).get("render_mode") == "art_embed"

        job = agent.run_until_waiting_or_done(job_id, max_steps=30)
        assert job.status.value == "completed"
        print("auto_revision_calls=", tools.auto_revision_calls)
        assert tools.auto_revision_calls == 2

        agent.handle_user_message(job_id, "第1页的 echo 应该统一一下")
        job = store.get_job(job_id)
        plan = job.pending_user_plan or {}
        print("strategy_after_correction=", plan.get("strategy_suggestions"))
        assert (plan.get("strategy_suggestions") or {}).get("revision_scope") == "page_only"
        action = engine.decide_next_action(job)
        assert action.action_type.value == "run_revision", action
        job = agent.run_until_waiting_or_done(job_id, max_steps=20)
        print("revision_pages_used=", tools.last_revision_pages)
        assert tools.last_revision_pages == 1

        agent.handle_user_message(job_id, "第1页整页重翻一下")
        job = store.get_job(job_id)
        plan = job.pending_user_plan or {}
        print("strategy_after_full_rerun_correction=", plan.get("strategy_suggestions"))
        action = engine.decide_next_action(job)
        print("correction_action=", action.action_type.value)
        assert (plan.get("strategy_suggestions") or {}).get("preferred_correction_action") == "retry_vlm_page"
        assert action.action_type.value == "retry_vlm_page", action

        job = store.get_job(job_id)
        page0 = job.pages[0]
        page0.stage = PageStage.DONE
        page0.erased_image_ref = "cached_erased.png"
        page0.rendered_image_ref = "cached_rendered.png"
        page0.regions = {
            1: RegionState(
                region_id=1,
                bbox=[10, 10, 100, 40],
                ocr_text="echo",
                translated_text="回声",
                stage=RegionStage.RENDERED,
                render_ok=True,
            )
        }
        store.save_job(job)

        agent.handle_user_message(job_id, "第1页重跑一下排版")
        job = store.get_job(job_id)
        plan = job.pending_user_plan or {}
        print("strategy_after_rerun=", plan.get("strategy_suggestions"))
        action = engine.decide_next_action(job)
        print("rerun_action=", action.action_type.value)
        assert (plan.get("strategy_suggestions") or {}).get("preferred_rerun_action") == "retry_typeset"
        assert action.action_type.value == "retry_typeset", action


if __name__ == "__main__":
    main()
