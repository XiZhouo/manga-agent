from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class StubBrain:
    def __init__(self) -> None:
        self.last_brain_context = None

    def plan_user_message(self, *, user_message: str, page_count: int, pipeline_mode: str, target_language: str, brain_context=None, skill_context=None):
        self.last_brain_context = brain_context
        return {
            "intent": "correction",
            "needs_clarification": False,
            "clarifying_question": None,
            "reference_page_number": 1,
            "scope": "page_only",
            "from_side": "src",
            "from_text": "Force",
            "to_text": "原力",
            "glossary_type": "term",
            "apply_glossary": True,
            "confidence": 0.95,
            "user_message": user_message,
        }


class MemoryTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.last_retrieval_context = None

    def run_vlm_page(self, page, **kw):
        self.last_retrieval_context = kw.get("retrieval_context")
        return {
            "ok": True,
            "regions": [
                {
                    "region_id": 1,
                    "bbox": [10, 10, 100, 40],
                    "region_type": "text",
                    "has_text_box": True,
                    "src_text": "May the Force be with you",
                    "tgt_text": "愿原力与你同在",
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

    def run_erase(self, page, *, retry: bool = False, skill_params=None):
        return {"ok": True, "erase_backend": "mock", "results": {}, "metrics": {"erase_backend": "mock"}}

    def run_typeset(self, page, *, skill_params=None):
        return {"ok": True, "results": {1: {"render_ok": True, "overflow": False}}, "metrics": {}}

    def run_vlm_translation_revision(self, **kwargs):
        pages = kwargs["pages"]
        revised = json.loads(json.dumps(pages))
        revised[0]["regions"][0]["tgt_text"] = "原力与你同在"
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
                        "expected_old_tgt": "愿原力与你同在",
                        "new_tgt": "原力与你同在",
                        "confidence": 0.99,
                        "why": "Canonical franchise term.",
                    }
                ],
                "glossary_updates": [
                    {
                        "type": "term",
                        "src_text": "Force",
                        "tgt_text": "原力",
                        "locked": True,
                        "aliases_src": [],
                        "aliases_tgt": [],
                    }
                ],
                "warnings": [],
            },
            "metrics": {"candidate_pages": 1, "patches": 1},
            "error": None,
        }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent1_mem_") as td:
        job_dir = Path(td)
        pages_dir = job_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        img_path = pages_dir / "page_0000.png"
        img_path.write_bytes(b"stub")

        store = InMemoryStateStore()
        tools = MemoryTools()
        brain = StubBrain()
        agent = ComicAgent(store=store, tools=tools, decision_engine=DecisionEngine(), brain=brain)

        job_id = agent.create_job(
            user_id="test",
            source_file_name="dummy",
            target_language="zh",
            pages=[(0, str(img_path))],
        )

        job = store.get_job(job_id)
        job.user_preferences["pipeline_mode"] = "vlm"
        job.user_preferences["job_dir"] = str(job_dir)
        agent.memory.store.upsert_entries(
            job,
            "glossary",
            [
                {
                    "type": "term",
                    "src_text": "Force",
                    "tgt_text": "原力",
                    "locked": True,
                    "scope": "project",
                    "source": "seed",
                }
            ],
        )
        store.save_job(job)

        job = agent.run_until_waiting_or_done(job_id, max_steps=10)
        assert job.status.value == "completed", job.status
        assert tools.last_retrieval_context is not None
        glossary_hits = tools.last_retrieval_context.get("glossary_hits") or []
        print("glossary_hits=", len(glossary_hits))
        assert any(str(x.get("src_text")) == "Force" for x in glossary_hits)

        agent.handle_user_message(job_id, "第1页的 Force 应该翻译成 原力")
        job = agent.run_until_waiting_or_done(job_id, max_steps=20)
        assert job.status.value == "completed", job.status
        assert brain.last_brain_context is not None
        print("brain_summary_present=", bool(brain.last_brain_context.get("brain_summary")))
        assert "Relevant terminology:" in str(brain.last_brain_context.get("brain_summary") or "")

        tm_path = job_dir / "memory" / "translation_memory.json"
        tm = json.loads(tm_path.read_text(encoding="utf-8"))
        entries = tm.get("entries") or []
        print("translation_memory_entries=", len(entries))
        assert len(entries) == 1
        assert any(e.get("src_text") == "May the Force be with you" and e.get("tgt_text") == "原力与你同在" for e in entries)

        brain_mem_path = job_dir / "memory" / "brain_memory.json"
        brain_mem = json.loads(brain_mem_path.read_text(encoding="utf-8"))
        print("brain_memory_entries=", len(brain_mem.get("entries") or []))
        assert len(brain_mem.get("entries") or []) >= 1


if __name__ == "__main__":
    main()
