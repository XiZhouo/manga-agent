from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class WindowRevisionBrain:
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


class WindowRevisionTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.last_revision_retrieval_context = None
        self.last_revision_document_range = None
        self.auto_revision_calls = 0

    def run_vlm_pages(self, pages, *, reading_direction="ltr", source_language="en", target_language="zh", retrieval_contexts=None):
        page_indices = [int(page.page_index) for page in pages]
        outputs = []
        source_pairs = {
            0: ("May the Force be with you", "愿力量与你同在"),
            1: ("Batman arrives in Gotham", "蝙蝠侠来到高谭市"),
            2: ("The Force returns", "力量归来"),
        }
        for page_index in page_indices:
            src, tgt = source_pairs[page_index]
            outputs.append(
                {
                    "page_index": page_index,
                    "regions": [
                        {
                            "region_id": 1,
                            "bbox": [10, 10, 100, 40],
                            "region_type": "text",
                            "has_text_box": True,
                            "src_text": src,
                            "tgt_text": tgt,
                        }
                    ],
                    "bbox_error": False,
                    "vlm_metrics": {
                        "orig": (100, 100),
                        "sent": (100, 100),
                        "downscaled": False,
                        "usage": {"total_tokens": 1},
                    },
                }
            )
        return {
            "ok": True,
            "page_range": {
                "start_page_index": page_indices[0],
                "end_page_index": page_indices[-1],
                "page_count": len(page_indices),
                "page_indices": page_indices,
                "label": f"{page_indices[0]}-{page_indices[-1]}",
            },
            "pages": outputs,
            "batch_metrics": {"page_count": len(outputs), "usage": {"total_tokens": 3}},
            "raw": {"ok": True},
        }

    def run_vlm_translation_revision(self, **kwargs):
        if kwargs.get("revision_mode") == "auto_rag_revision":
            self.auto_revision_calls += 1
        self.last_revision_retrieval_context = kwargs.get("retrieval_context")
        self.last_revision_document_range = kwargs.get("document_range")
        return {
            "ok": True,
            "data": {
                "decision": "apply",
                "document_range": kwargs.get("document_range"),
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
        return {"ok": True, "erase_backend": "mock", "results": {}, "metrics": {"erase_backend": "mock"}}

    def run_typeset(self, page, *, skill_params=None):
        return {"ok": True, "results": {1: {"render_ok": True, "overflow": False}}, "metrics": {"pass": 1}}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="agent1_revision_window_") as td:
        job_dir = Path(td)
        pages = []
        for idx in range(3):
            p = job_dir / f"page_{idx}.png"
            p.write_bytes(b"stub")
            pages.append((idx, str(p)))

        store = InMemoryStateStore()
        tools = WindowRevisionTools()
        agent = ComicAgent(store=store, tools=tools, decision_engine=DecisionEngine(), brain=WindowRevisionBrain())

        job_id = agent.create_job(
            user_id="test",
            source_file_name="dummy",
            target_language="zh",
            pages=pages,
        )
        job = store.get_job(job_id)
        job.user_preferences["pipeline_mode"] = "vlm"
        job.user_preferences["job_dir"] = str(job_dir)
        agent.memory.store.upsert_entries(
            job,
            "glossary",
            [
                {"type": "term", "src_text": "Force", "tgt_text": "原力", "locked": True, "scope": "project", "source": "seed"},
                {"type": "name", "src_text": "Batman", "tgt_text": "蝙蝠侠", "locked": True, "scope": "project", "source": "seed"},
            ],
        )
        agent.memory.store.upsert_entries(
            job,
            "translation_memory",
            [
                {"type": "term", "src_text": "Gotham", "tgt_text": "哥谭", "locked": True, "scope": "project", "source": "seed"},
            ],
        )
        store.save_job(job)

        job = agent.run_until_waiting_or_done(job_id, max_steps=40)
        assert job.status.value == "completed", job.status

        ctx = tools.last_revision_retrieval_context or {}
        print("auto_revision_calls=", tools.auto_revision_calls)
        print("revision_document_range=", tools.last_revision_document_range)
        print("retrieval_keys=", sorted(ctx.keys()))
        print("target_page_indices=", ctx.get("target_page_indices"))
        print("window_context=", ctx.get("window_context"))
        print("term_hints=", ctx.get("term_hints"))

        glossary_hits = ctx.get("glossary_hits") or []
        tm_hits = ctx.get("translation_memory_hits") or []
        target_examples = ctx.get("target_page_examples") or []

        assert tools.auto_revision_calls == 1
        assert (tools.last_revision_document_range or {}).get("label") == "0-2"
        assert (ctx.get("document_range") or {}).get("label") == "0-2"
        assert ctx.get("target_page_indices") == [0, 1, 2]
        assert len(target_examples) == 3
        assert any(str(item.get("src_text")) == "Force" for item in glossary_hits)
        assert any(str(item.get("src_text")) == "Batman" for item in glossary_hits)
        assert any(str(item.get("src_text")) == "Gotham" for item in tm_hits)
        assert "Revision window: 0-2" in str(ctx.get("window_context") or "")


if __name__ == "__main__":
    main()
