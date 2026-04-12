from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class JobBatchBrain:
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

    def suggest_job_strategy(self, *, job_summary, pipeline_mode: str, target_language: str, skill_context=None):
        next_pages = list(job_summary.get("next_not_started_page_indices") or [])
        active_page_count = int(job_summary.get("active_page_count") or 0)
        batch_size = int(job_summary.get("preferred_vlm_batch_size") or 10)
        if pipeline_mode == "vlm" and active_page_count == 0 and next_pages:
            chosen = next_pages[:batch_size]
            return {
                "strategy_suggestions": {
                    "recommended_job_action": "run_vlm_batch",
                    "vlm_batch_size": len(chosen),
                    "vlm_batch_page_indices": chosen,
                    "batching_mode": "contiguous_window",
                },
                "summary": f"open vlm batch {chosen[0]}-{chosen[-1]}",
                "confidence": 0.96,
            }
        return {
            "strategy_suggestions": {"recommended_job_action": "continue"},
            "summary": "continue current work",
            "confidence": 0.6,
        }


class JobBatchTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes = []
        self.batch_ranges = []
        self.revision_ranges = []

    def run_vlm_pages(self, pages, *, reading_direction="ltr", source_language="en", target_language="zh", retrieval_contexts=None):
        page_indices = [int(page.page_index) for page in pages]
        self.batch_sizes.append(len(page_indices))
        self.batch_ranges.append((page_indices[0], page_indices[-1]))
        return {
            "ok": True,
            "page_range": {
                "start_page_index": page_indices[0],
                "end_page_index": page_indices[-1],
                "page_count": len(page_indices),
                "page_indices": page_indices,
                "label": f"{page_indices[0]}-{page_indices[-1]}",
            },
            "pages": [
                {
                    "page_index": page_index,
                    "regions": [
                        {
                            "region_id": 1,
                            "bbox": [10, 10, 100, 40],
                            "region_type": "text",
                            "has_text_box": True,
                            "src_text": f"hello-{page_index}",
                            "tgt_text": f"你好-{page_index}",
                        }
                    ],
                    "bbox_error": False,
                    "vlm_metrics": {
                        "orig": (100, 100),
                        "sent": (100, 100),
                        "downscaled": False,
                        "usage": {"total_tokens": 10},
                    },
                }
                for page_index in page_indices
            ],
            "batch_metrics": {"page_count": len(page_indices), "usage": {"total_tokens": 100}},
            "raw": {"ok": True},
        }

    def run_vlm_translation_revision(self, **kwargs):
        self.revision_ranges.append(kwargs.get("document_range"))
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
    with tempfile.TemporaryDirectory(prefix="agent1_job_batch_") as td:
        img_dir = Path(td)
        pages = []
        for idx in range(12):
            p = img_dir / f"page_{idx}.png"
            p.write_bytes(b"stub")
            pages.append((idx, str(p)))

        store = InMemoryStateStore()
        tools = JobBatchTools()
        agent = ComicAgent(store=store, tools=tools, decision_engine=DecisionEngine(), brain=JobBatchBrain())

        job_id = agent.create_job(
            user_id="test",
            source_file_name="dummy",
            target_language="zh",
            pages=pages,
        )
        job = store.get_job(job_id)
        job.user_preferences["pipeline_mode"] = "vlm"
        store.save_job(job)

        job = agent.run_until_waiting_or_done(job_id, max_steps=300)
        assert job.status.value == "completed", job.status

        print("batch_sizes=", tools.batch_sizes)
        print("batch_ranges=", tools.batch_ranges)
        print("vlm_batches=", len(job.vlm_batches))
        print("revision_ranges=", tools.revision_ranges)
        print("job_brain_summary=", job.job_brain_summary)

        assert tools.batch_sizes == [10, 2]
        assert tools.batch_ranges == [(0, 9), (10, 11)]
        assert len(job.vlm_batches) == 2
        assert job.vlm_batches[0]["page_range"]["label"] == "0-9"
        assert job.vlm_batches[1]["page_range"]["label"] == "10-11"
        assert job.pages[0].vlm_batch_range["label"] == "0-9"
        assert job.pages[10].vlm_batch_range["label"] == "10-11"
        assert [item.get("label") for item in tools.revision_ranges] == ["0-9", "10-11"]


if __name__ == "__main__":
    main()
