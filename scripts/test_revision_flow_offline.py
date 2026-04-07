from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class StubBrain:
    def plan_user_message(self, *, user_message: str, page_count: int, pipeline_mode: str, target_language: str):
        # Simulate Route-A brain output (0 network calls)
        return {
            "intent": "correction",
            "needs_clarification": False,
            "clarifying_question": None,
            # User says 第1页, treat as human-facing 1-based.
            "reference_page_number": 1,
            "scope": "global",
            "from_side": "src",
            "from_text": "Hello",
            "to_text": "你好",
            "glossary_type": "quote",
            "apply_glossary": True,
            "confidence": 0.9,
            "user_message": user_message,
        }


class OfflineRevisionTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_calls = 0
        self.revision_calls = 0
        self.lettering_calls = 0

    def run_vlm_page(self, page, **kw):
        self.vlm_calls += 1
        if int(getattr(page, "page_index", 0)) == 1:
            # Page 2 already has correct translation.
            return {
                "ok": True,
                "regions": [
                    {"region_id": 1, "bbox": [10, 10, 100, 40], "region_type": "text", "src_text": "Hello", "tgt_text": "原力与你同在"},
                    {"region_id": 2, "bbox": [10, 60, 100, 40], "region_type": "text", "src_text": "World", "tgt_text": "世界"},
                ],
                "bbox_error": False,
                "vlm_metrics": {
                    "orig": (1988, 3057),
                    "sent": (1988, 3057),
                    "downscaled": False,
                    "usage": {"total_tokens": 111},
                },
                "raw": {"ok": True, "metrics": {"regions": 2}},
            }

        return {
            "ok": True,
            "regions": [
                {"region_id": 1, "bbox": [10, 10, 100, 40], "region_type": "text", "src_text": "Hello", "tgt_text": "希望力量陪着你"},
                {"region_id": 2, "bbox": [10, 60, 100, 40], "region_type": "text", "src_text": "World", "tgt_text": "世界"},
            ],
            "bbox_error": False,
            "vlm_metrics": {
                "orig": (1988, 3057),
                "sent": (1988, 3057),
                "downscaled": False,
                "usage": {"total_tokens": 111},
            },
            "raw": {"ok": True, "metrics": {"regions": 2}},
        }

    def run_vlm_translation_revision(self, *, pages, user_request, reference_page_number, scope, from_side, from_text, to_text, reading_direction, target_language, glossary=None):
        self.revision_calls += 1

        # Return full pages schema (same as VLM style), with updated tgt_text for region 1.
        revised = []
        for p in pages:
            if int(p.get("page_index")) != 0:
                revised.append(p)
                continue

            regions = []
            for r in p.get("regions") or []:
                r2 = dict(r)
                if int(r2.get("region_id")) == 1:
                    r2["tgt_text"] = "原力与你同在"
                regions.append(r2)

            revised.append({"page_index": 0, "regions": regions})

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
                        "expected_old_tgt": "希望力量陪着你",
                        "new_tgt": "原力与你同在",
                        "confidence": 0.95,
                        "why": "User corrected a known quote.",
                    }
                ],
                "glossary_updates": [
                    {
                        "type": "quote",
                        "src_text": "may the force be with you",
                        "tgt_text": "原力与你同在",
                        "locked": True,
                        "aliases_src": [],
                        "aliases_tgt": ["希望力量陪着你"],
                    }
                ],
                "warnings": [],
            },
            "metrics": {"candidate_pages": 1, "patches": 1, "usage": {"total_tokens": 0}},
            "error": None,
        }

    def run_lettering(self, page):
        self.lettering_calls += 1
        return super().run_lettering(page)


def main() -> None:
    store = InMemoryStateStore()
    tools = OfflineRevisionTools()
    engine = DecisionEngine()
    agent = ComicAgent(store=store, tools=tools, decision_engine=engine, brain=StubBrain())

    job_id = agent.create_job(
        user_id="test",
        source_file_name="dummy",
        target_language="zh",
        pages=[(0, "missing0.png"), (1, "missing1.png")],
    )

    job = store.get_job(job_id)
    job.user_preferences["pipeline_mode"] = "vlm"
    store.save_job(job)

    # First run: VLM -> lettering -> complete
    job = agent.run_until_waiting_or_done(job_id, max_steps=10)
    print("first_status=", job.status.value)
    print("vlm_calls=", tools.vlm_calls, "lettering_calls=", tools.lettering_calls, "revision_calls=", tools.revision_calls)
    print("tgt_before_p1=", job.pages[0].regions[1].translated_text)
    print("tgt_before_p2=", job.pages[1].regions[1].translated_text)

    # User correction
    agent.handle_user_message(job_id, "第1页的 Hello 应该翻译成 原力与你同在，请修改所有相关的问题")

    # Sanity: decision engine should interpret human page number "1" as page_index=0.
    job = store.get_job(job_id)
    action = engine.decide_next_action(job)
    assert action.action_type.value == "run_revision", action
    assert action.page_index == 0, action

    job = store.get_job(job_id)
    job.status = job.status  # keep whatever handle_user_message set
    store.save_job(job)

    # Second run: RUN_REVISION -> lettering -> complete
    job = agent.run_until_waiting_or_done(job_id, max_steps=20)

    print("second_status=", job.status.value)
    print("vlm_calls=", tools.vlm_calls, "lettering_calls=", tools.lettering_calls, "revision_calls=", tools.revision_calls)
    print("tgt_after_p1=", job.pages[0].regions[1].translated_text)
    print("tgt_after_p2=", job.pages[1].regions[1].translated_text)
    print("glossary_entries=", len((job.glossary or {}).get("entries") or []))
    print("tail_logs=")
    for line in job.logs[-12:]:
        print(" ", line)


if __name__ == "__main__":
    main()
