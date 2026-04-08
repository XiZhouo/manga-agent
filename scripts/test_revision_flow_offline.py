from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class StubBrain:
    def plan_user_message(self, *, user_message: str, page_count: int, pipeline_mode: str, target_language: str, brain_context=None, skill_context=None):
        if "World" in user_message:
            return {
                "intent": "correction",
                "needs_clarification": False,
                "clarifying_question": None,
                "reference_page_number": 1,
                "scope": "global",
                "from_side": "src",
                "from_text": "World",
                "to_text": "天下",
                "glossary_type": "term",
                "apply_glossary": True,
                "confidence": 0.9,
                "user_message": user_message,
            }

        return {
            "intent": "correction",
            "needs_clarification": False,
            "clarifying_question": None,
            "reference_page_number": 1,
            "scope": "global",
            "from_side": "src",
            "from_text": "Hello",
            "to_text": "原力与你同在",
            "glossary_type": "quote",
            "apply_glossary": True,
            "confidence": 0.9,
            "user_message": user_message,
        }


class OfflineRevisionTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_calls = 0
        self.auto_revision_calls = 0
        self.user_revision_calls = 0
        self.lettering_calls = 0

    def run_vlm_page(self, page, **kw):
        self.vlm_calls += 1
        if int(getattr(page, "page_index", 0)) == 1:
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

    def run_vlm_translation_revision(self, **kwargs):
        pages = kwargs["pages"]
        mode = kwargs.get("revision_mode")
        revised = []
        patches = []
        changed_pages = set()
        glossary_updates = []

        if mode == "auto_rag_revision":
            self.auto_revision_calls += 1
            for p in pages:
                p2 = {"page_index": int(p.get("page_index")), "regions": []}
                for r in p.get("regions") or []:
                    r2 = dict(r)
                    if int(r2.get("region_id")) == 1 and r2.get("tgt_text") == "希望力量陪着你":
                        r2["tgt_text"] = "原力与你同在"
                        changed_pages.add(int(p2["page_index"]))
                        patches.append(
                            {
                                "page_index": int(p2["page_index"]),
                                "region_id": 1,
                                "expected_old_tgt": "希望力量陪着你",
                                "new_tgt": "原力与你同在",
                                "confidence": 0.95,
                                "why": "Auto RAG aligned the franchise quote.",
                            }
                        )
                        glossary_updates.append(
                            {
                                "type": "quote",
                                "src_text": "Hello",
                                "tgt_text": "原力与你同在",
                                "locked": True,
                                "aliases_src": [],
                                "aliases_tgt": ["希望力量陪着你"],
                            }
                        )
                    p2["regions"].append(r2)
                revised.append(p2)
        else:
            self.user_revision_calls += 1
            for p in pages:
                p2 = {"page_index": int(p.get("page_index")), "regions": []}
                for r in p.get("regions") or []:
                    r2 = dict(r)
                    if r2.get("src_text") == "World":
                        r2["tgt_text"] = "天下"
                        changed_pages.add(int(p2["page_index"]))
                    p2["regions"].append(r2)
                revised.append(p2)
            patches = [
                {
                    "page_index": int(p.get("page_index")),
                    "region_id": 2,
                    "expected_old_tgt": "世界",
                    "new_tgt": "天下",
                    "confidence": 0.92,
                    "why": "User requested a different term.",
                }
                for p in pages
            ]
            glossary_updates = [
                {
                    "type": "term",
                    "src_text": "World",
                    "tgt_text": "天下",
                    "locked": False,
                    "aliases_src": [],
                    "aliases_tgt": ["世界"],
                }
            ]

        return {
            "ok": True,
            "data": {
                "decision": "apply",
                "pages": revised,
                "changed_pages": sorted(changed_pages),
                "patches": patches,
                "glossary_updates": glossary_updates,
                "warnings": [],
            },
            "metrics": {"candidate_pages": len(pages), "patches": len(patches), "usage": {"total_tokens": 0}},
            "error": None,
        }

    def run_typeset(self, page, *, skill_params=None):
        self.lettering_calls += 1
        return super().run_typeset(page, skill_params=skill_params)


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

    # First run: VLM -> auto revision -> lettering -> complete
    job = agent.run_until_waiting_or_done(job_id, max_steps=20)
    print("first_status=", job.status.value)
    print(
        "vlm_calls=",
        tools.vlm_calls,
        "lettering_calls=",
        tools.lettering_calls,
        "auto_revision_calls=",
        tools.auto_revision_calls,
        "user_revision_calls=",
        tools.user_revision_calls,
    )
    print("tgt_after_auto_p1=", job.pages[0].regions[1].translated_text)
    print("tgt_after_auto_p2=", job.pages[1].regions[1].translated_text)
    assert job.pages[0].regions[1].translated_text == "原力与你同在"
    assert tools.auto_revision_calls >= 2

    # User correction
    agent.handle_user_message(job_id, "第1页的 World 应该翻译成 天下，请修改所有相关的问题")

    job = store.get_job(job_id)
    action = engine.decide_next_action(job)
    assert action.action_type.value == "run_revision", action
    assert action.page_index == 0, action

    # Second run: explicit user revision -> re-typeset -> complete
    job = agent.run_until_waiting_or_done(job_id, max_steps=20)

    print("second_status=", job.status.value)
    print(
        "vlm_calls=",
        tools.vlm_calls,
        "lettering_calls=",
        tools.lettering_calls,
        "auto_revision_calls=",
        tools.auto_revision_calls,
        "user_revision_calls=",
        tools.user_revision_calls,
    )
    print("tgt_after_user_p1_r2=", job.pages[0].regions[2].translated_text)
    print("tgt_after_user_p2_r2=", job.pages[1].regions[2].translated_text)
    print("glossary_entries=", len((job.glossary or {}).get("entries") or []))
    assert job.pages[0].regions[2].translated_text == "天下"
    assert job.pages[1].regions[2].translated_text == "天下"
    assert tools.user_revision_calls == 1
    print("tail_logs=")
    for line in job.logs[-12:]:
        print(" ", line)


if __name__ == "__main__":
    main()
