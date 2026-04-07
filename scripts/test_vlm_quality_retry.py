from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class BadThenGoodTools(Tools):
    """First VLM call returns empty texts (bad quality); second returns good zh."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.force_fullres_seen: list[bool] = []

    def run_vlm_page(self, page, **kw):
        self.calls += 1
        force_fullres = bool(kw.get("force_fullres"))
        self.force_fullres_seen.append(force_fullres)

        if self.calls == 1:
            # Bad: 3 regions but all empty => should trigger quality gate
            return {
                "ok": True,
                "regions": [
                    {"region_id": 1, "bbox": [10, 10, 100, 40], "region_type": "text", "src_text": "", "tgt_text": ""},
                    {"region_id": 2, "bbox": [10, 60, 100, 40], "region_type": "text", "src_text": "", "tgt_text": ""},
                    {"region_id": 3, "bbox": [10, 110, 100, 40], "region_type": "text", "src_text": "", "tgt_text": ""},
                ],
                "bbox_error": False,
                "vlm_metrics": {
                    "orig": (1000, 1000),
                    "sent": (500, 500),
                    "downscaled": True,
                    "usage": {"total_tokens": 123},
                },
                "raw": {"ok": True, "metrics": {"regions": 3, "downscaled": True}},
            }

        return {
            "ok": True,
            "regions": [
                {"region_id": 1, "bbox": [10, 10, 100, 40], "region_type": "text", "src_text": "Hello", "tgt_text": "你好"},
                {"region_id": 2, "bbox": [10, 60, 100, 40], "region_type": "text", "src_text": "World", "tgt_text": "世界"},
                {"region_id": 3, "bbox": [10, 110, 100, 40], "region_type": "text", "src_text": "OK", "tgt_text": "好的"},
            ],
            "bbox_error": False,
            "vlm_metrics": {
                "orig": (1000, 1000),
                "sent": (1000, 1000),
                "downscaled": False,
                "usage": {"total_tokens": 456},
            },
            "raw": {"ok": True, "metrics": {"regions": 3, "downscaled": False}},
        }


def main() -> None:
    store = InMemoryStateStore()
    tools = BadThenGoodTools()
    engine = DecisionEngine()
    agent = ComicAgent(store=store, tools=tools, decision_engine=engine)

    job_id = agent.create_job(
        user_id="test",
        source_file_name="dummy",
        target_language="zh",
        pages=[(0, "missing.png")],
    )

    job = store.get_job(job_id)
    job.user_preferences["pipeline_mode"] = "vlm"
    store.save_job(job)

    job = agent.run_until_waiting_or_done(job_id, max_steps=10)

    print("status=", job.status.value)
    print("vlm_calls=", tools.calls)
    print("force_fullres_seen=", tools.force_fullres_seen)
    print("retry_count_vlm=", job.pages[0].retry_count_vlm)
    print("vlm_quality_ok=", job.pages[0].vlm_quality_ok)
    print("vlm_quality_reason=", job.pages[0].vlm_quality_reason)
    print("region_count=", len(job.pages[0].regions))
    print("tail_logs=")
    for line in job.logs[-8:]:
        print(" ", line)


if __name__ == "__main__":
    main()
