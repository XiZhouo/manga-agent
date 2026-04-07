from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


def _has_cjk(s: str) -> bool:
    for ch in s:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            return True
    return False


class NonObviousBadThenGoodTools(Tools):
    """First VLM call is subtly wrong (zh target but English tgt_text, only 2 regions).

    Heuristics are permissive for n<3, so this should pass the heuristic gate.
    The text-only judge should request a retry, then second VLM call returns good zh.

    No network calls are made: the judge is stubbed locally.
    """

    def __init__(self) -> None:
        super().__init__()
        self.vlm_calls = 0
        self.judge_calls = 0

    def run_vlm_page(self, page, **kw):
        self.vlm_calls += 1

        if self.vlm_calls == 1:
            return {
                "ok": True,
                "regions": [
                    {"region_id": 1, "bbox": [10, 10, 100, 40], "region_type": "text", "src_text": "Hello", "tgt_text": "Hello"},
                    {"region_id": 2, "bbox": [10, 60, 100, 40], "region_type": "text", "src_text": "World", "tgt_text": "World"},
                ],
                "bbox_error": False,
                "vlm_metrics": {
                    "orig": (1988, 3057),
                    "sent": (1988, 3057),
                    "downscaled": False,
                    "usage": {"total_tokens": 111},
                },
                "raw": {"ok": True, "metrics": {"regions": 2, "downscaled": False}},
            }

        return {
            "ok": True,
            "regions": [
                {"region_id": 1, "bbox": [10, 10, 100, 40], "region_type": "text", "src_text": "Hello", "tgt_text": "你好"},
                {"region_id": 2, "bbox": [10, 60, 100, 40], "region_type": "text", "src_text": "World", "tgt_text": "世界"},
            ],
            "bbox_error": False,
            "vlm_metrics": {
                "orig": (1988, 3057),
                "sent": (1988, 3057),
                "downscaled": False,
                "usage": {"total_tokens": 222},
            },
            "raw": {"ok": True, "metrics": {"regions": 2, "downscaled": False}},
        }

    def run_vlm_judge_text_only(self, *, page_index, target_language, reading_direction, regions, metrics=None, warnings=None):
        self.judge_calls += 1

        # Minimal local "judge": if zh target and no CJK appears, request retry.
        if target_language == "zh":
            nonempty = [r for r in regions if (r.get("tgt_text") or "").strip()]
            if nonempty:
                cjk = sum(1 for r in nonempty if _has_cjk((r.get("tgt_text") or "").strip()))
                if cjk == 0:
                    return {
                        "ok": True,
                        "data": {
                            "decision": "retry",
                            "reason_code": "target_language_mismatch_small_n",
                            "explanation": "Target language is zh but translations contain no CJK; retry once.",
                            "confidence": 0.85,
                            "warnings": [],
                        },
                        "metrics": {"usage": {"total_tokens": 0}},
                        "error": None,
                    }

        return {
            "ok": True,
            "data": {
                "decision": "proceed",
                "reason_code": "mock_proceed",
                "explanation": "Looks acceptable.",
                "confidence": 0.8,
                "warnings": [],
            },
            "metrics": {"usage": {"total_tokens": 0}},
            "error": None,
        }


def main() -> None:
    store = InMemoryStateStore()
    tools = NonObviousBadThenGoodTools()
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

    page = job.pages[0]
    print("status=", job.status.value)
    print("vlm_calls=", tools.vlm_calls)
    print("judge_calls=", tools.judge_calls)
    print("retry_count_vlm=", page.retry_count_vlm)
    print("vlm_quality_ok=", page.vlm_quality_ok)
    print("vlm_quality_reason=", page.vlm_quality_reason)
    print("vlm_judge_decision=", page.vlm_judge_decision)
    print("region_count=", len(page.regions))
    print("tail_logs=")
    for line in job.logs[-10:]:
        print(" ", line)


if __name__ == "__main__":
    main()
