from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.tools import Tools


class OfflineClassicTools(Tools):
    def run_translation(self, page, target_language: str):
        # Avoid real LangChain translation in offline tests.
        results = {}
        for region_id, region in page.regions.items():
            src = (region.ocr_text or "").strip()
            results[region_id] = {"translated_text": f"[{target_language}] {src}", "confidence": 0.9}
        return {"ok": True, "results": results, "consistent": True}


def main() -> None:
    store = InMemoryStateStore()
    tools = OfflineClassicTools()
    engine = DecisionEngine()
    agent = ComicAgent(store=store, tools=tools, decision_engine=engine)

    pages = [(i, "") for i in range(13)]  # no images -> mock detection
    job_id = agent.create_job(user_id="test", source_file_name="dummy", target_language="zh", pages=pages)

    # Classic pipeline (default) should auto-scale max_steps and finish.
    job = agent.run_until_waiting_or_done(job_id)

    done = sum(1 for p in job.pages.values() if p.stage.value == "done")
    print("status=", job.status.value)
    print("done_pages=", done, "/", len(job.pages))
    if job.status.value != "completed":
        print("waiting_reason=", job.waiting_reason)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
