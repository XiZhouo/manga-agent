from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore
from orchestrator.memory_store import LocalMemoryStore
from orchestrator.tools import Tools


def main() -> None:
    store = InMemoryStateStore()
    agent = ComicAgent(store=store, tools=Tools(), decision_engine=DecisionEngine())
    memory = LocalMemoryStore()

    job_id = agent.create_job(
        user_id="test",
        source_file_name="dummy",
        target_language="zh",
        pages=[(0, "missing0.png")],
    )
    job = store.get_job(job_id)
    resolved = memory.resolve_job_dir(job)
    expected = Path("data/jobs") / job_id
    print("resolved_job_dir=", resolved)
    assert resolved == expected, (resolved, expected)


if __name__ == "__main__":
    main()
