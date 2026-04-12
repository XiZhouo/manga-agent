from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import _build_run_manifest
from orchestrator import InMemoryStateStore, PageStage, ComicAgent, DecisionEngine
from scripts.test_job_level_vlm_batching_offline import JobBatchBrain, JobBatchTools


def main() -> None:
    store = InMemoryStateStore()
    tools = JobBatchTools()
    agent = ComicAgent(store=store, tools=tools, decision_engine=DecisionEngine(), brain=JobBatchBrain())

    job_id = agent.create_job(
        user_id="test",
        source_file_name="dummy.cbz",
        target_language="zh",
        pages=[(0, "missing0.png"), (1, "missing1.png"), (2, "missing2.png"), (3, "missing3.png")],
    )
    job = store.get_job(job_id)
    job.user_preferences["pipeline_mode"] = "vlm"
    job.vlm_batches = [
        {
            "batch_id": "vlm:0-3:1",
            "page_range": {"start_page_index": 0, "end_page_index": 3, "page_count": 4, "page_indices": [0, 1, 2, 3], "label": "0-3"},
            "metrics": {"page_count": 4},
            "created_at": "2026-04-09T00:00:00+00:00",
        }
    ]
    job.revision_batches = [
        {
            "revision_mode": "auto_rag_revision",
            "page_range": {"start_page_index": 0, "end_page_index": 3, "page_count": 4, "page_indices": [0, 1, 2, 3], "label": "0-3"},
            "candidate_pages": 4,
            "changed_pages": [0, 2],
            "patch_count": 2,
            "warnings": [],
            "created_at": "2026-04-09T00:01:00+00:00",
        }
    ]
    for pi, page in job.pages.items():
        page.stage = PageStage.DONE if pi < 3 else PageStage.TRANSLATED
        page.vlm_batch_range = {"start_page_index": 0, "end_page_index": 3, "page_count": 4, "page_indices": [0, 1, 2, 3], "label": "0-3"}
        page.revision_batch_range = {"start_page_index": 0, "end_page_index": 3, "page_count": 4, "page_indices": [0, 1, 2, 3], "label": "0-3"}
        page.rendered_image_ref = f"rendered/page_{pi:04d}.png" if pi < 3 else None
        page.erased_image_ref = f"erased/page_{pi:04d}.png"
        page.auto_revision_done = True
        page.auto_revision_changed = pi in {0, 2}
        page.translation_consistent = True

    manifest = _build_run_manifest(job)
    print("workflow_keys=", sorted((manifest.get("workflow") or {}).keys()))
    print("batch_view_keys=", sorted((manifest.get("batch_views") or {}).keys()))
    print("page_view_keys=", sorted((manifest.get("page_views") or {}).keys()))
    print("vlm_window_count=", len((manifest.get("batch_views") or {}).get("vlm_windows") or []))
    print("revision_window_count=", len((manifest.get("batch_views") or {}).get("revision_windows") or []))
    print("page_to_vlm_window=", (manifest.get("page_views") or {}).get("page_to_vlm_window"))

    workflow = manifest.get("workflow") or {}
    batch_views = manifest.get("batch_views") or {}
    page_views = manifest.get("page_views") or {}

    assert workflow.get("pipeline_mode") == "vlm"
    assert "job_brain" in workflow
    assert len(batch_views.get("vlm_windows") or []) == 1
    assert len(batch_views.get("revision_windows") or []) == 1
    assert (batch_views.get("vlm_windows") or [])[0].get("status_counts", {}).get("done") == 3
    assert (batch_views.get("revision_windows") or [])[0].get("changed_pages") == [0, 2]
    assert page_views.get("summary", {}).get("page_count") == 4
    assert page_views.get("summary", {}).get("rendered_pages") == 3
    assert (page_views.get("page_to_vlm_window") or {}).get("0", {}).get("label") == "0-3"
    assert len(page_views.get("pages") or []) == 4


if __name__ == "__main__":
    main()
