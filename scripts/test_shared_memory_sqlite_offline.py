from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore, Tools
from orchestrator.state import RegionState


def main() -> None:
    db_path = Path(tempfile.gettempdir()) / "agent1_shared_memory_test.db"
    if db_path.exists():
        db_path.unlink()
    os.environ["MANGA_AGENT_SHARED_MEMORY_DB"] = str(db_path)

    root = Path(tempfile.mkdtemp(prefix="agent1_shared_mem_"))
    img1 = root / "page1.png"
    img2 = root / "page2.png"
    img3 = root / "page3.png"
    for p in (img1, img2, img3):
        p.write_bytes(b"fake")

    agent = ComicAgent(store=InMemoryStateStore(), tools=Tools(), decision_engine=DecisionEngine())

    job1_id = agent.create_job(
        user_id="u",
        source_file_name="Batman 001 (2025).cbz",
        target_language="zh",
        pages=[(0, str(img1))],
        job_dir=str(root / "job1"),
    )
    job1 = agent.store.get_job(job1_id)
    job1.pages[0].regions = {1: RegionState(region_id=1, ocr_text="Nightwing", translated_text="奈特温")}
    agent.memory.record_revision_updates(
        job1,
        page_index_to_regions={0: {1: job1.pages[0].regions[1]}},
        patches=[{"page_index": 0, "region_id": 1, "new_tgt": "夜翼", "expected_old_tgt": "奈特温"}],
        glossary_updates=[{"type": "name", "src_text": "Nightwing", "tgt_text": "夜翼", "locked": True}],
        glossary_type="name",
    )
    agent._append_experience(
        job1,
        kind="successful_strategy",
        note="Balanced erase works well for boxed dialogue pages.",
        situation={"stage": "erase", "page_type": "content"},
        recommended_action="continue",
        skill_params={"profile": "balanced", "mask_dilate_px": 2},
        tags=["erase", "success"],
        confidence=0.7,
    )

    job2_id = agent.create_job(
        user_id="u",
        source_file_name="Batman 002 (2025).cbz",
        target_language="zh",
        pages=[(0, str(img2))],
        job_dir=str(root / "job2"),
    )
    job2 = agent.store.get_job(job2_id)
    job2.pages[0].regions = {1: RegionState(region_id=1, ocr_text="Nightwing", translated_text="")}
    rev2 = agent.memory.build_revision_retrieval(job2, pages=[job2.pages[0]])
    tech2 = agent.memory.build_technical_context(
        job2,
        user_message="page_strategy:after_vlm_page",
        page_summary={
            "stage": "translated",
            "page_type": "content",
            "page_error_stage": "",
            "page_error_code": "",
            "page_error_message": "",
            "embedded_ratio": 0.1,
            "empty_tgt_regions": 0,
            "overflow_regions": 0,
            "bad_render_regions": 0,
            "has_erased_image": False,
            "erase_backend": "",
            "vlm_qa_status": "good",
            "vlm_qa": {"issues": []},
            "vlm_qa_retry_hint": {},
            "vlm_qa_issue_count": 0,
            "erase_qa_status": "",
            "erase_qa_retry_hint": {},
            "erase_qa_issue_count": 0,
            "erase_mask_quality": "",
            "erase_mask_quality_reason": "",
            "erase_mask_retry_hint": {},
            "erase_service_error": {},
            "erase_service_unavailable": False,
            "typeset_qa_status": "",
            "typeset_qa_retry_hint": {},
            "typeset_qa_issue_count": 0,
            "auto_revision_done": False,
            "vlm_quality_ok": True,
            "vlm_quality_reason": "",
            "vlm_judge_decision": "",
            "vlm_judge_reason_code": "",
        },
    )

    job3_id = agent.create_job(
        user_id="u",
        source_file_name="Superman 001 (2025).cbz",
        target_language="zh",
        pages=[(0, str(img3))],
        job_dir=str(root / "job3"),
    )
    job3 = agent.store.get_job(job3_id)
    job3.pages[0].regions = {1: RegionState(region_id=1, ocr_text="Nightwing", translated_text="")}
    rev3 = agent.memory.build_revision_retrieval(job3, pages=[job3.pages[0]])

    glossary_job2 = [(x.get("src_text"), x.get("tgt_text")) for x in rev2.get("glossary_hits", [])[:3]]
    tm_job2 = [(x.get("src_text"), x.get("tgt_text")) for x in rev2.get("translation_memory_hits", [])[:3]]
    glossary_job3 = [(x.get("src_text"), x.get("tgt_text")) for x in rev3.get("glossary_hits", [])[:3]]
    tm_job3 = [(x.get("src_text"), x.get("tgt_text")) for x in rev3.get("translation_memory_hits", [])[:3]]
    proactive = tech2.get("proactive_success_params", {}).get("skill_param_suggestions", {})

    assert ("Nightwing", "夜翼") in glossary_job2, f"job2 glossary miss: {glossary_job2}"
    assert ("Nightwing", "夜翼") in tm_job2, f"job2 TM miss: {tm_job2}"
    assert proactive.get("lama_eraser", {}).get("profile") == "balanced", f"job2 proactive miss: {proactive}"
    assert glossary_job3 == [], f"job3 glossary should be isolated: {glossary_job3}"
    assert tm_job3 == [], f"job3 TM should be isolated: {tm_job3}"

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT memory_type, scope_type, scope_value, source_text, target_text, recommended_action, repeat_count, use_count, last_used_at
            FROM memory_entries
            ORDER BY memory_type, source_text
            """
        ).fetchall()
        usage_rows = conn.execute(
            """
            SELECT route, reason, job_id, page_index, COUNT(*)
            FROM memory_usage_log
            GROUP BY route, reason, job_id, page_index
            ORDER BY route, reason
            """
        ).fetchall()
    finally:
        conn.close()

    assert any(row[0] == "glossary" and int(row[7] or 0) >= 1 for row in rows), f"glossary use_count missing: {rows}"
    assert any(row[0] == "translation" and int(row[7] or 0) >= 1 for row in rows), f"translation use_count missing: {rows}"
    assert any(row[0] == "experience" and int(row[7] or 0) >= 1 for row in rows), f"experience use_count missing: {rows}"
    assert any(str(row[0]) == "revision" for row in usage_rows), f"revision usage missing: {usage_rows}"
    assert any(str(row[0]) == "technical_decision" for row in usage_rows), f"technical_decision usage missing: {usage_rows}"

    print("DB_PATH", db_path)
    print("ROWS", rows)
    print("USAGE_ROWS", usage_rows)
    print("JOB2_GLOSSARY", glossary_job2)
    print("JOB2_TM", tm_job2)
    print("JOB2_PROACTIVE", proactive)
    print("JOB3_GLOSSARY", glossary_job3)
    print("JOB3_TM", tm_job3)
    print("OK shared SQLite memory works")


if __name__ == "__main__":
    main()
