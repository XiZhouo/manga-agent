from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore, JobState, PageState, Tools
from skills.input_normalization.normalize import normalize_file


DEFAULT_INPUT = Path("mangatest") / "Nightwing 124 (2025) (Pyrate-DCP).cbz"
DEFAULT_OUTPUT = Path("data/test_runs/pipeline/live")


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, tuple):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_pages(raw: str) -> List[int]:
    out: List[int] = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return sorted(dict.fromkeys(out))


def _page_to_dict(page: PageState) -> Dict[str, Any]:
    return {
        "page_index": page.page_index,
        "page_type": page.page_type,
        "stage": page.stage.value,
        "image_ref": page.image_ref,
        "erased_image_ref": page.erased_image_ref,
        "rendered_image_ref": page.rendered_image_ref,
        "detection_count": page.detection_count,
        "vlm_quality_ok": page.vlm_quality_ok,
        "vlm_quality_reason": page.vlm_quality_reason,
        "vlm_warnings": list(page.vlm_warnings or []),
        "auto_revision_done": page.auto_revision_done,
        "auto_revision_changed": page.auto_revision_changed,
        "erase_ok": page.erase_ok,
        "erase_backend": page.erase_backend,
        "erase_metrics": page.erase_metrics,
        "lettering_ok": page.lettering_ok,
        "typeset_metrics": page.typeset_metrics,
        "qa_reports": page.qa_reports,
        "regions": [
            {
                "region_id": r.region_id,
                "bbox_xyxy": r.bbox_xyxy,
                "has_text_box": r.has_text_box,
                "color": r.color,
                "src_text": r.ocr_text,
                "tgt_text": r.translated_text,
                "placement_source": r.placement_source,
                "placement_lines": r.placement_lines,
                "render_ok": r.render_ok,
                "render_overflow": r.render_overflow,
            }
            for r in sorted(page.regions.values(), key=lambda x: x.region_id)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live VLM -> erase -> lettering pipeline on selected pages.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input CBZ path.")
    parser.add_argument("--pages", type=str, default="2", help="Comma-separated 0-based page indices.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT), help="Output directory.")
    parser.add_argument("--target-language", type=str, default="zh", help="Target language.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional agent max steps override.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = output_dir / "progress.log"
    progress_log.write_text("", encoding="utf-8")

    if not input_path.exists():
        raise SystemExit(f"input_not_found: {input_path}")

    started = time.time()
    pages = normalize_file(input_path.read_bytes(), input_path.suffix.lower().lstrip("."))
    selected = _parse_pages(args.pages)
    if not selected:
        raise SystemExit("no_pages_selected")

    def _log(msg: str) -> None:
        with progress_log.open("a", encoding="utf-8") as fh:
            fh.write(msg.rstrip() + "\n")

    store = InMemoryStateStore()
    engine = DecisionEngine()
    agent = ComicAgent(store=store, tools=Tools(), decision_engine=engine)

    job = JobState(
        job_id=f"live-pipeline-{'-'.join(str(x) for x in selected)}-{int(started)}",
        user_id="local-test",
        source_file_name=input_path.name,
        target_language=args.target_language,
    )
    job.user_preferences["pipeline_mode"] = "vlm"
    job.user_preferences["source_language"] = "en"
    job.user_preferences["reading_direction"] = "ltr"

    for page_index in selected:
        if page_index < 0 or page_index >= len(pages):
            raise SystemExit(f"page_index_out_of_range: {page_index}, total={len(pages)}")
        item = pages[page_index]
        page_dir = output_dir / "pages" / f"page_{page_index:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        image_path = page_dir / "page.png"
        image_path.write_bytes(item.image_bytes)
        job.pages[page_index] = PageState(page_index=page_index, image_ref=str(image_path), source_filename=item.filename)

    agent.memory.initialize_job(job)
    store.create_job(job)
    _log(f"[start] job={job.job_id} selected_pages={selected}")

    final_job = agent.run_until_waiting_or_done(job.job_id, max_steps=(args.max_steps or None))
    _log(f"[done] status={final_job.status.value} waiting_reason={final_job.waiting_reason}")

    summary = {
        "job_id": final_job.job_id,
        "status": final_job.status.value,
        "waiting_reason": final_job.waiting_reason,
        "selected_pages": selected,
        "duration_sec": round(time.time() - started, 3),
        "vlm_batches": final_job.vlm_batches,
        "revision_batches": final_job.revision_batches,
        "pages": {
            str(pi): _page_to_dict(page)
            for pi, page in sorted(final_job.pages.items(), key=lambda kv: kv[0])
        },
        "logs": list(final_job.logs),
    }
    _write_json(output_dir / "summary.json", summary)

    for page_index, page in final_job.pages.items():
        _write_json(output_dir / "pages" / f"page_{page_index:04d}" / "page_state.json", _page_to_dict(page))

    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
