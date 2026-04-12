from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore, JobState, PageState
from orchestrator.tools import Tools
from skills.input_normalization.normalize import normalize_file


DEFAULT_INPUT = Path("mangatest") / "Nightwing 124 (2025) (Pyrate-DCP).cbz"
DEFAULT_OUTPUT = Path("data/test_runs/page002_vlm_agent_live")


class CaptureTools(Tools):
    def __init__(self) -> None:
        super().__init__()
        self.last_vlm_result: Dict[str, Any] | None = None

    def run_vlm_page(self, page, **kw):
        result = super().run_vlm_page(page, **kw)
        self.last_vlm_result = result
        return result


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


def _page_to_dict(page: PageState) -> Dict[str, Any]:
    return {
        "page_index": page.page_index,
        "image_ref": page.image_ref,
        "page_type": page.page_type,
        "stage": page.stage.value,
        "retry_count_vlm": page.retry_count_vlm,
        "detection_ok": page.detection_ok,
        "detection_count": page.detection_count,
        "bbox_error": page.bbox_error,
        "vlm_quality_ok": page.vlm_quality_ok,
        "vlm_quality_reason": page.vlm_quality_reason,
        "vlm_warnings": list(page.vlm_warnings or []),
        "vlm_judge_decision": page.vlm_judge_decision,
        "vlm_judge_reason_code": page.vlm_judge_reason_code,
        "vlm_judge_explanation": page.vlm_judge_explanation,
        "vlm_judge_confidence": page.vlm_judge_confidence,
        "vlm_judge_warnings": list(page.vlm_judge_warnings or []),
        "vlm_orig_size": page.vlm_orig_size,
        "vlm_sent_size": page.vlm_sent_size,
        "vlm_downscaled": page.vlm_downscaled,
        "vlm_usage": page.vlm_usage,
        "regions": [
            {
                "region_id": r.region_id,
                "bbox": r.bbox,
                "has_text_box": r.has_text_box,
                "color": r.color,
                "src_text": r.ocr_text,
                "tgt_text": r.translated_text,
            }
            for r in sorted(page.regions.values(), key=lambda x: x.region_id)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live agent-side VLM quality check on one selected page.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to input CBZ.")
    parser.add_argument("--page", type=int, default=2, help="0-based page index to test.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT), help="Directory for logs and outputs.")
    parser.add_argument("--target-language", type=str, default="zh", help="Target language.")
    parser.add_argument("--retry", action="store_true", help="Force retry/full-resolution mode.")
    parser.add_argument("--vlm-timeout-sec", type=float, default=0.0, help="Optional VLM request timeout in seconds.")
    parser.add_argument("--judge-mode", type=str, default="auto", choices=["off", "auto", "on"], help="Temporary VLM judge mode for this test.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"input_not_found: {input_path}")

    if args.vlm_timeout_sec and args.vlm_timeout_sec > 0:
        os.environ["MANGA_AGENT_VLM_TIMEOUT_SECONDS"] = str(args.vlm_timeout_sec)
    os.environ["MANGA_AGENT_VLM_JUDGE_MODE"] = str(args.judge_mode)

    started_at = time.time()
    pages = normalize_file(input_path.read_bytes(), input_path.suffix.lower().lstrip("."))
    if args.page < 0 or args.page >= len(pages):
        raise SystemExit(f"page_index_out_of_range: {args.page}, total={len(pages)}")

    item = pages[args.page]
    page_dir = output_dir / f"page_{args.page:04d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    image_path = page_dir / "page.png"
    image_path.write_bytes(item.image_bytes)

    store = InMemoryStateStore()
    tools = CaptureTools()
    engine = DecisionEngine()
    agent = ComicAgent(store=store, tools=tools, decision_engine=engine)

    job = JobState(
        job_id=f"live-page-{args.page}",
        user_id="local-test",
        source_file_name=input_path.name,
        target_language=args.target_language,
    )
    job.user_preferences["pipeline_mode"] = "vlm"
    job.user_preferences["source_language"] = "en"
    job.user_preferences["reading_direction"] = "ltr"

    page = PageState(page_index=args.page, image_ref=str(image_path), source_filename=item.filename)
    job.pages[args.page] = page

    agent.memory.initialize_job(job)
    agent._do_vlm_page(job, page, target_language=args.target_language, retry=bool(args.retry))

    result = {
        "ok": True,
        "input": str(input_path),
        "page_index": args.page,
        "target_language": args.target_language,
        "retry": bool(args.retry),
        "judge_mode": args.judge_mode,
        "duration_sec": round(time.time() - started_at, 3),
        "page": _page_to_dict(page),
        "job_logs": list(job.logs),
    }

    _write_json(page_dir / "agent_page_state.json", result)
    if tools.last_vlm_result is not None:
        _write_json(page_dir / "vlm_result.json", tools.last_vlm_result)

    summary = {
        "page_index": args.page,
        "ok": True,
        "stage": page.stage.value,
        "page_type": page.page_type,
        "region_count": len(page.regions),
        "vlm_quality_ok": page.vlm_quality_ok,
        "vlm_quality_reason": page.vlm_quality_reason,
        "vlm_warning_count": len(page.vlm_warnings or []),
        "vlm_judge_decision": page.vlm_judge_decision,
        "duration_sec": round(time.time() - started_at, 3),
    }
    _write_json(page_dir / "summary.json", summary)
    _write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
