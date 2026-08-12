from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.state import PageState
from orchestrator.tools import Tools
from skills.input_normalization.normalize import normalize_file


DEFAULT_INPUT = Path("mangatest") / "Nightwing 124 (2025) (Pyrate-DCP).cbz"
DEFAULT_OUTPUT = Path("data/test_runs/translate/nightwing_vlm_live")


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, tuple):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (bytes, bytearray)):
        return f"<bytes:{len(obj)}>"
    return obj


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _compact_vlm_result(page_index: int, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page_index": page_index,
        "page_type": result.get("page_type"),
        "original_image_size": result.get("original_image_size"),
        "sent_image_size": result.get("sent_image_size"),
        "coordinate_space": result.get("coordinate_space"),
        "bbox_error": bool(result.get("bbox_error", False)),
        "regions": result.get("regions") or [],
    }


def _debug_vlm_result(page_index: int, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "page_index": page_index,
        "ok": bool(result.get("ok")),
        "error": result.get("error"),
        "vlm_metrics": result.get("vlm_metrics"),
        "raw": result.get("raw"),
    }


def _append_progress(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def _parse_page_indices(raw: str) -> List[int]:
    out: List[int] = []
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return sorted(dict.fromkeys(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM-only live checks on selected Nightwing pages.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to input CBZ.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT), help="Directory for logs and outputs.")
    parser.add_argument("--pages", type=str, default="0,1,2,3", help="Comma-separated page indices.")
    parser.add_argument("--vlm-timeout-sec", type=float, default=0.0, help="Optional VLM request timeout in seconds.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    progress_log = output_dir / "progress.log"
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    progress_log.write_text("", encoding="utf-8")

    if not input_path.exists():
        raise SystemExit(f"input_not_found: {input_path}")

    if args.vlm_timeout_sec and args.vlm_timeout_sec > 0:
        os.environ["MANGA_AGENT_VLM_TIMEOUT_SECONDS"] = str(args.vlm_timeout_sec)

    selected_indices = _parse_page_indices(args.pages)
    started_at = time.time()

    _append_progress(progress_log, "[start] normalize input")
    pages = normalize_file(input_path.read_bytes(), input_path.suffix.lower().lstrip("."))
    _append_progress(progress_log, f"[done] normalize input pages={len(pages)}")

    tools = Tools()
    summary: Dict[str, Any] = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "selected_pages": selected_indices,
        "normalized_pages": len(pages),
        "results": [],
        "started_at_epoch": started_at,
    }

    for page_index in selected_indices:
        if page_index < 0 or page_index >= len(pages):
            summary["results"].append(
                {
                    "page_index": page_index,
                    "ok": False,
                    "error": f"page_index_out_of_range: {page_index}",
                }
            )
            continue

        item = pages[page_index]
        page_dir = output_dir / f"page_{page_index:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)
        image_path = page_dir / "page.png"
        image_path.write_bytes(item.image_bytes)

        page = PageState(page_index=page_index, image_ref=str(image_path), source_filename=item.filename)
        _append_progress(progress_log, f"[start] vlm page={page_index}")
        t0 = time.time()
        result = tools.run_vlm_page(page)
        duration_sec = round(time.time() - t0, 3)
        _write_json(page_dir / "vlm_result.json", _compact_vlm_result(page_index, result))
        _write_json(page_dir / "vlm_debug.json", _debug_vlm_result(page_index, result))

        page_summary = {
            "page_index": page_index,
            "filename": item.filename,
            "ok": bool(result.get("ok")),
            "page_type": result.get("page_type"),
            "region_count": len(result.get("regions") or []),
            "duration_sec": duration_sec,
            "vlm_metrics": result.get("vlm_metrics"),
            "regions_preview": [
                {
                    "region_id": r.get("region_id"),
                    "has_text_box": r.get("has_text_box"),
                    "color": r.get("color"),
                    "src_text": str(r.get("src_text") or "")[:80],
                    "tgt_text": str(r.get("tgt_text") or "")[:80],
                }
                for r in (result.get("regions") or [])[:8]
            ],
            "error": result.get("error"),
        }
        summary["results"].append(page_summary)
        _append_progress(
            progress_log,
            f"[done] vlm page={page_index} ok={page_summary['ok']} page_type={page_summary['page_type']} "
            f"regions={page_summary['region_count']} duration={duration_sec:.3f}s",
        )
        _write_json(page_dir / "summary.json", page_summary)
        _write_json(output_dir / "summary.json", summary)

    summary["finished_at_epoch"] = time.time()
    summary["total_duration_sec"] = round(summary["finished_at_epoch"] - started_at, 3)
    _write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
