from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.state import PageState, RegionState
from orchestrator.tools import Tools
from skills.input_normalization.normalize import normalize_file


DEFAULT_INPUT = Path("mangatest") / "Nightwing 124 (2025) (Pyrate-DCP).cbz"
DEFAULT_OUTPUT = Path("data/test_runs/nightwing_skill_stage1")


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


def _append_progress(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live stage-1 skill validation on the Nightwing CBZ.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to the Nightwing CBZ.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT), help="Directory for logs and artifacts.")
    parser.add_argument("--page-index", type=int, default=0, help="Which page to validate.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = output_dir / "progress.log"
    progress_log.write_text("", encoding="utf-8")
    started_at = time.time()

    summary: Dict[str, Any] = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "page_index": args.page_index,
        "steps": {},
        "started_at_epoch": started_at,
    }

    try:
        if not input_path.exists():
            summary["error"] = f"input_not_found: {input_path}"
            _append_progress(progress_log, f"[error] input not found: {input_path}")
            _write_json(output_dir / "summary.json", summary)
            raise SystemExit(2)

        _append_progress(progress_log, "[start] normalize input")
        t0 = time.time()
        pages = normalize_file(input_path.read_bytes(), input_path.suffix.lower().lstrip("."))
        summary["normalized_pages"] = len(pages)
        summary["first_filenames"] = [p.filename for p in pages[:3]]
        summary["steps"]["normalize"] = {"ok": True, "duration_sec": round(time.time() - t0, 3)}
        _append_progress(progress_log, f"[done] normalize input pages={len(pages)} duration={time.time() - t0:.3f}s")
        _write_json(output_dir / "summary.json", summary)

        if args.page_index < 0 or args.page_index >= len(pages):
            summary["error"] = f"page_index_out_of_range: {args.page_index}"
            _append_progress(progress_log, f"[error] page index out of range: {args.page_index}")
            _write_json(output_dir / "summary.json", summary)
            raise SystemExit(2)

        _append_progress(progress_log, f"[start] extract page index={args.page_index}")
        t0 = time.time()
        selected = pages[args.page_index]
        page_image_path = output_dir / f"page_{args.page_index:04d}.png"
        page_image_path.write_bytes(selected.image_bytes)
        image = Image.open(page_image_path)
        summary["page_image"] = {
            "path": str(page_image_path),
            "filename": selected.filename,
            "size": list(image.size),
        }
        summary["steps"]["extract_page"] = {"ok": True, "duration_sec": round(time.time() - t0, 3)}
        _append_progress(progress_log, f"[done] extract page file={selected.filename} duration={time.time() - t0:.3f}s")
        _write_json(output_dir / "summary.json", summary)

        page = PageState(page_index=args.page_index, image_ref=str(page_image_path), source_filename=selected.filename)
        tools = Tools()

        _append_progress(progress_log, "[start] vlm")
        t0 = time.time()
        vlm = tools.run_vlm_page(page)
        _write_json(output_dir / "vlm_result.json", vlm)
        summary["steps"]["vlm"] = {
            "ok": bool(vlm.get("ok")),
            "page_type": vlm.get("page_type"),
            "region_count": len(vlm.get("regions") or []),
            "metrics": vlm.get("vlm_metrics"),
            "error": vlm.get("error"),
            "duration_sec": round(time.time() - t0, 3),
        }
        _append_progress(
            progress_log,
            f"[done] vlm ok={bool(vlm.get('ok'))} regions={len(vlm.get('regions') or [])} duration={time.time() - t0:.3f}s",
        )
        _write_json(output_dir / "summary.json", summary)
        if not vlm.get("ok"):
            raise SystemExit(1)

        page.page_type = str(vlm.get("page_type") or "content")
        for region in vlm.get("regions") or []:
            region_id = int(region["region_id"])
            page.regions[region_id] = RegionState(
                region_id=region_id,
                bbox=region.get("bbox"),
                region_type=region.get("region_type", "text"),
                has_text_box=region.get("has_text_box"),
                color=str(region.get("color") or "black"),
                ocr_text=region.get("src_text"),
                translated_text=region.get("tgt_text"),
            )

        if page.page_type in {"cover", "info"}:
            page.rendered_image_ref = page.image_ref
            summary["steps"]["skip_non_content"] = {
                "ok": True,
                "page_type": page.page_type,
                "rendered_image_ref": page.rendered_image_ref,
            }
            _append_progress(progress_log, f"[done] skip downstream page_type={page.page_type}")
            _write_json(output_dir / "summary.json", summary)
            return

        _append_progress(progress_log, f"[start] erase regions={len(page.regions)}")
        t0 = time.time()
        erase = tools.run_erase(page)
        _write_json(output_dir / "erase_result.json", erase)
        summary["steps"]["erase"] = {
            "ok": bool(erase.get("ok")),
            "backend": erase.get("erase_backend"),
            "metrics": erase.get("metrics"),
            "error": erase.get("error"),
            "erased_image_ref": page.erased_image_ref,
            "duration_sec": round(time.time() - t0, 3),
        }
        _append_progress(
            progress_log,
            f"[done] erase ok={bool(erase.get('ok'))} backend={erase.get('erase_backend')} duration={time.time() - t0:.3f}s",
        )
        _write_json(output_dir / "summary.json", summary)
        if not erase.get("ok"):
            raise SystemExit(1)

        _append_progress(progress_log, f"[start] typeset regions={len(page.regions)}")
        t0 = time.time()
        typeset = tools.run_typeset(page)
        _write_json(output_dir / "typeset_result.json", typeset)
        summary["steps"]["typeset"] = {
            "ok": bool(typeset.get("ok")),
            "metrics": typeset.get("metrics"),
            "error": typeset.get("error"),
            "rendered_image_ref": page.rendered_image_ref,
            "result_count": len(typeset.get("results") or {}),
            "duration_sec": round(time.time() - t0, 3),
        }
        _append_progress(
            progress_log,
            f"[done] typeset ok={bool(typeset.get('ok'))} results={len(typeset.get('results') or {})} duration={time.time() - t0:.3f}s",
        )
        _write_json(output_dir / "summary.json", summary)
        if not typeset.get("ok"):
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:
        summary["error"] = {
            "code": "stage1_exception",
            "message": str(exc),
            "type": exc.__class__.__name__,
        }
        _write_json(
            output_dir / "exception.json",
            {
                "error": summary["error"],
                "traceback": traceback.format_exc(),
            },
        )
        _append_progress(progress_log, f"[exception] {exc.__class__.__name__}: {exc}")
        _write_json(output_dir / "summary.json", summary)
        raise
    finally:
        summary["finished_at_epoch"] = time.time()
        summary["total_duration_sec"] = round(summary["finished_at_epoch"] - started_at, 3)
        _write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
