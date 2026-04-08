from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore, Tools, JobStatus
from orchestrator.memory_store import LocalMemoryStore
from skills.input_normalization.normalize import normalize_file


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _infer_file_type(input_path: Path) -> str:
    # returns extension without dot, lower-cased
    return input_path.suffix.lower().lstrip(".")


def _load_pages_from_directory(input_dir: Path, max_pages: Optional[int]) -> List[Dict[str, Any]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if max_pages is not None:
        files = files[: max_pages]

    pages: List[Dict[str, Any]] = []
    for idx, p in enumerate(files):
        pages.append({"page_index": idx, "image_bytes": p.read_bytes(), "filename": p.name})
    return pages


def _normalize_input(input_path: Path, max_pages: Optional[int]) -> List[Dict[str, Any]]:
    if input_path.is_dir():
        return _load_pages_from_directory(input_path, max_pages)

    file_type = _infer_file_type(input_path)
    file_bytes = input_path.read_bytes()
    normalized = normalize_file(file_bytes, file_type)

    pages: List[Dict[str, Any]] = []
    for item in normalized[: (max_pages or len(normalized))]:
        pages.append(
            {
                "page_index": int(item.page_index),
                "image_bytes": item.image_bytes,
                "filename": item.filename,
            }
        )
    return pages


def _write_job_pages(job_dir: Path, pages: List[Dict[str, Any]]) -> List[Tuple[int, str]]:
    pages_dir = job_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    page_refs: List[Tuple[int, str]] = []
    for page in pages:
        page_index = int(page["page_index"])
        filename = str(page.get("filename") or f"page_{page_index:04d}.png")

        ext = Path(filename).suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp"}:
            ext = ".png"

        out_name = f"page_{page_index:04d}{ext}"
        out_path = pages_dir / out_name
        out_path.write_bytes(page["image_bytes"])
        page_refs.append((page_index, str(out_path)))

    return page_refs


def _build_run_manifest(job_state) -> Dict[str, Any]:
    pages_out = []
    for page_index in sorted(job_state.pages.keys()):
        page = job_state.pages[page_index]
        regions_out = []
        for region_id in sorted(page.regions.keys()):
            r = page.regions[region_id]
            regions_out.append(
                {
                    "page_index": page.page_index,
                    "region_id": r.region_id,
                    "bbox": r.bbox,
                    "region_type": r.region_type,
                    "has_text_box": r.has_text_box,
                    "ocr_text": r.ocr_text,
                    "ocr_confidence": r.ocr_confidence,
                    "translated_text": r.translated_text,
                    "translation_confidence": r.translation_confidence,
                    "render_ok": r.render_ok,
                    "render_overflow": r.render_overflow,
                }
            )

        pages_out.append(
            {
                "page_index": page.page_index,
                "image_ref": page.image_ref,
                "erased_image_ref": getattr(page, "erased_image_ref", None),
                "rendered_image_ref": getattr(page, "rendered_image_ref", None),
                "skill_param_suggestions": getattr(page, "skill_param_suggestions", None),
                "stage": page.stage.value,
                "vlm": {
                    "quality_ok": getattr(page, "vlm_quality_ok", None),
                    "quality_reason": getattr(page, "vlm_quality_reason", None),
                    "warnings": getattr(page, "vlm_warnings", None),
                    "downscaled": getattr(page, "vlm_downscaled", None),
                    "orig_size": getattr(page, "vlm_orig_size", None),
                    "sent_size": getattr(page, "vlm_sent_size", None),
                    "usage": getattr(page, "vlm_usage", None),
                    "retry_count_vlm": getattr(page, "retry_count_vlm", None),
                    "judge": {
                        "mode": getattr(page, "vlm_judge_mode", None),
                        "decision": getattr(page, "vlm_judge_decision", None),
                        "reason_code": getattr(page, "vlm_judge_reason_code", None),
                        "explanation": getattr(page, "vlm_judge_explanation", None),
                        "confidence": getattr(page, "vlm_judge_confidence", None),
                        "warnings": getattr(page, "vlm_judge_warnings", None),
                        "usage": getattr(page, "vlm_judge_usage", None),
                    },
                },
                "bbox_error": page.bbox_error,
                "avg_ocr_confidence": page.avg_ocr_confidence,
                "translation_consistent": page.translation_consistent,
                "auto_revision": {
                    "done": getattr(page, "auto_revision_done", None),
                    "changed": getattr(page, "auto_revision_changed", None),
                    "mode_last": getattr(page, "revision_mode_last", None),
                    "warnings": getattr(page, "revision_warnings", None),
                    "usage": getattr(page, "revision_usage", None),
                },
                "erase_ok": getattr(page, "erase_ok", None),
                "erase_backend": getattr(page, "erase_backend", None),
                "lettering_ok": page.lettering_ok,
                "regions": regions_out,
            }
        )

    return {
        "version": "0.1.0",
        "job_id": job_state.job_id,
        "source_file_name": job_state.source_file_name,
        "target_language": job_state.target_language,
        "created_at": job_state.created_at,
        "updated_at": job_state.updated_at,
        "pages": pages_out,
    }


def _default_glossary() -> Dict[str, Any]:
    return {"version": "0.1.0", "entries": [], "updated_at": _utc_iso()}


def _build_glossary(job_state) -> Dict[str, Any]:
    g = getattr(job_state, "glossary", None)
    if isinstance(g, dict) and isinstance(g.get("entries"), list):
        if "version" not in g:
            g = {**g, "version": "0.1.0"}
        if "updated_at" not in g:
            g = {**g, "updated_at": _utc_iso()}
        return g
    return _default_glossary()


def _package_zip(output_zip: Path, *, job_dir: Path, run_manifest: Dict[str, Any], glossary: Dict[str, Any], rag_bundle: Dict[str, Any]) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    pages_dir = job_dir / "pages"
    erased_dir = job_dir / "erased"
    rendered_dir = job_dir / "rendered"
    memory_dir = job_dir / "memory"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # job artifacts
        zf.writestr("run_manifest.json", json.dumps(run_manifest, ensure_ascii=False, indent=2))
        zf.writestr("glossary.json", json.dumps(glossary, ensure_ascii=False, indent=2))
        zf.writestr("rag_bundle.json", json.dumps(rag_bundle, ensure_ascii=False, indent=2))

        # original pages
        if pages_dir.exists():
            for p in sorted(pages_dir.glob("page_*")):
                if p.is_file():
                    zf.write(p, arcname=f"pages/{p.name}")

        # erased pages (cached for re-typesetting)
        if erased_dir.exists():
            for p in sorted(erased_dir.glob("page_*")):
                if p.is_file():
                    zf.write(p, arcname=f"erased/{p.name}")

        # rendered pages (lettered)
        if rendered_dir.exists():
            for p in sorted(rendered_dir.glob("page_*")):
                if p.is_file():
                    zf.write(p, arcname=f"rendered/{p.name}")

        if memory_dir.exists():
            for p in sorted(memory_dir.glob("*.json")):
                if p.is_file():
                    zf.write(p, arcname=f"memory/{p.name}")


def _write_artifacts(job_dir: Path, *, run_manifest: Dict[str, Any], glossary: Dict[str, Any], rag_bundle: Dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (job_dir / "glossary.json").write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (job_dir / "rag_bundle.json").write_text(
        json.dumps(rag_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Manga Translation Agent (orchestrator-driven)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input file (pdf/zip/cbz/cbr/rar/jpg/png) or a directory of images")
    parser.add_argument("--output", "-o", type=str, default="data/output", help="Output directory")
    parser.add_argument("--target-language", type=str, default="zh", help="Target language (v0: zh)")
    parser.add_argument("--reading-direction", type=str, default="ltr", choices=["ltr", "rtl"], help="Reading direction hint (reserved for future script builder)")
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        default="classic",
        choices=["classic", "vlm"],
        help="Pipeline mode: classic (detection+OCR+translation) or vlm (VLM core does all three).",
    )
    parser.add_argument("--max-pages", type=int, default=None, help="Limit pages for quick runs")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Max orchestrator steps. 0 means auto-scale based on page count and pipeline mode.",
    )
    parser.add_argument(
        "--package-draft",
        action="store_true",
        help="Package a ZIP even if the job is not completed (draft/failed/waiting).",
    )
    parser.add_argument("--rag-bundle", type=str, default="", help="Optional path to a prior rag_bundle.json to import before this run.")
    parser.add_argument("--user-id", type=str, default="cli", help="User id")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    # Normalize input -> page images
    try:
        pages = _normalize_input(input_path, max_pages=args.max_pages)
    except Exception as e:
        raise SystemExit(
            "Failed to normalize input. "
            "If input is PDF, ensure Poppler (pdftoppm) is installed; "
            "if input is RAR/CBR, ensure unrar is available. "
            f"Error: {e}"
        )

    if not pages:
        raise SystemExit("No pages produced by input_normalization.")

    # Prepare job folder (gitignored)
    job_root = Path("data/jobs")
    job_root.mkdir(parents=True, exist_ok=True)

    store = InMemoryStateStore()
    tools = Tools()
    decision_engine = DecisionEngine()
    agent = ComicAgent(store=store, tools=tools, decision_engine=decision_engine)
    memory_store = LocalMemoryStore()

    # Create job_id first by creating the job, then write pages under that folder.
    # We create a temporary job with placeholder refs, then rewrite them.
    job_id = agent.create_job(
        user_id=args.user_id,
        source_file_name=input_path.name,
        target_language=args.target_language,
        pages=[(p["page_index"], "") for p in pages],
    )

    job_dir = job_root / job_id
    page_refs = _write_job_pages(job_dir, pages)

    job_state = store.get_job(job_id)
    for page_index, image_ref in page_refs:
        job_state.pages[page_index].image_ref = image_ref
        job_state.pages[page_index].source_filename = Path(image_ref).name
    job_state.user_preferences["job_dir"] = str(job_dir)
    # store the reading direction as a preference for future script builder
    job_state.user_preferences["reading_direction"] = args.reading_direction
    job_state.user_preferences["pipeline_mode"] = args.pipeline_mode
    if args.rag_bundle:
        bundle_path = Path(args.rag_bundle)
        if not bundle_path.exists():
            raise SystemExit(f"RAG bundle not found: {bundle_path}")
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        imported = memory_store.import_rag_bundle(job_state, bundle)
        job_state.logs.append(f"Imported rag bundle from {bundle_path} with counts={imported}")
    store.save_job(job_state)

    # Run orchestrator
    job_state = agent.run_until_waiting_or_done(job_id, max_steps=(args.max_steps or None))

    # Build artifacts
    run_manifest = _build_run_manifest(job_state)
    glossary = _build_glossary(job_state)
    rag_bundle = memory_store.export_rag_bundle(job_state)
    _write_artifacts(job_dir, run_manifest=run_manifest, glossary=glossary, rag_bundle=rag_bundle)

    # Package only when completed (or explicitly requested).
    out_zip: Optional[Path] = None
    if job_state.status == JobStatus.COMPLETED or args.package_draft:
        out_dir = Path(args.output)
        out_zip = out_dir / f"{job_id}.zip"
        _package_zip(out_zip, job_dir=job_dir, run_manifest=run_manifest, glossary=glossary, rag_bundle=rag_bundle)
    else:
        print("job_id:", job_id)
        print("status:", job_state.status.value)
        if job_state.status in {JobStatus.WAITING_USER, JobStatus.FAILED}:
            print("waiting_reason:", job_state.waiting_reason)
        print("zip_skipped: job not completed (use --package-draft to force)")
        print("artifacts_dir:", job_dir)
        raise SystemExit(2)

    print("job_id:", job_id)
    print("status:", job_state.status.value)
    if out_zip is not None:
        print("output_zip:", out_zip)
    if getattr(job_state, "logs", None):
        vlm_line = None
        for line in reversed(job_state.logs):
            if "VLM core done" in line:
                vlm_line = line
                break
        if vlm_line:
            print("vlm_log:", vlm_line)
        else:
            print("last_log:", job_state.logs[-1])
    if job_state.status in {JobStatus.WAITING_USER, JobStatus.FAILED}:
        print("waiting_reason:", job_state.waiting_reason)


if __name__ == "__main__":
    main()
