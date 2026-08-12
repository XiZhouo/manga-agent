from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore, Tools, JobStatus
from orchestrator.job_io import (
    build_job_id,
    manifest_dir,
    page_skill_dir,
    resolve_job_dir_from_path,
    skill_dir,
    write_json,
)
from orchestrator.memory_store import LocalMemoryStore


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

    # Import lazily so manifest-only/offline tests do not require imaging deps.
    from skills.input_normalization.normalize import normalize_file

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
    pages_dir = skill_dir(job_dir, "normalization") / "pages"
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


def _resolve_job_dir_from_state(job_state) -> Optional[Path]:
    pref = str((getattr(job_state, "user_preferences", {}) or {}).get("job_dir") or "").strip()
    if pref:
        return Path(pref)
    for page in getattr(job_state, "pages", {}).values():
        image_ref = str(getattr(page, "image_ref", "") or "").strip()
        if image_ref:
            return resolve_job_dir_from_path(Path(image_ref))
    return None


def _artifact_ref(job_dir: Optional[Path], raw_path: Optional[str]) -> Optional[str]:
    if not raw_path:
        return None
    p = Path(str(raw_path))
    if job_dir is not None:
        try:
            return str(p.relative_to(job_dir))
        except Exception:
            pass
    return str(p)


def _artifact_item(job_dir: Optional[Path], raw_path: Optional[str], *, missing_reason: Optional[str] = None) -> Dict[str, Any]:
    ref = _artifact_ref(job_dir, raw_path)
    exists = False
    if raw_path:
        try:
            exists = Path(str(raw_path)).exists()
        except Exception:
            exists = False
    item: Dict[str, Any] = {"ref": ref, "exists": bool(exists)}
    if (not exists) and missing_reason:
        item["missing_reason"] = str(missing_reason)
    return item


def _existing_artifact_item(job_dir: Optional[Path], path: Path, *, missing_reason: Optional[str] = None) -> Dict[str, Any]:
    return _artifact_item(job_dir, str(path), missing_reason=missing_reason)


def _page_artifact_missing_reason(name: str, page) -> Optional[str]:
    page_type = str(getattr(page, "page_type", "content") or "content").strip().lower()
    stage = str(getattr(page, "stage", "") or "").strip().lower()
    error_stage = str(getattr(page, "page_error_stage", "") or "").strip().lower()

    if name == "revision_result":
        if getattr(page, "auto_revision_done", False) or getattr(page, "revision_usage", None) or getattr(page, "revision_batch_range", None):
            return None
        return "revision_not_run"

    if name in {"erase_result", "erase_mask_coarse", "erase_mask_refined", "erase_mask_overlay", "erased_image"}:
        if page_type in {"cover", "info"}:
            return "erase_skipped_non_content"
        if stage in {"not_started", "translating", "translated", "revising"}:
            return "erase_not_run"
        if error_stage == "erase":
            return "erase_failed_before_artifact"
        if stage in {"failed", "typesetting", "done"} and not getattr(page, "erase_ok", False):
            return "erase_not_completed"
        return "erase_artifact_missing"

    if name in {"lettering_result", "rendered_image"}:
        if page_type in {"cover", "info"}:
            return "typeset_skipped_non_content"
        if stage in {"not_started", "translating", "translated", "revising"}:
            return "typeset_not_run"
        if error_stage == "erase":
            return "typeset_blocked_by_erase_failure"
        if error_stage == "typeset":
            return "typeset_failed_before_artifact"
        if stage in {"erasing", "erased"}:
            return "typeset_not_run"
        return "typeset_artifact_missing"

    if name == "vlm_result":
        if stage == "not_started":
            return "vlm_not_run"
        return "vlm_artifact_missing"

    if name == "normalized_page":
        return "normalization_artifact_missing"

    return None


def _page_artifact_refs(job_dir: Optional[Path], page) -> Dict[str, Optional[str]]:
    if job_dir is None:
        return {
            "normalized_page": _artifact_item(None, getattr(page, "image_ref", None), missing_reason=_page_artifact_missing_reason("normalized_page", page)),
            "erased_image": _artifact_item(None, getattr(page, "erased_image_ref", None), missing_reason=_page_artifact_missing_reason("erased_image", page)),
            "rendered_image": _artifact_item(None, getattr(page, "rendered_image_ref", None), missing_reason=_page_artifact_missing_reason("rendered_image", page)),
        }

    page_index = int(page.page_index)
    vlm_dir = page_skill_dir(job_dir, "vlm_page_translate", page_index)
    revision_dir = page_skill_dir(job_dir, "vlm_translation_revision", page_index)
    erase_dir = page_skill_dir(job_dir, "lama_eraser", page_index)
    lettering_dir = page_skill_dir(job_dir, "lettering", page_index)

    return {
        "normalized_page": _artifact_item(job_dir, getattr(page, "image_ref", None), missing_reason=_page_artifact_missing_reason("normalized_page", page)),
        "vlm_result": _existing_artifact_item(job_dir, vlm_dir / "result.json", missing_reason=_page_artifact_missing_reason("vlm_result", page)),
        "revision_result": _existing_artifact_item(job_dir, revision_dir / "result.json", missing_reason=_page_artifact_missing_reason("revision_result", page)),
        "erase_result": _existing_artifact_item(job_dir, erase_dir / "result.json", missing_reason=_page_artifact_missing_reason("erase_result", page)),
        "erase_mask_coarse": _existing_artifact_item(job_dir, erase_dir / "mask_coarse.png", missing_reason=_page_artifact_missing_reason("erase_mask_coarse", page)),
        "erase_mask_refined": _existing_artifact_item(job_dir, erase_dir / "mask_refined.png", missing_reason=_page_artifact_missing_reason("erase_mask_refined", page)),
        "erase_mask_overlay": _existing_artifact_item(job_dir, erase_dir / "mask_overlay.png", missing_reason=_page_artifact_missing_reason("erase_mask_overlay", page)),
        "erased_image": _artifact_item(job_dir, getattr(page, "erased_image_ref", None), missing_reason=_page_artifact_missing_reason("erased_image", page)),
        "lettering_result": _existing_artifact_item(job_dir, lettering_dir / "result.json", missing_reason=_page_artifact_missing_reason("lettering_result", page)),
        "rendered_image": _artifact_item(job_dir, getattr(page, "rendered_image_ref", None), missing_reason=_page_artifact_missing_reason("rendered_image", page)),
    }


def _build_run_manifest(job_state) -> Dict[str, Any]:
    job_dir = _resolve_job_dir_from_state(job_state)
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
                    "color": r.color,
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
                "page_type": getattr(page, "page_type", "content"),
                "image_ref": page.image_ref,
                "erased_image_ref": getattr(page, "erased_image_ref", None),
                "rendered_image_ref": getattr(page, "rendered_image_ref", None),
                "skill_param_suggestions": getattr(page, "skill_param_suggestions", None),
                "strategy_suggestions": getattr(page, "strategy_suggestions", None),
                "page_brain": {
                    "summary": getattr(page, "page_brain_summary", None),
                    "last_actionable_summary": getattr(page, "page_brain_last_actionable_summary", None),
                    "confidence": getattr(page, "page_brain_confidence", None),
                    "stage_last": getattr(page, "page_brain_stage_last", None),
                },
                "document_ranges": {
                    "vlm_batch": getattr(page, "vlm_batch_range", None),
                    "revision": getattr(page, "revision_batch_range", None),
                    "vlm_batch_id": getattr(page, "vlm_batch_id", None),
                },
                "stage": page.stage.value,
                "page_error": {
                    "message": getattr(page, "page_error_message", None),
                    "stage": getattr(page, "page_error_stage", None),
                    "code": getattr(page, "page_error_code", None),
                    "details": getattr(page, "page_error_details", None),
                },
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
                "erase_metrics": getattr(page, "erase_metrics", None),
                "lettering_ok": page.lettering_ok,
                "typeset_metrics": getattr(page, "typeset_metrics", None),
                "artifacts": _page_artifact_refs(job_dir, page),
                "regions": regions_out,
                }
            )

    page_stage_counts: Dict[str, int] = {}
    for item in pages_out:
        stage = str(item.get("stage") or "")
        page_stage_counts[stage] = page_stage_counts.get(stage, 0) + 1

    pages_by_index = {
        int(item["page_index"]): item
        for item in pages_out
        if str(item.get("page_index")).strip().isdigit()
    }

    def _pages_for_indices(page_indices: List[int]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for page_index in page_indices:
            page = pages_by_index.get(int(page_index))
            if not page:
                continue
            out.append(
                {
                    "page_index": page["page_index"],
                    "page_type": page.get("page_type"),
                    "stage": page.get("stage"),
                    "rendered_image_ref": page.get("rendered_image_ref"),
                    "erased_image_ref": page.get("erased_image_ref"),
                    "translation_consistent": page.get("translation_consistent"),
                    "auto_revision": {
                        "done": ((page.get("auto_revision") or {}).get("done")),
                        "changed": ((page.get("auto_revision") or {}).get("changed")),
                    },
                }
            )
        return out

    raw_vlm_batches = list(getattr(job_state, "vlm_batches", None) or [])
    vlm_windows = []
    page_to_vlm_window: Dict[str, Any] = {}
    for idx, batch in enumerate(raw_vlm_batches, start=1):
        if not isinstance(batch, dict):
            continue
        page_range = batch.get("page_range") if isinstance(batch.get("page_range"), dict) else {}
        page_indices = [int(x) for x in (page_range.get("page_indices") or []) if str(x).strip().isdigit()]
        pages_in_window = _pages_for_indices(page_indices)
        status_counts: Dict[str, int] = {}
        for page in pages_in_window:
            stage = str(page.get("stage") or "")
            status_counts[stage] = status_counts.get(stage, 0) + 1
            page_to_vlm_window[str(page["page_index"])] = {
                "window_index": idx,
                "label": page_range.get("label"),
                "batch_id": batch.get("batch_id"),
            }
        vlm_windows.append(
            {
                "window_index": idx,
                "batch_id": batch.get("batch_id"),
                "page_range": page_range,
                "page_count": len(page_indices),
                "metrics": batch.get("metrics"),
                "created_at": batch.get("created_at"),
                "status_counts": status_counts,
                "pages": pages_in_window,
            }
        )

    raw_revision_batches = list(getattr(job_state, "revision_batches", None) or [])
    revision_windows = []
    page_to_revision_windows: Dict[str, List[Dict[str, Any]]] = {}
    for idx, batch in enumerate(raw_revision_batches, start=1):
        if not isinstance(batch, dict):
            continue
        page_range = batch.get("page_range") if isinstance(batch.get("page_range"), dict) else {}
        page_indices = [int(x) for x in (page_range.get("page_indices") or []) if str(x).strip().isdigit()]
        changed_pages = [int(x) for x in (batch.get("changed_pages") or []) if str(x).strip().isdigit()]
        pages_in_window = _pages_for_indices(page_indices)
        changed_page_details = _pages_for_indices(changed_pages)
        for page_index in page_indices:
            page_to_revision_windows.setdefault(str(page_index), []).append(
                {
                    "window_index": idx,
                    "label": page_range.get("label"),
                    "revision_mode": batch.get("revision_mode"),
                    "changed": page_index in changed_pages,
                }
            )
        revision_windows.append(
            {
                "window_index": idx,
                "revision_mode": batch.get("revision_mode"),
                "page_range": page_range,
                "candidate_pages": batch.get("candidate_pages"),
                "patch_count": batch.get("patch_count"),
                "warnings": batch.get("warnings"),
                "created_at": batch.get("created_at"),
                "changed_pages": changed_pages,
                "changed_page_details": changed_page_details,
                "pages": pages_in_window,
            }
        )

    return {
        "version": "0.1.0",
        "job_id": job_state.job_id,
        "source_file_name": job_state.source_file_name,
        "target_language": job_state.target_language,
        "created_at": job_state.created_at,
        "updated_at": job_state.updated_at,
        "workflow": {
            "pipeline_mode": getattr(job_state, "user_preferences", {}).get("pipeline_mode"),
            "vlm_batch_size": getattr(job_state, "vlm_batch_size", None),
            "job_brain": {
                "summary": getattr(job_state, "job_brain_summary", None),
                "confidence": getattr(job_state, "job_brain_confidence", None),
                "stage_last": getattr(job_state, "job_brain_stage_last", None),
                "strategy_suggestions": getattr(job_state, "job_strategy_suggestions", None),
            },
        },
        "artifact_roots": {
            "job_dir": str(job_dir) if job_dir is not None else None,
            "skill_outputs_dir": _artifact_ref(job_dir, str(job_dir / "skill_outputs")) if job_dir is not None else None,
            "memory_dir": _artifact_ref(job_dir, str(job_dir / "memory")) if job_dir is not None else None,
            "manifest_dir": _artifact_ref(job_dir, str(job_dir / "manifest")) if job_dir is not None else None,
        },
        "artifacts": {
            "run_manifest": _artifact_item(job_dir, str(job_dir / "manifest" / "run_manifest.json") if job_dir is not None else None),
            "glossary": _artifact_item(job_dir, str(job_dir / "manifest" / "glossary.json") if job_dir is not None else None),
            "rag_bundle": _artifact_item(job_dir, str(job_dir / "manifest" / "rag_bundle.json") if job_dir is not None else None),
            "vlm_batches": _artifact_item(job_dir, str(job_dir / "manifest" / "vlm_batches.json") if job_dir is not None else None),
            "revision_batches": _artifact_item(job_dir, str(job_dir / "manifest" / "revision_batches.json") if job_dir is not None else None),
            "brain_memory": _artifact_item(job_dir, str(job_dir / "memory" / "brain_memory.json") if job_dir is not None else None),
            "translation_memory": _artifact_item(job_dir, str(job_dir / "memory" / "translation_memory.json") if job_dir is not None else None),
            "project_memory": _artifact_item(job_dir, str(job_dir / "memory" / "project_memory.json") if job_dir is not None else None),
            "style_memory": _artifact_item(job_dir, str(job_dir / "memory" / "style_memory.json") if job_dir is not None else None),
            "experience_memory": _artifact_item(job_dir, str(job_dir / "memory" / "experience_memory.json") if job_dir is not None else None),
        },
        "batch_views": {
            "vlm_windows": vlm_windows,
            "revision_windows": revision_windows,
        },
        "page_views": {
            "summary": {
                "page_count": len(pages_out),
                "stage_counts": page_stage_counts,
                "rendered_pages": sum(1 for item in pages_out if item.get("rendered_image_ref")),
                "erased_pages": sum(1 for item in pages_out if item.get("erased_image_ref")),
            },
            "page_to_vlm_window": page_to_vlm_window,
            "page_to_revision_windows": page_to_revision_windows,
            "pages": pages_out,
        },
        # Backward-compatible flat keys for older tooling.
        "job_brain": {
            "summary": getattr(job_state, "job_brain_summary", None),
            "confidence": getattr(job_state, "job_brain_confidence", None),
            "stage_last": getattr(job_state, "job_brain_stage_last", None),
            "strategy_suggestions": getattr(job_state, "job_strategy_suggestions", None),
        },
        "vlm_batch_size": getattr(job_state, "vlm_batch_size", None),
        "vlm_batches": getattr(job_state, "vlm_batches", None),
        "revision_batches": getattr(job_state, "revision_batches", None),
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


def _refresh_manifest_artifact_flags(run_manifest: Dict[str, Any], *, job_dir: Path) -> None:
    def _refresh_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        ref = item.get("ref")
        if ref:
            exists = (job_dir / str(ref)).exists()
            item["exists"] = bool(exists)
            if exists:
                item.pop("missing_reason", None)
        return item

    artifacts = run_manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for key, value in list(artifacts.items()):
            artifacts[key] = _refresh_item(value)

    page_views = run_manifest.get("page_views")
    pages = (page_views or {}).get("pages") if isinstance(page_views, dict) else None
    if isinstance(pages, list):
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_artifacts = page.get("artifacts")
            if isinstance(page_artifacts, dict):
                for key, value in list(page_artifacts.items()):
                    page_artifacts[key] = _refresh_item(value)

    flat_pages = run_manifest.get("pages")
    if isinstance(flat_pages, list):
        for page in flat_pages:
            if not isinstance(page, dict):
                continue
            page_artifacts = page.get("artifacts")
            if isinstance(page_artifacts, dict):
                for key, value in list(page_artifacts.items()):
                    page_artifacts[key] = _refresh_item(value)


def _package_zip(output_zip: Path, *, job_dir: Path, run_manifest: Dict[str, Any], glossary: Dict[str, Any], rag_bundle: Dict[str, Any]) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    skill_outputs_root = job_dir / "skill_outputs"
    memory_dir = job_dir / "memory"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # job artifacts
        zf.writestr("run_manifest.json", json.dumps(run_manifest, ensure_ascii=False, indent=2))
        zf.writestr("glossary.json", json.dumps(glossary, ensure_ascii=False, indent=2))
        zf.writestr("rag_bundle.json", json.dumps(rag_bundle, ensure_ascii=False, indent=2))
        zf.writestr("vlm_batches.json", json.dumps(run_manifest.get("vlm_batches") or [], ensure_ascii=False, indent=2))
        zf.writestr("revision_batches.json", json.dumps(run_manifest.get("revision_batches") or [], ensure_ascii=False, indent=2))

        if skill_outputs_root.exists():
            for p in sorted(skill_outputs_root.rglob("*")):
                if p.is_file():
                    zf.write(p, arcname=f"skill_outputs/{p.relative_to(skill_outputs_root)}")

        if memory_dir.exists():
            for p in sorted(memory_dir.glob("*.json")):
                if p.is_file():
                    zf.write(p, arcname=f"memory/{p.name}")


def _write_artifacts(job_dir: Path, *, run_manifest: Dict[str, Any], glossary: Dict[str, Any], rag_bundle: Dict[str, Any]) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir = manifest_dir(job_dir)
    write_json(out_dir / "glossary.json", glossary)
    write_json(out_dir / "rag_bundle.json", rag_bundle)
    write_json(out_dir / "vlm_batches.json", run_manifest.get("vlm_batches") or [])
    write_json(out_dir / "revision_batches.json", run_manifest.get("revision_batches") or [])

    # Keep root-level copies for backward compatibility with older scripts.
    write_json(job_dir / "glossary.json", glossary)
    write_json(job_dir / "rag_bundle.json", rag_bundle)
    write_json(job_dir / "vlm_batches.json", run_manifest.get("vlm_batches") or [])
    write_json(job_dir / "revision_batches.json", run_manifest.get("revision_batches") or [])

    # Write manifest after sibling artifacts exist so its navigational exists-flags are accurate.
    _refresh_manifest_artifact_flags(run_manifest, job_dir=job_dir)
    manifest_artifacts = run_manifest.get("artifacts")
    if isinstance(manifest_artifacts, dict) and isinstance(manifest_artifacts.get("run_manifest"), dict):
        manifest_artifacts["run_manifest"]["exists"] = True
        manifest_artifacts["run_manifest"].pop("missing_reason", None)
    write_json(out_dir / "run_manifest.json", run_manifest)
    write_json(job_dir / "run_manifest.json", run_manifest)


def _build_pipeline_parser() -> argparse.ArgumentParser:
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
    return parser


def run_pipeline_cli(argv: Optional[List[str]] = None) -> int:
    parser = _build_pipeline_parser()
    args = parser.parse_args(argv)

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
    job_id = build_job_id(input_path.name)
    job_id = agent.create_job(
        user_id=args.user_id,
        source_file_name=input_path.name,
        target_language=args.target_language,
        pages=[(p["page_index"], "") for p in pages],
        job_id=job_id,
        job_dir=str(job_root / job_id),
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
    return 0


def run_chat_cli(*, cwd: Optional[Path] = None) -> int:
    from orchestrator.chat_controller import ChatController

    workdir = (cwd or Path.cwd()).resolve()
    store = InMemoryStateStore()
    agent = ComicAgent(store=store, tools=Tools(), decision_engine=DecisionEngine())
    controller = ChatController(agent=agent, cwd=workdir)

    active_job_id = None
    print("Manga Brain CLI 已启动。输入 /quit 退出。")

    while True:
        try:
            msg = input("\n你> ").strip()
        except EOFError:
            print()
            break
        if not msg:
            continue
        if msg in {"/quit", "/exit"}:
            break

        resp = controller.handle_message(msg, active_job_id=active_job_id)
        print(f"\nBrain> {resp.get('message')}")
        if resp.get("job_id"):
            active_job_id = str(resp["job_id"])

    return 0


def _print_main_help() -> None:
    print(
        "\n".join(
            [
                "Manga Agent",
                "",
                "默认直接启动 Brain 聊天：",
                "  python main.py",
                "",
                "正式批处理运行入口：",
                "  python main.py run --input <comic.cbz> --pipeline-mode vlm --target-language zh",
                "  python run_pipeline.py --input <comic.cbz> --pipeline-mode vlm --target-language zh",
                "",
                "兼容旧用法：",
                "  python main.py --input <comic.cbz> --pipeline-mode vlm --target-language zh",
            ]
        )
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return run_chat_cli()

    if args[0] in {"-h", "--help", "help"}:
        _print_main_help()
        return 0

    if args[0] == "chat":
        return run_chat_cli()

    if args[0] == "run":
        return run_pipeline_cli(args[1:])

    if any(arg in {"--input", "-i"} for arg in args):
        return run_pipeline_cli(args)

    _print_main_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
