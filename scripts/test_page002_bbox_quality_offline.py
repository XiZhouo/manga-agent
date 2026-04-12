from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator import ComicAgent, DecisionEngine, InMemoryStateStore, JobState, JobStatus, PageStage, PageState, RegionState
from orchestrator.tools import Tools


DEFAULT_IMAGE = Path("data/test_runs/nightwing_vlm_live/page_0002/page.png")
DEFAULT_VLM = Path("data/test_runs/nightwing_vlm_live/page_0002/vlm_result.json")
DEFAULT_OUTPUT = Path("data/test_runs/page002_bbox_quality_offline")


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


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def _load_page(vlm_result_path: Path, image_path: Path) -> PageState:
    raw = json.loads(vlm_result_path.read_text(encoding="utf-8"))
    vm = raw.get("vlm_metrics") or {}
    page = PageState(page_index=int(raw.get("page_index") or 2), image_ref=str(image_path))
    page.page_type = str(raw.get("page_type") or "content")
    page.vlm_orig_size = list(vm.get("orig") or []) or None
    page.vlm_sent_size = list(vm.get("sent") or []) or None
    page.vlm_downscaled = vm.get("downscaled") if vm.get("downscaled") is not None else None
    page.vlm_usage = vm.get("usage") if isinstance(vm.get("usage"), dict) else None

    for item in raw.get("regions") or []:
        rid = int(item["region_id"])
        page.regions[rid] = RegionState(
            region_id=rid,
            bbox=item.get("bbox") or item.get("bbox_xywh"),
            region_type=item.get("region_type", "text"),
            has_text_box=item.get("has_text_box"),
            color=str(item.get("color") or "black"),
            ocr_text=item.get("src_text"),
            translated_text=item.get("tgt_text"),
        )
    return page


def _region_metrics(page: PageState) -> Dict[str, Any]:
    size = page.vlm_orig_size or []
    if not (isinstance(size, list) and len(size) == 2 and all(isinstance(x, int) and x > 0 for x in size)):
        return {"page_size": size, "regions": [], "pairs": []}

    page_w, page_h = int(size[0]), int(size[1])
    page_area = float(page_w * page_h)
    rows: List[Dict[str, Any]] = []
    boxed: List[Dict[str, Any]] = []

    for region in sorted(page.regions.values(), key=lambda r: r.region_id):
        bbox = region.bbox
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x, y, w, h = [int(round(float(v))) for v in bbox]
        if w <= 0 or h <= 0:
            continue
        area = float(w * h)
        src = str(region.ocr_text or "").strip()
        tgt = str(region.translated_text or "").strip()
        char_count = max(len(src), len(tgt))
        chars_per_10k = (char_count * 10000.0 / area) if area > 0 else 0.0
        item = {
            "region_id": region.region_id,
            "bbox": [x, y, w, h],
            "has_text_box": region.has_text_box,
            "char_count": char_count,
            "bbox_area_ratio": round(area / page_area, 6),
            "chars_per_10k": round(chars_per_10k, 4),
            "src_preview": src[:80],
            "tgt_preview": tgt[:80],
        }
        rows.append(item)
        if region.has_text_box is not False:
            boxed.append(item)

    pairs: List[Dict[str, Any]] = []
    for idx in range(len(boxed)):
        for jdx in range(idx + 1, len(boxed)):
            iou = _bbox_iou(boxed[idx]["bbox"], boxed[jdx]["bbox"])
            if iou <= 0:
                continue
            pairs.append(
                {
                    "a": boxed[idx]["region_id"],
                    "b": boxed[jdx]["region_id"],
                    "iou": round(iou, 4),
                }
            )

    pairs.sort(key=lambda x: x["iou"], reverse=True)
    return {"page_size": [page_w, page_h], "regions": rows, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate page_0002 bbox quality against saved VLM result.")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument("--vlm-result", type=str, default=str(DEFAULT_VLM))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--target-language", type=str, default="zh")
    args = parser.parse_args()

    image_path = Path(args.image)
    vlm_result_path = Path(args.vlm_result)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page = _load_page(vlm_result_path, image_path)

    agent = ComicAgent(
        store=InMemoryStateStore(),
        tools=Tools(),
        decision_engine=DecisionEngine(),
    )
    ok, reason, warnings = agent._evaluate_vlm_page_quality(None, page, target_language=args.target_language)  # type: ignore[arg-type]

    page.vlm_quality_ok = ok
    page.vlm_quality_reason = reason
    page.vlm_warnings = list(warnings)
    page.stage = PageStage.TRANSLATED
    page.detection_ok = True
    page.detection_count = len(page.regions)

    job = JobState(
        job_id="offline-page002-bbox-check",
        user_id="local-test",
        source_file_name=image_path.name,
        target_language=args.target_language,
        status=JobStatus.RUNNING,
    )
    job.user_preferences["pipeline_mode"] = "vlm"
    job.pages[page.page_index] = page
    next_action = agent.decision_engine.decide_next_action(job)

    report = {
        "image": str(image_path),
        "vlm_result": str(vlm_result_path),
        "page_index": page.page_index,
        "page_type": page.page_type,
        "vlm_quality_ok": ok,
        "vlm_quality_reason": reason,
        "vlm_warnings": warnings,
        "next_action": {
            "action_type": next_action.action_type.value,
            "page_index": next_action.page_index,
            "reason": next_action.reason,
            "payload": next_action.payload,
        },
        "metrics": _region_metrics(page),
    }
    _write_json(output_dir / "summary.json", report)


if __name__ == "__main__":
    main()
