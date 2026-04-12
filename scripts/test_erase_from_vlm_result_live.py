from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skills.lettering_basic_v0.lettering import erase_page_image


DEFAULT_IMAGE = Path("data/test_runs/nightwing_vlm_live/page_0002/page.png")
DEFAULT_VLM = Path("data/test_runs/nightwing_vlm_live/page_0002/vlm_result.json")
DEFAULT_OUTPUT = Path("data/test_runs/nightwing_page_0002_erase_live")


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


def _load_regions(vlm_result_path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(vlm_result_path.read_text(encoding="utf-8"))
    regions_in = raw.get("regions") or []
    regions: List[Dict[str, Any]] = []
    for item in regions_in:
        if not isinstance(item, dict):
            continue
        try:
            region_id = int(item.get("region_id"))
        except Exception:
            continue
        regions.append(
            {
                "region_id": region_id,
                "bbox_xywh": item.get("bbox") or item.get("bbox_xywh"),
                "has_text_box": item.get("has_text_box"),
                "color": item.get("color", "black"),
                "translated_text": item.get("tgt_text") or "",
            }
        )
    return regions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run erase skill using a saved page image + VLM result.")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="Input page image path.")
    parser.add_argument("--vlm-result", type=str, default=str(DEFAULT_VLM), help="VLM JSON result path.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT), help="Directory for outputs.")
    args = parser.parse_args()

    image_path = Path(args.image)
    vlm_result_path = Path(args.vlm_result)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "image": str(image_path),
        "vlm_result": str(vlm_result_path),
        "output_dir": str(output_dir),
        "started_at_epoch": time.time(),
    }

    if not image_path.exists():
        summary["error"] = f"image_not_found: {image_path}"
        _write_json(output_dir / "summary.json", summary)
        raise SystemExit(2)
    if not vlm_result_path.exists():
        summary["error"] = f"vlm_result_not_found: {vlm_result_path}"
        _write_json(output_dir / "summary.json", summary)
        raise SystemExit(2)

    image_bytes = image_path.read_bytes()
    regions = _load_regions(vlm_result_path)
    summary["region_count"] = len(regions)
    summary["page_type"] = json.loads(vlm_result_path.read_text(encoding="utf-8")).get("page_type")

    started = time.time()
    result = erase_page_image(image_bytes=image_bytes, regions=regions)
    summary["duration_sec"] = round(time.time() - started, 3)
    summary["ok"] = bool(result.get("ok"))
    summary["metrics"] = result.get("metrics")
    summary["error"] = result.get("error")

    if isinstance(result.get("erased_image_bytes"), (bytes, bytearray)):
        (output_dir / "erased.png").write_bytes(bytes(result["erased_image_bytes"]))
    if isinstance(result.get("coarse_mask_bytes"), (bytes, bytearray)):
        (output_dir / "mask_coarse.png").write_bytes(bytes(result["coarse_mask_bytes"]))
    if isinstance(result.get("mask_bytes"), (bytes, bytearray)):
        (output_dir / "mask_refined.png").write_bytes(bytes(result["mask_bytes"]))
    if isinstance(result.get("mask_overlay_bytes"), (bytes, bytearray)):
        (output_dir / "mask_overlay.png").write_bytes(bytes(result["mask_overlay_bytes"]))

    _write_json(output_dir / "erase_result.json", result)
    summary["finished_at_epoch"] = time.time()
    summary["total_duration_sec"] = round(summary["finished_at_epoch"] - summary["started_at_epoch"], 3)
    _write_json(output_dir / "summary.json", summary)

    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
