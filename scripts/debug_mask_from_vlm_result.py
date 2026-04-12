from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skills.lettering_basic_v0.lettering import _cfg_from_env, _mask_cfg_from_lettering
from skills.lettering_basic_v0.mask_refinement import (
    build_page_mask,
    coerce_has_text_box,
    expand_bbox_xywh,
    refine_region_mask,
    safe_bbox_xywh,
)


DEFAULT_IMAGE = Path("data/test_runs/nightwing_vlm_live/page_0002/page.png")
DEFAULT_VLM = Path("data/test_runs/nightwing_vlm_live/page_0002/vlm_result.json")
DEFAULT_OUTPUT = Path("data/test_runs/nightwing_page_0002_mask_debug")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def _save_mask_inverted(mask: np.ndarray, path: Path) -> None:
    inv = np.where(mask > 0, 0, 255).astype(np.uint8)
    Image.fromarray(inv, mode="L").save(path)


def _overlay_mask(base: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    out = base.astype(np.float32).copy()
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.42
    tint = np.zeros_like(out)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    out = out * (1.0 - alpha) + tint * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def _load_regions(vlm_result_path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(vlm_result_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for item in raw.get("regions") or []:
        if not isinstance(item, dict):
            continue
        try:
            region_id = int(item.get("region_id"))
        except Exception:
            continue
        out.append(
            {
                "region_id": region_id,
                "bbox_xywh": item.get("bbox") or item.get("bbox_xywh"),
                "has_text_box": item.get("has_text_box"),
                "color": item.get("color", "black"),
                "src_text": item.get("src_text", ""),
                "tgt_text": item.get("tgt_text", ""),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mask debug artifacts from a saved VLM result.")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    parser.add_argument("--vlm-result", type=str, default=str(DEFAULT_VLM))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    image_path = Path(args.image)
    vlm_result_path = Path(args.vlm_result)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    img_rgb = np.array(image)
    W, H = image.size
    regions = _load_regions(vlm_result_path)

    cfg = _mask_cfg_from_lettering(_cfg_from_env())
    page_mask = build_page_mask(image, regions, cfg=cfg)

    _save_mask(page_mask["coarse_mask"], output_dir / "page_mask_coarse_white.png")
    _save_mask_inverted(page_mask["coarse_mask"], output_dir / "page_mask_coarse_black.png")
    _save_mask(page_mask["mask"], output_dir / "page_mask_refined_white.png")
    _save_mask_inverted(page_mask["mask"], output_dir / "page_mask_refined_black.png")
    Image.fromarray(page_mask["mask_overlay"], mode="RGB").save(output_dir / "page_mask_overlay.png")
    _overlay_mask(img_rgb, page_mask["mask"], color=(255, 0, 0)).save(output_dir / "page_mask_overlay_red.png")

    bbox_canvas = image.copy()
    draw = ImageDraw.Draw(bbox_canvas)

    region_debug: List[Dict[str, Any]] = []
    regions_dir = output_dir / "regions"
    regions_dir.mkdir(parents=True, exist_ok=True)

    for r in regions:
        region_id = int(r["region_id"])
        bbox = r.get("bbox_xywh") or r.get("bbox")
        xywh = safe_bbox_xywh(bbox, W, H)
        if not xywh:
            continue
        boxed = coerce_has_text_box(r.get("has_text_box")) is not False
        expand_px = cfg.box_expand_px if boxed else cfg.embedded_box_expand_px
        x, y, w, h = expand_bbox_xywh(xywh, W, H, expand_px)
        roi = img_rgb[y : y + h, x : x + w]
        refined = refine_region_mask(roi, has_text_box=boxed, cfg=cfg)

        draw.rectangle((xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]), outline=(255, 64, 64), width=3)
        draw.text((xywh[0] + 4, max(0, xywh[1] - 18)), f"r{region_id}", fill=(255, 64, 64))
        draw.rectangle((x, y, x + w, y + h), outline=(64, 255, 255), width=1)

        region_dir = regions_dir / f"region_{region_id:02d}"
        region_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(roi, mode="RGB").save(region_dir / "roi.png")
        _save_mask(refined["coarse_mask"], region_dir / "coarse_white.png")
        _save_mask_inverted(refined["coarse_mask"], region_dir / "coarse_black.png")
        _save_mask(refined["refined_mask"], region_dir / "refined_white.png")
        _save_mask_inverted(refined["refined_mask"], region_dir / "refined_black.png")
        _overlay_mask(roi, refined["refined_mask"]).save(region_dir / "overlay_red.png")

        region_debug.append(
            {
                "region_id": region_id,
                "bbox_xywh": list(xywh),
                "expanded_bbox_xywh": [x, y, w, h],
                "has_text_box": boxed,
                "src_text": r.get("src_text"),
                "tgt_text": r.get("tgt_text"),
                "metrics": refined.get("metrics") or {},
            }
        )

    bbox_canvas.save(output_dir / "page_bboxes.png")
    _write_json(
        output_dir / "summary.json",
        {
            "image": str(image_path),
            "vlm_result": str(vlm_result_path),
            "page_size": [W, H],
            "region_count": len(regions),
            "page_metrics": page_mask.get("metrics") or {},
            "regions": region_debug,
        },
    )


if __name__ == "__main__":
    main()
