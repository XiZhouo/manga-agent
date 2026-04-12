from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw


SOURCE_CANVAS = (1920, 3072)
DEFAULT_PAGE = Path("data/test_runs/nightwing_vlm_live/page_0002/page.png")
DEFAULT_OUTPUT = Path("data/test_runs/page002_manual_bbox_mask_xyxy")

MANUAL_BOXES: List[Dict[str, Any]] = [
    {"region_id": 1, "bbox_xyxy": [68, 54, 250, 110]},
    {"region_id": 2, "bbox_xyxy": [386, 38, 607, 108]},
    {"region_id": 3, "bbox_xyxy": [695, 118, 847, 178]},
    {"region_id": 4, "bbox_xyxy": [675, 536, 809, 586]},
    {"region_id": 5, "bbox_xyxy": [668, 592, 817, 636]},
]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _xyxy_to_xywh(box: List[int]) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _scale_box_xyxy(box: List[int], *, scale_x: float, scale_y: float, width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    sx1 = int(round(x1 * scale_x))
    sy1 = int(round(y1 * scale_y))
    sx2 = int(round(x2 * scale_x))
    sy2 = int(round(y2 * scale_y))

    sx1 = max(0, min(width - 1, sx1))
    sy1 = max(0, min(height - 1, sy1))
    sx2 = max(sx1 + 1, min(width, sx2))
    sy2 = max(sy1 + 1, min(height, sy2))
    return [sx1, sy1, sx2, sy2]


def _render_mask(size: Tuple[int, int], boxes: List[Dict[str, Any]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for item in boxes:
        draw.rectangle(tuple(int(v) for v in item["bbox_xyxy"]), fill=255)
    return mask


def _render_overlay(page: Image.Image, boxes: List[Dict[str, Any]], *, alpha: float = 0.38) -> Image.Image:
    base = page.convert("RGBA")
    overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (255, 0, 0, int(round(255 * alpha)))
    outline = (255, 255, 64, 255)
    for item in boxes:
        xyxy = tuple(int(v) for v in item["bbox_xyxy"])
        draw.rectangle(xyxy, fill=fill, outline=outline, width=3)
        draw.text((xyxy[0] + 4, max(0, xyxy[1] - 18)), f"r{item['region_id']}", fill=(255, 255, 64, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    if left.height != right.height:
        if left.height > right.height:
            right = right.resize((int(round(right.width * left.height / right.height)), left.height))
        else:
            left = left.resize((int(round(left.width * right.height / left.height)), right.height))

    canvas = Image.new("RGB", (left.width + right.width, left.height), (20, 20, 20))
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right.convert("RGB"), (left.width, 0))
    return canvas


def main() -> None:
    page_path = DEFAULT_PAGE
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    if not page_path.exists():
        raise SystemExit(f"page_not_found: {page_path}")

    source_w, source_h = SOURCE_CANVAS
    page = Image.open(page_path).convert("RGB")
    page_w, page_h = page.size
    scale_x = page_w / float(source_w)
    scale_y = page_h / float(source_h)

    scaled_boxes: List[Dict[str, Any]] = []
    for item in MANUAL_BOXES:
        scaled_boxes.append(
            {
                "region_id": int(item["region_id"]),
                "bbox_xyxy": _scale_box_xyxy(
                    item["bbox_xyxy"],
                    scale_x=scale_x,
                    scale_y=scale_y,
                    width=page_w,
                    height=page_h,
                ),
            }
        )

    raw_mask = _render_mask((source_w, source_h), MANUAL_BOXES)
    scaled_mask = _render_mask((page_w, page_h), scaled_boxes)
    overlay = _render_overlay(page, scaled_boxes)
    side_by_side = _make_side_by_side(scaled_mask.convert("RGB"), overlay)

    raw_mask.save(output_dir / "manual_mask_source_1920x3072.png")
    scaled_mask.save(output_dir / "manual_mask_on_page002.png")
    overlay.save(output_dir / "manual_mask_overlay_page002.png")
    side_by_side.save(output_dir / "manual_mask_compare_page002.png")

    _write_json(
        output_dir / "summary.json",
        {
            "source_canvas": {"width": source_w, "height": source_h},
            "page_image": str(page_path),
            "page_size": {"width": page_w, "height": page_h},
            "scale_to_page": {"x": round(scale_x, 6), "y": round(scale_y, 6)},
            "source_boxes": MANUAL_BOXES,
            "source_boxes_xywh": [
                {"region_id": int(item["region_id"]), "bbox_xywh": _xyxy_to_xywh(item["bbox_xyxy"])}
                for item in MANUAL_BOXES
            ],
            "scaled_boxes": scaled_boxes,
            "scaled_boxes_xywh": [
                {"region_id": int(item["region_id"]), "bbox_xywh": _xyxy_to_xywh(item["bbox_xyxy"])}
                for item in scaled_boxes
            ],
            "artifacts": {
                "raw_mask": str(output_dir / "manual_mask_source_1920x3072.png"),
                "scaled_mask": str(output_dir / "manual_mask_on_page002.png"),
                "overlay": str(output_dir / "manual_mask_overlay_page002.png"),
                "compare": str(output_dir / "manual_mask_compare_page002.png"),
            },
        },
    )


if __name__ == "__main__":
    main()
