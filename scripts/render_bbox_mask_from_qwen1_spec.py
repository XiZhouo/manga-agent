from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw


DEFAULT_PAGE = Path("data/test_runs/nightwing_vlm_live/page_0002/page.png")
DEFAULT_OUTPUT = Path("data/test_runs/page002_json_bbox_mask_xyxy_byqwen1")
IMAGE_SIZE = (1988, 3057)

ITEMS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "bbox_2d": [136, 166, 477, 296],
        "text_content": "THE FLYBOIZ\nHAVE BEEN INHALING THE\nCHEMICAL-PERVADED DECAY\nOF CHEMO-INFECTED\nCORPSES.",
        "translated_text": "蝇人帮（Flyboiz）一直在吸入受化学药剂感染的尸体散发出的化学腐烂气体。",
    },
    {
        "id": 2,
        "bbox_2d": [786, 126, 1186, 296],
        "text_content": "IT'S ENTERED THEIR\nBLOODSTREAMS VIA DIFFUSION:\nSOAKING THROUGH THE ALVEOLI\nINTO THE CAPILLARIES, CARRIED\nTHROUGH THE VEINS TO MUTATE\nTHEIR ALREADY-MODIFIED\nCELLS.",
        "translated_text": "它通过扩散进入他们的血液：渗透肺泡进入毛细血管，通过静脉输送，变异他们已经被改造过的细胞。",
    },
    {
        "id": 3,
        "bbox_2d": [1426, 376, 1663, 513],
        "text_content": "THIS HAS\nBEEN TURNING THEM\nINTO MONSTERS...\nOR KILLING THEM\nOUTRIGHT.",
        "translated_text": "这一直在把他们变成怪物……或者直接杀死他们。",
    },
    {
        "id": 4,
        "bbox_2d": [1366, 1666, 1586, 1766],
        "text_content": "ALL OF WHICH\nMEANS THAT TO\nSAVE THEM...",
        "translated_text": "所有这些意味着为了救他们……",
    },
    {
        "id": 5,
        "bbox_2d": [1376, 1813, 1576, 1906],
        "text_content": "THEY NEED BLOOD\nTRANSFUSIONS.",
        "translated_text": "他们需要输血。",
    },
]


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _xyxy_to_xywh(box: List[int]) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return [x1, y1, max(1, x2 - x1), max(1, y2 - y1)]


def _render_mask(size: Tuple[int, int], items: List[Dict[str, Any]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for item in items:
        draw.rectangle(tuple(int(v) for v in item["bbox_2d"]), fill=255)
    return mask


def _render_overlay(page: Image.Image, items: List[Dict[str, Any]], *, alpha: float = 0.38) -> Image.Image:
    base = page.convert("RGBA")
    overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (255, 0, 0, int(round(255 * alpha)))
    outline = (255, 255, 64, 255)
    for item in items:
        bbox = tuple(int(v) for v in item["bbox_2d"])
        draw.rectangle(bbox, fill=fill, outline=outline, width=3)
        draw.text((bbox[0] + 4, max(0, bbox[1] - 18)), f"r{item['id']}", fill=(255, 255, 64, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")


def _make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (left.width + right.width, left.height), (20, 20, 20))
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right.convert("RGB"), (left.width, 0))
    return canvas


def _mask_coverage(mask: Image.Image) -> float:
    px = mask.load()
    w, h = mask.size
    count = sum(1 for y in range(h) for x in range(w) if px[x, y] > 0)
    return count / float(w * h)


def main() -> None:
    page_path = DEFAULT_PAGE
    output_dir = DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    if not page_path.exists():
        raise SystemExit(f"page_not_found: {page_path}")

    page = Image.open(page_path).convert("RGB")
    if page.size != IMAGE_SIZE:
        raise SystemExit(f"image_size_mismatch: expected={IMAGE_SIZE} actual={page.size}")

    mask = _render_mask(IMAGE_SIZE, ITEMS)
    overlay = _render_overlay(page, ITEMS)
    compare = _make_side_by_side(mask.convert("RGB"), overlay)

    mask.save(output_dir / "json_mask_on_page002.png")
    overlay.save(output_dir / "json_mask_overlay_page002.png")
    compare.save(output_dir / "json_mask_compare_page002.png")

    _write_json(
        output_dir / "summary.json",
        {
            "page_image": str(page_path),
            "image_size": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
            "coverage": round(_mask_coverage(mask), 6),
            "items": [
                {
                    "id": int(item["id"]),
                    "bbox_xyxy": list(item["bbox_2d"]),
                    "bbox_xywh": _xyxy_to_xywh(item["bbox_2d"]),
                    "text_content": item.get("text_content", ""),
                    "translated_text": item.get("translated_text", ""),
                }
                for item in ITEMS
            ],
            "artifacts": {
                "mask": str(output_dir / "json_mask_on_page002.png"),
                "overlay": str(output_dir / "json_mask_overlay_page002.png"),
                "compare": str(output_dir / "json_mask_compare_page002.png"),
            },
        },
    )


if __name__ == "__main__":
    main()
