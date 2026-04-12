from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw


DEFAULT_PAGE = Path("data/test_runs/nightwing_vlm_live/page_0002/page.png")
DEFAULT_OUTPUT = Path("data/test_runs/page002_json_bbox_mask_xyxy_compact")

SPEC: Dict[str, Any] = {
    "image_size": [1988, 3057],
    "items": [
        {
            "id": 1,
            "bbox": [202, 172, 470, 345],
            "source_text": "THE FLYBOIZ HAVE BEEN INHALING THE CHEMICAL-PERVADED DECAY OF CHEMO-INFECTED CORPSES.",
            "translated_text_zh": "那些 Flyboiz 一直在吸入被化学物质浸染、受化疗感染尸体腐烂后产生的气体。",
        },
        {
            "id": 2,
            "bbox": [813, 58, 1158, 339],
            "source_text": "IT'S ENTERED THEIR BLOODSTREAMS VIA DIFFUSION: SOAKING THROUGH THE ALVEOLI INTO THE CAPILLARIES, CARRIED THROUGH THE VEINS TO MUTATE THEIR ALREADY-MODIFIED CELLS.",
            "translated_text_zh": "它已经通过扩散进入了他们的血液循环：先穿过肺泡渗入毛细血管，再经由静脉输送，去突变他们那些本就已经被改造过的细胞。",
        },
        {
            "id": 3,
            "bbox": [1400, 370, 1680, 531],
            "source_text": "THIS HAS BEEN TURNING THEM INTO MONSTERS... OR KILLING THEM OUTRIGHT.",
            "translated_text_zh": "这一直在把他们变成怪物……或者干脆直接杀死他们。",
        },
        {
            "id": 4,
            "bbox": [1361, 1678, 1616, 1757],
            "source_text": "ALL OF WHICH MEANS THAT TO SAVE THEM...",
            "translated_text_zh": "而这一切意味着，想救他们的话……",
        },
        {
            "id": 5,
            "bbox": [1349, 1833, 1600, 1914],
            "source_text": "THEY NEED BLOOD TRANSFUSIONS.",
            "translated_text_zh": "他们需要输血。",
        },
    ],
}


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
        draw.rectangle(tuple(int(v) for v in item["bbox"]), fill=255)
    return mask


def _render_overlay(page: Image.Image, items: List[Dict[str, Any]], *, alpha: float = 0.38) -> Image.Image:
    base = page.convert("RGBA")
    overlay = Image.new("RGBA", page.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    fill = (255, 0, 0, int(round(255 * alpha)))
    outline = (255, 255, 64, 255)
    for item in items:
        bbox = tuple(int(v) for v in item["bbox"])
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
    page_w, page_h = page.size
    spec_w, spec_h = [int(v) for v in SPEC["image_size"]]
    if (page_w, page_h) != (spec_w, spec_h):
        raise SystemExit(f"image_size_mismatch: spec={(spec_w, spec_h)} page={(page_w, page_h)}")

    mask = _render_mask((spec_w, spec_h), SPEC["items"])
    overlay = _render_overlay(page, SPEC["items"])
    compare = _make_side_by_side(mask.convert("RGB"), overlay)

    mask.save(output_dir / "json_mask_on_page002.png")
    overlay.save(output_dir / "json_mask_overlay_page002.png")
    compare.save(output_dir / "json_mask_compare_page002.png")

    items_xywh = [
        {
            "id": int(item["id"]),
            "bbox_xyxy": list(item["bbox"]),
            "bbox_xywh": _xyxy_to_xywh(item["bbox"]),
            "source_text": item.get("source_text", ""),
            "translated_text_zh": item.get("translated_text_zh", ""),
        }
        for item in SPEC["items"]
    ]

    _write_json(
        output_dir / "summary.json",
        {
            "page_image": str(page_path),
            "image_size": {"width": spec_w, "height": spec_h},
            "coverage": round(_mask_coverage(mask), 6),
            "items": items_xywh,
            "artifacts": {
                "mask": str(output_dir / "json_mask_on_page002.png"),
                "overlay": str(output_dir / "json_mask_overlay_page002.png"),
                "compare": str(output_dir / "json_mask_compare_page002.png"),
            },
        },
    )


if __name__ == "__main__":
    main()
