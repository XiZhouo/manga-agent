from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skills.lettering_basic_v0.lettering import typeset_page_image
from skills.lettering_basic_v0.mask_refinement import MaskRefinementConfig, build_page_mask


def _make_demo_page() -> bytes:
    img = Image.new("RGB", (360, 220), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Boxed speech region
    draw.ellipse((20, 20, 170, 120), fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    draw.text((55, 55), "HELLO", fill=(0, 0, 0))

    # Art-integrated title region
    draw.rectangle((210, 20, 340, 120), fill=(90, 90, 90))
    draw.text((225, 58), "BOOM", fill=(240, 240, 240))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    image_bytes = _make_demo_page()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    regions = [
        {"region_id": 1, "bbox_xywh": [20, 20, 150, 100], "has_text_box": True, "translated_text": "你好"},
        {"region_id": 2, "bbox_xywh": [210, 20, 130, 100], "has_text_box": False, "translated_text": "轰"},
    ]

    mask_bundle = build_page_mask(pil_img, regions, cfg=MaskRefinementConfig())
    print("mask_pixels=", mask_bundle["metrics"]["mask_pixels"])
    print("embedded_regions=", mask_bundle["metrics"]["embedded_regions"])
    assert mask_bundle["metrics"]["embedded_regions"] == 1
    assert mask_bundle["metrics"]["mask_pixels"] > 0

    rendered = typeset_page_image(image_bytes=image_bytes, regions=regions)
    assert rendered["ok"], rendered
    print("render_mode_r1=", rendered["results"][1]["render_params"]["render_mode"])
    print("render_mode_r2=", rendered["results"][2]["render_params"]["render_mode"])
    assert rendered["results"][1]["render_params"]["render_mode"] == "bubble_fit"
    assert rendered["results"][2]["render_params"]["render_mode"] == "art_embed"


if __name__ == "__main__":
    main()
