from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skills.lama_erase_server.server import InpaintRequest, inpaint


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_demo() -> tuple[str, str]:
    image = Image.new("RGB", (96, 64), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 18, 88, 46), fill=(240, 240, 240))
    draw.text((20, 22), "TEXT", fill=(0, 0, 0))

    mask = Image.new("L", (96, 64), 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rectangle((18, 18, 70, 40), fill=255)

    return _png_b64(image), _png_b64(mask)


def main() -> None:
    image_b64, mask_b64 = _build_demo()
    req = InpaintRequest(
        image_b64=image_b64,
        mask_b64=mask_b64,
        backend="opencv",
        profile="balanced",
    )
    resp = inpaint(req)
    assert resp.backend == "opencv"

    out = Image.open(io.BytesIO(base64.b64decode(resp.image_b64))).convert("RGB")
    arr = np.array(out)
    print("backend=", resp.backend)
    print("profile=", resp.profile)
    print("mean_pixel=", round(float(arr.mean()), 2))
    assert arr.shape == (64, 96, 3)


if __name__ == "__main__":
    main()
