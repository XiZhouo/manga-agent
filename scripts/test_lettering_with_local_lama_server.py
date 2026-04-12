from __future__ import annotations

import base64
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image


def _start_server() -> None:
    # Import inside thread so the app module is available.
    import uvicorn
    from skills.lama_erase_server import server as lama_server

    lama_server._resolve_lama_runner = lambda cfg: "simple_lama"  # type: ignore[assignment]
    lama_server._inpaint_simple_lama = lambda img_bgr, mask: img_bgr  # type: ignore[assignment]
    os.environ.setdefault("LAMA_RUNNER", "simple_lama")

    uvicorn.run(
        "skills.lama_erase_server.server:app",
        host="127.0.0.1",
        port=8765,
        log_level="warning",
    )


def _make_test_image() -> bytes:
    img = Image.new("RGB", (640, 360), (180, 210, 240))
    # add some background detail
    for x in range(0, 640, 16):
        for y in range(0, 360, 16):
            if (x // 16 + y // 16) % 2 == 0:
                img.putpixel((x, y), (160, 190, 220))

    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont

    d = ImageDraw.Draw(img)
    f = ImageFont.load_default()
    d.text((120, 120), "HELLO", fill=(10, 10, 10), font=f)
    d.text((120, 160), "WORLD", fill=(10, 10, 10), font=f)

    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    # Start local HTTP server with a stubbed simple_lama backend.
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()

    # Wait for server
    import requests

    for _ in range(30):
        try:
            r = requests.get("http://127.0.0.1:8765/health", timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        raise SystemExit("server did not start")

    from skills.lettering_basic_v0.lettering import render_lettered_page

    image_bytes = _make_test_image()
    regions = [
        {"region_id": 1, "bbox_xywh": [90, 100, 220, 120], "translated_text": "你好\n世界"},
        {"region_id": 2, "bbox_xywh": [350, 220, 220, 90], "translated_text": "这是一句很长很长的话，用于测试自动换行与缩小字号"},
    ]

    os.environ["MANGA_AGENT_LAMA_URL"] = "http://127.0.0.1:8765"
    os.environ["MANGA_AGENT_LAMA_BACKEND"] = "lama"

    out = render_lettered_page(image_bytes=image_bytes, regions=regions)
    assert out.get("ok") is True, out

    rendered = out.get("rendered_image_bytes")
    assert isinstance(rendered, (bytes, bytearray))

    out_path = Path("data/output/test_lettering_rendered.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(rendered))

    print("ok=", True)
    print("output=", str(out_path))
    print("region1=", out["results"][1])
    print("region2=", out["results"][2])


if __name__ == "__main__":
    main()
