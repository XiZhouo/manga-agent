from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
from PIL import Image, ImageDraw


DEFAULT_OUTPUT = Path("data/test_runs/remote_inpaint_live")


def _load_runtime_url() -> str:
    env_url = os.getenv("MANGA_AGENT_LAMA_URL")
    if env_url:
        return env_url.rstrip("/")

    config_path = Path(__file__).resolve().parents[1] / "config" / "runtime.local.json"
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        lama = data.get("lama") or {}
        if isinstance(lama, dict) and lama.get("url"):
            return str(lama["url"]).rstrip("/")

    return "http://127.0.0.1:8765"


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _build_demo() -> Tuple[bytes, bytes]:
    image = Image.new("RGB", (256, 160), (240, 245, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 236, 140), outline=(120, 140, 160), width=2)
    draw.rectangle((40, 48, 216, 108), fill=(255, 255, 255))
    draw.text((66, 64), "TEXT", fill=(20, 20, 20))

    mask = Image.new("L", image.size, 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.rectangle((58, 56, 150, 92), fill=255)

    return _png_bytes(image), _png_bytes(mask)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the remote inpainting service.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT), help="Directory for artifacts.")
    parser.add_argument("--profile", type=str, default="balanced", help="Inpainting profile.")
    parser.add_argument("--timeout-sec", type=int, default=180, help="HTTP timeout.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = _load_runtime_url()
    image_bytes, mask_bytes = _build_demo()
    (output_dir / "input_image.png").write_bytes(image_bytes)
    (output_dir / "input_mask.png").write_bytes(mask_bytes)

    payload = {
        "image_b64": _b64(image_bytes),
        "mask_b64": _b64(mask_bytes),
        "backend": "lama",
        "profile": args.profile,
    }

    started_at = time.time()
    summary: Dict[str, Any] = {
        "url": url,
        "profile": args.profile,
        "timeout_sec": args.timeout_sec,
        "started_at_epoch": started_at,
    }

    try:
        resp = requests.post(f"{url}/inpaint", json=payload, timeout=args.timeout_sec)
        summary["status_code"] = resp.status_code
        summary["elapsed_sec"] = round(time.time() - started_at, 3)

        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text[:4000]}

        _write_json(output_dir / "response.json", body if isinstance(body, dict) else {"body": body})

        if resp.status_code == 200 and isinstance(body, dict) and body.get("image_b64"):
            out_bytes = base64.b64decode(body["image_b64"])
            (output_dir / "output_image.png").write_bytes(out_bytes)
            summary["ok"] = True
            summary["backend"] = body.get("backend")
            summary["runner_profile"] = body.get("profile")
            summary["latency_ms"] = body.get("latency_ms")
        else:
            summary["ok"] = False
            if isinstance(body, dict):
                summary["error"] = body.get("error") or body
            else:
                summary["error"] = {"message": "non_json_response"}
    except Exception as exc:
        summary["ok"] = False
        summary["elapsed_sec"] = round(time.time() - started_at, 3)
        summary["error"] = {"type": exc.__class__.__name__, "message": str(exc)}

    _write_json(output_dir / "summary.json", summary)

    if not summary.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
