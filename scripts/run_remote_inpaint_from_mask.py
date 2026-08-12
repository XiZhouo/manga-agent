from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests


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


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remote inpaint service on a given page image + prepared mask.")
    parser.add_argument("--image", type=str, required=True, help="Input page image path.")
    parser.add_argument("--mask", type=str, required=True, help="Input mask image path (white=erase).")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--url", type=str, default="", help="Optional service base URL override.")
    parser.add_argument("--profile", type=str, default="balanced", help="Remote inpaint profile.")
    parser.add_argument("--timeout-sec", type=int, default=180, help="HTTP timeout seconds.")
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise SystemExit(f"image_not_found: {image_path}")
    if not mask_path.exists():
        raise SystemExit(f"mask_not_found: {mask_path}")

    image_bytes = image_path.read_bytes()
    mask_bytes = mask_path.read_bytes()
    (output_dir / "input_image.png").write_bytes(image_bytes)
    (output_dir / "input_mask.png").write_bytes(mask_bytes)

    url = (args.url or "").strip() or _load_runtime_url()
    payload = {
        "image_b64": _b64(image_bytes),
        "mask_b64": _b64(mask_bytes),
        "backend": "lama",
        "profile": args.profile,
    }

    started_at = time.time()
    summary: Dict[str, Any] = {
        "url": url,
        "image": str(image_path),
        "mask": str(mask_path),
        "profile": args.profile,
        "timeout_sec": args.timeout_sec,
        "params": {
            "profile": args.profile,
            "timeout_sec": args.timeout_sec,
            "url": url,
        },
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
            summary["error"] = body if isinstance(body, dict) else {"message": "non_json_response"}
    except Exception as exc:
        summary["ok"] = False
        summary["elapsed_sec"] = round(time.time() - started_at, 3)
        summary["error"] = {"type": exc.__class__.__name__, "message": str(exc)}

    _write_json(output_dir / "summary.json", summary)

    if not summary.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
