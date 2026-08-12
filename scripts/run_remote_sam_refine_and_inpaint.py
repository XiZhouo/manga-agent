from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from PIL import Image


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


def _decode_image_b64(data: str) -> bytes:
    return base64.b64decode(data)


def _load_regions(summary_path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    items = []
    for item in raw.get("items") or []:
        if not isinstance(item, dict):
            continue
        try:
            region_id = int(item.get("id"))
        except Exception:
            continue
        region: Dict[str, Any] = {
            "region_id": region_id,
            "has_text_box": item.get("has_text_box", True),
        }
        bbox_xyxy = item.get("bbox_xyxy")
        bbox_xywh = item.get("bbox_xywh")
        if isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4:
            region["bbox_xyxy"] = bbox_xyxy
        if isinstance(bbox_xywh, list) and len(bbox_xywh) == 4:
            region["bbox_xywh"] = bbox_xywh
        items.append(region)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remote SAM refine-mask then remote inpaint on one page.")
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--coarse-mask", required=True, type=str)
    parser.add_argument("--summary", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--profile", default="balanced", type=str)
    parser.add_argument("--timeout-sec", default=180, type=int)
    parser.add_argument("--sam-timeout-sec", default=120, type=int)
    parser.add_argument("--positive-points", default=4, type=int)
    parser.add_argument("--negative-points", default=8, type=int)
    parser.add_argument("--edge-protect-px", default=10, type=int)
    parser.add_argument("--mask-dilate-px", default=0, type=int)
    args = parser.parse_args()

    image_path = Path(args.image)
    coarse_mask_path = Path(args.coarse_mask)
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise SystemExit(f"image_not_found: {image_path}")
    if not coarse_mask_path.exists():
        raise SystemExit(f"coarse_mask_not_found: {coarse_mask_path}")
    if not summary_path.exists():
        raise SystemExit(f"summary_not_found: {summary_path}")

    url = _load_runtime_url()
    image_bytes = image_path.read_bytes()
    coarse_mask_bytes = coarse_mask_path.read_bytes()
    regions = _load_regions(summary_path)

    (output_dir / "input_image.png").write_bytes(image_bytes)
    (output_dir / "input_coarse_mask.png").write_bytes(coarse_mask_bytes)

    refine_payload = {
        "image_b64": _b64(image_bytes),
        "coarse_mask_b64": _b64(coarse_mask_bytes),
        "regions": regions,
        "profile": args.profile,
        "positive_points_per_region": args.positive_points,
        "negative_points_per_region": args.negative_points,
        "edge_protect_px": args.edge_protect_px,
        "multimask_output": True,
        "mask_dilate_px": args.mask_dilate_px,
    }
    _write_json(output_dir / "refine_request.json", refine_payload)

    started_refine = time.time()
    refine_resp = requests.post(f"{url}/refine-mask", json=refine_payload, timeout=args.sam_timeout_sec)
    refine_elapsed = round(time.time() - started_refine, 3)
    try:
        refine_body = refine_resp.json()
    except Exception:
        refine_body = {"raw_text": refine_resp.text[:4000]}
    _write_json(output_dir / "refine_response.json", refine_body if isinstance(refine_body, dict) else {"body": refine_body})
    if refine_resp.status_code != 200 or not isinstance(refine_body, dict) or not refine_body.get("refined_mask_b64"):
        _write_json(
            output_dir / "summary.json",
            {
                "ok": False,
                "stage": "refine-mask",
                "url": url,
                "status_code": refine_resp.status_code,
                "elapsed_sec": refine_elapsed,
                "error": refine_body,
            },
        )
        raise SystemExit(1)

    refined_mask_bytes = _decode_image_b64(str(refine_body["refined_mask_b64"]))
    (output_dir / "refined_mask.png").write_bytes(refined_mask_bytes)
    if refine_body.get("overlay_b64"):
        (output_dir / "refined_overlay.png").write_bytes(_decode_image_b64(str(refine_body["overlay_b64"])))

    inpaint_payload = {
        "image_b64": _b64(image_bytes),
        "mask_b64": _b64(refined_mask_bytes),
        "backend": "lama",
        "profile": args.profile,
    }
    _write_json(output_dir / "inpaint_request.json", inpaint_payload)

    started_inpaint = time.time()
    inpaint_resp = requests.post(f"{url}/inpaint", json=inpaint_payload, timeout=args.timeout_sec)
    inpaint_elapsed = round(time.time() - started_inpaint, 3)
    try:
        inpaint_body = inpaint_resp.json()
    except Exception:
        inpaint_body = {"raw_text": inpaint_resp.text[:4000]}
    _write_json(output_dir / "inpaint_response.json", inpaint_body if isinstance(inpaint_body, dict) else {"body": inpaint_body})
    if inpaint_resp.status_code != 200 or not isinstance(inpaint_body, dict) or not inpaint_body.get("image_b64"):
        _write_json(
            output_dir / "summary.json",
            {
                "ok": False,
                "stage": "inpaint",
                "url": url,
                "refine_status_code": refine_resp.status_code,
                "refine_elapsed_sec": refine_elapsed,
                "inpaint_status_code": inpaint_resp.status_code,
                "inpaint_elapsed_sec": inpaint_elapsed,
                "refine_metrics": refine_body.get("metrics") if isinstance(refine_body, dict) else None,
                "error": inpaint_body,
            },
        )
        raise SystemExit(1)

    output_image_bytes = _decode_image_b64(str(inpaint_body["image_b64"]))
    (output_dir / "output_image.png").write_bytes(output_image_bytes)

    summary: Dict[str, Any] = {
        "ok": True,
        "url": url,
        "profile": args.profile,
        "image": str(image_path),
        "coarse_mask": str(coarse_mask_path),
        "summary": str(summary_path),
        "refine_status_code": refine_resp.status_code,
        "refine_elapsed_sec": refine_elapsed,
        "refine_backend": refine_body.get("backend"),
        "refine_metrics": refine_body.get("metrics"),
        "inpaint_status_code": inpaint_resp.status_code,
        "inpaint_elapsed_sec": inpaint_elapsed,
        "inpaint_backend": inpaint_body.get("backend"),
        "inpaint_profile": inpaint_body.get("profile"),
    }
    _write_json(output_dir / "summary.json", summary)

    # Save decoded image sizes for quick review.
    refined_mask_img = Image.open(io.BytesIO(refined_mask_bytes))
    output_img = Image.open(io.BytesIO(output_image_bytes))
    _write_json(
        output_dir / "artifacts.json",
        {
            "refined_mask_size": list(refined_mask_img.size),
            "refined_mask_mode": refined_mask_img.mode,
            "output_image_size": list(output_img.size),
            "output_image_mode": output_img.mode,
        },
    )


if __name__ == "__main__":
    main()
