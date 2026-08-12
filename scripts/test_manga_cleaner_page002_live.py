from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.lettering_basic_v0.mask_refinement import (
    MaskRefinementConfig,
    constrain_remote_detect_heatmap,
    constrain_remote_detect_mask,
)


DEFAULT_IMAGE = Path("data/test_runs/page002_vlm_agent_live/page_0002/page.png")
DEFAULT_VLM_RESULT = Path("data/test_runs/page002_vlm_agent_live/page_0002/vlm_result.json")
DEFAULT_EXTERNAL_MASK = Path("data/test_runs/inpaint/page002_json_bbox_mask_xyxy_byqwen1/input_mask.png")
DEFAULT_OUTPUT = Path("data/test_runs/manga_cleaner_page002_live")
DEFAULT_URL = "http://100.118.147.113:9171"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _decode_b64(data: str) -> bytes:
    return base64.b64decode(data)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_regions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    regions = data.get("regions")
    return regions if isinstance(regions, list) else []


def _load_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGBA")


def _save_overlay(image_bytes: bytes, mask_bytes: bytes, out_path: Path) -> None:
    image = _load_image(image_bytes)
    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
    red = Image.new("RGBA", image.size, (255, 0, 0, 96))
    transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay = Image.composite(red, transparent, mask)
    merged = Image.alpha_composite(image, overlay)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save(out_path)


def _post_json(
    url: str,
    endpoint: str,
    payload: Dict[str, Any],
    timeout_sec: int,
) -> Dict[str, Any]:
    response = requests.post(f"{url.rstrip('/')}{endpoint}", json=payload, timeout=timeout_sec)
    body = response.json()
    if response.status_code != 200:
        raise RuntimeError(f"{endpoint} status={response.status_code} body={body}")
    if not isinstance(body, dict):
        raise RuntimeError(f"{endpoint} returned_non_dict")
    return body


def _run_detect(
    *,
    url: str,
    image_bytes: bytes,
    output_dir: Path,
    profile: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    payload = {
        "request_id": "page002-detect",
        "image_b64": _b64(image_bytes),
        "profile": profile,
        "timeout_sec": timeout_sec,
        "debug": True,
    }
    started = time.time()
    body = _post_json(url, "/detect-text-mask", payload, timeout_sec)
    elapsed = round(time.time() - started, 3)

    _write_json(output_dir / "detect_request.json", payload)
    _write_json(output_dir / "detect_response.json", body)

    summary: Dict[str, Any] = {
        "ok": bool(body.get("ok")),
        "elapsed_sec": elapsed,
        "backend": body.get("backend"),
        "profile": body.get("profile"),
        "latency_ms": body.get("latency_ms"),
        "warnings": body.get("warnings", []),
        "metrics": body.get("metrics", {}),
    }

    if not body.get("ok") or not body.get("mask_b64"):
        summary["error"] = body.get("error") or "missing_mask_b64"
        _write_json(output_dir / "summary.json", summary)
        raise RuntimeError(f"detect_failed: {summary['error']}")

    mask_bytes = _decode_b64(body["mask_b64"])
    (output_dir / "detected_mask.png").write_bytes(mask_bytes)
    _save_overlay(image_bytes, mask_bytes, output_dir / "detected_overlay.png")

    debug_artifacts = body.get("debug_artifacts") or {}
    if isinstance(debug_artifacts, dict):
        heatmap_b64 = debug_artifacts.get("heatmap_b64")
        if isinstance(heatmap_b64, str) and heatmap_b64:
            (output_dir / "detected_heatmap.png").write_bytes(_decode_b64(heatmap_b64))
        overlay_b64 = debug_artifacts.get("overlay_b64")
        if isinstance(overlay_b64, str) and overlay_b64:
            (output_dir / "detected_overlay_remote.png").write_bytes(_decode_b64(overlay_b64))

    _write_json(output_dir / "summary.json", summary)
    return body


def _run_inpaint(
    *,
    url: str,
    image_bytes: bytes,
    mask_bytes: bytes,
    output_dir: Path,
    profile: str,
    timeout_sec: int,
    request_id: str,
    mask_source: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "request_id": request_id,
        "image_b64": _b64(image_bytes),
        "mask_b64": _b64(mask_bytes),
        "backend": "lama",
        "profile": profile,
        "timeout_sec": timeout_sec,
        "debug": True,
        "mask_source": mask_source,
    }
    started = time.time()
    body = _post_json(url, "/inpaint", payload, timeout_sec)
    elapsed = round(time.time() - started, 3)

    _write_json(output_dir / "inpaint_request.json", payload)
    _write_json(output_dir / "inpaint_response.json", body)

    summary: Dict[str, Any] = {
        "ok": bool(body.get("ok")),
        "elapsed_sec": elapsed,
        "backend": body.get("backend"),
        "profile": body.get("profile"),
        "latency_ms": body.get("latency_ms"),
        "warnings": body.get("warnings", []),
        "metrics": body.get("metrics", {}),
        "mask_source": mask_source,
    }
    if params:
        summary["params"] = params

    if not body.get("ok") or not body.get("image_b64"):
        summary["error"] = body.get("error") or "missing_image_b64"
        _write_json(output_dir / "summary.json", summary)
        raise RuntimeError(f"inpaint_failed: {summary['error']}")

    out_bytes = _decode_b64(body["image_b64"])
    (output_dir / "output_image.png").write_bytes(out_bytes)

    debug_artifacts = body.get("debug_artifacts") or {}
    if isinstance(debug_artifacts, dict):
        mask_used_b64 = debug_artifacts.get("mask_used_b64")
        if isinstance(mask_used_b64, str) and mask_used_b64:
            mask_used = _decode_b64(mask_used_b64)
            (output_dir / "mask_used.png").write_bytes(mask_used)
            _save_overlay(image_bytes, mask_used, output_dir / "mask_used_overlay.png")
        overlay_b64 = debug_artifacts.get("overlay_b64")
        if isinstance(overlay_b64, str) and overlay_b64:
            (output_dir / "overlay_remote.png").write_bytes(_decode_b64(overlay_b64))

    _write_json(output_dir / "summary.json", summary)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Run manga-cleaner live detect+inpaint test on page002.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the Windows service.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Page image path.")
    parser.add_argument("--vlm-result", default=str(DEFAULT_VLM_RESULT), help="VLM result JSON for region constraints.")
    parser.add_argument("--external-mask", default=str(DEFAULT_EXTERNAL_MASK), help="Optional external mask path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output directory.")
    parser.add_argument("--profile", default="balanced", help="Logical profile.")
    parser.add_argument("--timeout-sec", type=int, default=180, help="HTTP timeout.")
    parser.add_argument("--auto-mask-box-expand-px", type=int, default=12, help="Constraint gate expand pixels for boxed text.")
    parser.add_argument("--auto-mask-final-erode-px", type=int, default=1, help="Final erosion applied after constraint masking.")
    parser.add_argument("--heatmap-component-assign-expand-px", type=int, default=8, help="How far VLM bbox may look outward when assigning heatmap seed components.")
    parser.add_argument("--heatmap-safe-expand-px", type=int, default=3, help="Hard safety margin outside bbox for the final heatmap-constrained mask.")
    parser.add_argument("--heatmap-threshold", type=int, default=1, help="Binary threshold for detected heatmap.")
    parser.add_argument("--heatmap-min-component-area-px", type=int, default=12, help="Smallest heatmap component area to keep before region assignment.")
    parser.add_argument("--heatmap-close-px", type=int, default=2, help="Per-region close kernel size for heatmap constrained mask.")
    parser.add_argument("--heatmap-dilate-px", type=int, default=0, help="Per-region dilate size for heatmap constrained mask.")
    parser.add_argument("--heatmap-max-growth-ratio", type=float, default=1.35, help="Maximum allowed area growth ratio from heatmap seed to final region mask.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"image_not_found: {image_path}")
    image_bytes = image_path.read_bytes()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_page.png").write_bytes(image_bytes)

    detect_dir = output_dir / "pure_detect"
    detect_body = _run_detect(
        url=args.url,
        image_bytes=image_bytes,
        output_dir=detect_dir,
        profile=args.profile,
        timeout_sec=args.timeout_sec,
    )

    detected_mask_bytes = _decode_b64(detect_body["mask_b64"])
    detected_mask = np.array(Image.open(io.BytesIO(detected_mask_bytes)).convert("L"), dtype=np.uint8)
    _run_inpaint(
        url=args.url,
        image_bytes=image_bytes,
        mask_bytes=detected_mask_bytes,
        output_dir=output_dir / "pure_detect_inpaint",
        profile=args.profile,
        timeout_sec=args.timeout_sec,
        request_id="page002-inpaint-detect",
        mask_source="manga_cleaner_detect",
        params={"profile": args.profile, "timeout_sec": args.timeout_sec},
    )

    vlm_result_path = Path(args.vlm_result)
    if vlm_result_path.exists():
        regions = _load_regions(vlm_result_path)
        mask_cfg = MaskRefinementConfig(
            auto_mask_box_expand_px=int(args.auto_mask_box_expand_px),
            auto_mask_final_erode_px=int(args.auto_mask_final_erode_px),
            heatmap_threshold=int(args.heatmap_threshold),
            heatmap_component_assign_expand_px=int(args.heatmap_component_assign_expand_px),
            heatmap_safe_expand_px=int(args.heatmap_safe_expand_px),
            heatmap_min_component_area_px=int(args.heatmap_min_component_area_px),
            heatmap_close_px=int(args.heatmap_close_px),
            heatmap_dilate_px=int(args.heatmap_dilate_px),
            heatmap_max_growth_ratio=float(args.heatmap_max_growth_ratio),
        )
        constrained = constrain_remote_detect_mask(
            detected_mask,
            regions,
            image_size=Image.open(io.BytesIO(image_bytes)).size,
            cfg=mask_cfg,
        )
        constrained_mask = (constrained["mask"] > 0).astype(np.uint8) * 255
        constrained_mask_bytes = io.BytesIO()
        Image.fromarray(constrained_mask, mode="L").save(constrained_mask_bytes, format="PNG")
        constrained_dir = output_dir / "constrained_detect"
        constrained_dir.mkdir(parents=True, exist_ok=True)
        (constrained_dir / "mask.png").write_bytes(constrained_mask_bytes.getvalue())
        _save_overlay(image_bytes, constrained_mask_bytes.getvalue(), constrained_dir / "overlay.png")
        gate_bytes = io.BytesIO()
        Image.fromarray(constrained["gate_mask"], mode="L").save(gate_bytes, format="PNG")
        (constrained_dir / "gate_mask.png").write_bytes(gate_bytes.getvalue())
        _save_overlay(image_bytes, gate_bytes.getvalue(), constrained_dir / "gate_overlay.png")
        _write_json(constrained_dir / "summary.json", constrained.get("metrics") or {})
        _run_inpaint(
            url=args.url,
            image_bytes=image_bytes,
            mask_bytes=constrained_mask_bytes.getvalue(),
            output_dir=output_dir / "constrained_detect_inpaint",
            profile=args.profile,
            timeout_sec=args.timeout_sec,
            request_id="page002-inpaint-constrained",
            mask_source="manga_cleaner_detect_constrained",
            params={
                "profile": args.profile,
                "timeout_sec": args.timeout_sec,
                "auto_mask_box_expand_px": int(args.auto_mask_box_expand_px),
                "auto_mask_final_erode_px": int(args.auto_mask_final_erode_px),
            },
        )

        debug_artifacts = detect_body.get("debug_artifacts") or {}
        heatmap_b64 = debug_artifacts.get("heatmap_b64") if isinstance(debug_artifacts, dict) else None
        if isinstance(heatmap_b64, str) and heatmap_b64:
            heatmap_bytes = _decode_b64(heatmap_b64)
            heatmap = np.array(Image.open(io.BytesIO(heatmap_bytes)).convert("L"), dtype=np.uint8)
            heatmap_constrained = constrain_remote_detect_heatmap(
                heatmap,
                regions,
                image_size=Image.open(io.BytesIO(image_bytes)).size,
                cfg=mask_cfg,
            )
            heatmap_dir = output_dir / "heatmap_constrained_detect"
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            (heatmap_dir / "input_heatmap.png").write_bytes(heatmap_bytes)
            heatmap_seed_bytes = io.BytesIO()
            Image.fromarray((heatmap_constrained["seed_mask"] > 0).astype(np.uint8) * 255, mode="L").save(
                heatmap_seed_bytes, format="PNG"
            )
            (heatmap_dir / "seed_mask.png").write_bytes(heatmap_seed_bytes.getvalue())
            _save_overlay(image_bytes, heatmap_seed_bytes.getvalue(), heatmap_dir / "seed_overlay.png")

            heatmap_mask = (heatmap_constrained["mask"] > 0).astype(np.uint8) * 255
            heatmap_mask_bytes = io.BytesIO()
            Image.fromarray(heatmap_mask, mode="L").save(heatmap_mask_bytes, format="PNG")
            (heatmap_dir / "mask.png").write_bytes(heatmap_mask_bytes.getvalue())
            _save_overlay(image_bytes, heatmap_mask_bytes.getvalue(), heatmap_dir / "overlay.png")
            _write_json(heatmap_dir / "summary.json", heatmap_constrained.get("metrics") or {})
            _run_inpaint(
                url=args.url,
                image_bytes=image_bytes,
                mask_bytes=heatmap_mask_bytes.getvalue(),
                output_dir=output_dir / "heatmap_constrained_detect_inpaint",
                profile=args.profile,
                timeout_sec=args.timeout_sec,
                request_id="page002-inpaint-heatmap-constrained",
                mask_source="manga_cleaner_heatmap_constrained",
                params={
                    "profile": args.profile,
                    "timeout_sec": args.timeout_sec,
                    "heatmap_threshold": int(args.heatmap_threshold),
                    "heatmap_component_assign_expand_px": int(args.heatmap_component_assign_expand_px),
                    "heatmap_safe_expand_px": int(args.heatmap_safe_expand_px),
                    "heatmap_min_component_area_px": int(args.heatmap_min_component_area_px),
                    "heatmap_close_px": int(args.heatmap_close_px),
                    "heatmap_dilate_px": int(args.heatmap_dilate_px),
                    "heatmap_max_growth_ratio": float(args.heatmap_max_growth_ratio),
                },
            )

    external_mask_path = Path(args.external_mask)
    if external_mask_path.exists():
        external_mask_bytes = external_mask_path.read_bytes()
        (output_dir / "external_mask.png").write_bytes(external_mask_bytes)
        _save_overlay(image_bytes, external_mask_bytes, output_dir / "external_mask_overlay.png")
        _run_inpaint(
            url=args.url,
            image_bytes=image_bytes,
            mask_bytes=external_mask_bytes,
            output_dir=output_dir / "external_mask_inpaint",
            profile=args.profile,
            timeout_sec=args.timeout_sec,
            request_id="page002-inpaint-external",
            mask_source="external",
            params={"profile": args.profile, "timeout_sec": args.timeout_sec, "external_mask": str(external_mask_path)},
        )

    final_summary = {
        "ok": True,
        "url": args.url,
        "image": str(image_path),
        "external_mask": str(external_mask_path) if external_mask_path.exists() else None,
        "profile": args.profile,
        "timeout_sec": args.timeout_sec,
        "params": {
            "auto_mask_box_expand_px": int(args.auto_mask_box_expand_px),
            "auto_mask_final_erode_px": int(args.auto_mask_final_erode_px),
            "heatmap_threshold": int(args.heatmap_threshold),
            "heatmap_component_assign_expand_px": int(args.heatmap_component_assign_expand_px),
            "heatmap_safe_expand_px": int(args.heatmap_safe_expand_px),
            "heatmap_min_component_area_px": int(args.heatmap_min_component_area_px),
            "heatmap_close_px": int(args.heatmap_close_px),
            "heatmap_dilate_px": int(args.heatmap_dilate_px),
            "heatmap_max_growth_ratio": float(args.heatmap_max_growth_ratio),
        },
    }
    _write_json(output_dir / "summary.json", final_summary)


if __name__ == "__main__":
    main()
