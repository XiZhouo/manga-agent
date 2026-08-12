from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class LamaServerConfig:
    """Config for the HTTP inpainting service.

    Expected deployment:
    - orchestrator/macOS calls this service over HTTP
    - this service runs on Windows + CUDA

    Real LaMa execution supports two runners:
    - `simple_lama`: use the Python package wrapper if installed
    - `saicinpainting_cli`: call the official LaMa repo `bin/predict.py`
    """

    max_image_side: int = 0
    runner: str = "auto"  # auto|simple_lama|saicinpainting_cli
    python_exe: str = "python"
    repo_dir: Optional[str] = None
    model_dir: Optional[str] = None
    device: Optional[str] = None
    tmp_root: Optional[str] = None
    sam_enabled: str = "auto"  # auto|off|on
    sam_checkpoint: Optional[str] = None
    sam_model_type: str = "vit_h"
    sam_device: Optional[str] = None


class InpaintRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded PNG/JPG bytes")
    mask_b64: str = Field(..., description="Base64-encoded PNG bytes; 0=keep, 255=remove")
    backend: str = Field(default="lama", description="lama")
    profile: str = Field(default="balanced", description="conservative|balanced|aggressive")


class InpaintResponse(BaseModel):
    image_b64: str
    backend: str
    profile: str


class RefineMaskRegion(BaseModel):
    region_id: int
    bbox_xywh: Optional[List[int]] = None
    bbox_xyxy: Optional[List[int]] = None
    has_text_box: Optional[bool] = True


class RefineMaskRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded source page image")
    coarse_mask_b64: str = Field(..., description="Base64-encoded heuristic mask bytes")
    regions: List[RefineMaskRegion] = Field(default_factory=list)
    profile: str = Field(default="balanced", description="conservative|balanced|aggressive")
    positive_points_per_region: int = Field(default=4, ge=1, le=12)
    negative_points_per_region: int = Field(default=8, ge=0, le=16)
    edge_protect_px: int = Field(default=10, ge=1, le=64)
    multimask_output: bool = Field(default=True)
    mask_dilate_px: int = Field(default=0, ge=0, le=8)


app = FastAPI(title="LamaEraseServer", version="0.3.0")


def _server_config_from_env() -> LamaServerConfig:
    return LamaServerConfig(
        max_image_side=int(os.getenv("LAMA_SERVER_MAX_IMAGE_SIDE", "0")),
        runner=(os.getenv("LAMA_RUNNER", "auto") or "auto").strip().lower(),
        python_exe=os.getenv("LAMA_PYTHON_EXE", "python"),
        repo_dir=os.getenv("LAMA_REPO_DIR"),
        model_dir=os.getenv("LAMA_MODEL_DIR"),
        device=os.getenv("LAMA_DEVICE"),
        tmp_root=os.getenv("LAMA_TMP_ROOT"),
        sam_enabled=(os.getenv("LAMA_SAM_ENABLED", "auto") or "auto").strip().lower(),
        sam_checkpoint=os.getenv("LAMA_SAM_CHECKPOINT"),
        sam_model_type=(os.getenv("LAMA_SAM_MODEL_TYPE", "vit_h") or "vit_h").strip().lower(),
        sam_device=os.getenv("LAMA_SAM_DEVICE") or os.getenv("LAMA_DEVICE"),
    )


def _b64_to_bytes(s: str) -> bytes:
    try:
        return base64.b64decode(s)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad_base64: {e}")


def _decode_image_bytes(data: bytes) -> np.ndarray:
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="bad_image")
    return img


def _decode_mask_bytes(data: bytes, *, W: int, H: int) -> np.ndarray:
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise HTTPException(status_code=400, detail="bad_mask")
    if m.shape[0] != H or m.shape[1] != W:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8) * 255


def _encode_png(img_bgr: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="encode_failed")
    return bytes(buf)


def _encode_mask_png(mask_u8: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", (mask_u8 > 0).astype(np.uint8) * 255)
    if not ok:
        raise HTTPException(status_code=500, detail="encode_mask_failed")
    return bytes(buf)


@lru_cache(maxsize=1)
def _load_simple_lama():
    # Try the most common import paths.
    try:
        from simple_lama_inpainting import SimpleLama  # type: ignore
    except Exception:
        from simple_lama import SimpleLama  # type: ignore

    try:
        return SimpleLama()
    except TypeError:
        # Some forks expose optional kwargs; keep the fallback broad.
        return SimpleLama


def _inpaint_simple_lama(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    import cv2
    from PIL import Image

    model = _load_simple_lama()
    if callable(model) and not hasattr(model, "__call__"):
        raise RuntimeError("simple_lama_loader_invalid")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb, mode="RGB")
    mask_pil = Image.fromarray(mask, mode="L")

    try:
        out_pil = model(image_pil, mask_pil)
    except TypeError:
        # Some variants return a class instead of an instance from loader fallback.
        out_pil = model()(image_pil, mask_pil)

    out_rgb = np.array(out_pil.convert("RGB"))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def _profile_to_refine(profile: str) -> bool:
    profile2 = (profile or "balanced").strip().lower()
    return profile2 == "aggressive"


def _find_cli_output(outdir: Path, input_name: str) -> Path:
    direct = outdir / input_name
    if direct.exists():
        return direct

    for candidate in sorted(outdir.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        if "_mask" in candidate.name.lower():
            continue
        return candidate

    raise FileNotFoundError("no_output_image_found")


def _run_saicinpainting_cli(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    cfg: LamaServerConfig,
    profile: str,
) -> np.ndarray:
    import cv2

    repo_dir = Path(cfg.repo_dir or "")
    model_dir = Path(cfg.model_dir or "")
    predict_py = repo_dir / "bin" / "predict.py"

    if not cfg.repo_dir or not repo_dir.exists():
        raise RuntimeError("missing_lama_repo_dir")
    if not predict_py.exists():
        raise RuntimeError("missing_predict_py")
    if not cfg.model_dir or not model_dir.exists():
        raise RuntimeError("missing_lama_model_dir")

    with tempfile.TemporaryDirectory(prefix="lama_server_", dir=cfg.tmp_root) as td:
        tmp = Path(td)
        indir = tmp / "input"
        outdir = tmp / "output"
        indir.mkdir(parents=True, exist_ok=True)
        outdir.mkdir(parents=True, exist_ok=True)

        input_name = "page.png"
        image_path = indir / input_name
        mask_path = indir / "page_mask001.png"

        cv2.imwrite(str(image_path), img_bgr)
        cv2.imwrite(str(mask_path), mask)

        cmd = [
            cfg.python_exe,
            str(predict_py),
            f"model.path={model_dir}",
            f"indir={indir}",
            f"outdir={outdir}",
        ]
        if cfg.device:
            cmd.append(f"device={cfg.device}")
        if _profile_to_refine(profile):
            cmd.append("refine=True")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_dir) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise RuntimeError(f"saicinpainting_cli_failed: {tail}")

        out_path = _find_cli_output(outdir, input_name)
        out = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        if out is None:
            raise RuntimeError("saicinpainting_output_decode_failed")
        return out


def _resolve_lama_runner(cfg: LamaServerConfig) -> str:
    runner = (cfg.runner or "auto").strip().lower()
    if runner and runner != "auto":
        return runner

    if cfg.repo_dir and cfg.model_dir:
        return "saicinpainting_cli"

    try:
        import simple_lama_inpainting  # type: ignore  # noqa: F401

        return "simple_lama"
    except Exception:
        try:
            import simple_lama  # type: ignore  # noqa: F401

            return "simple_lama"
        except Exception:
            return "unavailable"


def _resolve_sam_available(cfg: LamaServerConfig) -> str:
    mode = (cfg.sam_enabled or "auto").strip().lower()
    if mode == "off":
        return "disabled"
    if not cfg.sam_checkpoint or not Path(cfg.sam_checkpoint).exists():
        return "missing_checkpoint"
    try:
        import segment_anything  # type: ignore  # noqa: F401

        return "available"
    except Exception:
        return "missing_dependency"


@lru_cache(maxsize=1)
def _load_sam_predictor(sam_checkpoint: str, sam_model_type: str, sam_device: Optional[str]):
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    device = sam_device or ("cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
    sam.to(device=device)
    return SamPredictor(sam)


def _refine_mask_with_sam(
    img_bgr: np.ndarray,
    coarse_mask: np.ndarray,
    *,
    regions: List[RefineMaskRegion],
    req: RefineMaskRequest,
    cfg: LamaServerConfig,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    import cv2

    from skills.lama_erase_server.sam_text_mask_refiner import (
        SAMTextMaskRefineConfig,
        build_region_prompts,
        choose_best_sam_candidate,
        region_bbox_xyxy,
    )

    predictor = _load_sam_predictor(str(cfg.sam_checkpoint), str(cfg.sam_model_type), cfg.sam_device)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    refine_cfg = SAMTextMaskRefineConfig(
        positive_points_per_region=int(req.positive_points_per_region),
        negative_points_per_region=int(req.negative_points_per_region),
        edge_protect_px=int(req.edge_protect_px),
        mask_dilate_px=int(req.mask_dilate_px),
    )

    H, W = coarse_mask.shape[:2]
    merged = np.zeros((H, W), dtype=np.uint8)
    overlay = img_rgb.copy()
    region_metrics: List[Dict[str, Any]] = []

    for region in regions:
        region_dict = region.model_dump() if hasattr(region, "model_dump") else region.dict()
        bbox_xyxy = region_bbox_xyxy(region_dict, width=W, height=H)
        if bbox_xyxy is None:
            continue
        has_text_box = bool(region.has_text_box is not False)
        prompts = build_region_prompts(
            coarse_mask=coarse_mask,
            bbox_xyxy=bbox_xyxy,
            has_text_box=has_text_box,
            cfg=refine_cfg,
        )
        point_coords = np.array(prompts["positive_points"] + prompts["negative_points"], dtype=np.float32)
        point_labels = np.array(prompts["positive_labels"] + prompts["negative_labels"], dtype=np.int32)
        box = np.array(bbox_xyxy, dtype=np.float32)

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=bool(req.multimask_output),
        )

        candidates = [((m > 0).astype(np.uint8) * 255) for m in masks]
        chosen = choose_best_sam_candidate(
            candidate_masks=candidates,
            candidate_scores=list(scores) if scores is not None else None,
            coarse_mask=coarse_mask,
            bbox_xyxy=bbox_xyxy,
            has_text_box=has_text_box,
            cfg=refine_cfg,
        )
        selected = chosen.get("mask")
        if selected is None:
            continue

        if refine_cfg.mask_dilate_px > 0:
            kernel = np.ones((refine_cfg.mask_dilate_px + 1, refine_cfg.mask_dilate_px + 1), np.uint8)
            selected = cv2.dilate(selected, kernel, iterations=1)

        merged = np.maximum(merged, selected.astype(np.uint8))

        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)
        for px, py in prompts["positive_points"]:
            cv2.circle(overlay, (int(px), int(py)), 5, (0, 255, 0), -1)
        for px, py in prompts["negative_points"]:
            cv2.circle(overlay, (int(px), int(py)), 5, (255, 64, 64), -1)

        region_metrics.append(
            {
                "region_id": int(region.region_id),
                "bbox_xyxy": [int(v) for v in bbox_xyxy],
                "has_text_box": has_text_box,
                "positive_points": [[int(x), int(y)] for x, y in prompts["positive_points"]],
                "negative_points": [[int(x), int(y)] for x, y in prompts["negative_points"]],
                "selected_index": int(chosen.get("index") or 0),
                **(chosen.get("metrics") or {}),
            }
        )

    alpha = ((merged > 0).astype(np.float32) * 0.35)[..., None]
    tint = np.zeros_like(overlay, dtype=np.float32)
    tint[..., 1] = 255.0
    overlay = (overlay.astype(np.float32) * (1.0 - alpha)) + (tint * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    metrics = {
        "regions": len(region_metrics),
        "mask_pixels": int(np.count_nonzero(merged)),
        "mask_coverage": round(float(np.count_nonzero(merged)) / float(max(1, H * W)), 6),
        "sam_model_type": cfg.sam_model_type,
        "sam_device": cfg.sam_device,
        "region_metrics": region_metrics,
    }
    return merged, metrics, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


def _inpaint_lama(img_bgr: np.ndarray, mask: np.ndarray, *, cfg: LamaServerConfig, profile: str) -> Tuple[np.ndarray, str]:
    runner = _resolve_lama_runner(cfg)

    if runner == "simple_lama":
        try:
            return _inpaint_simple_lama(img_bgr, mask), "lama:simple_lama"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"simple_lama_failed: {e}")

    if runner == "saicinpainting_cli":
        try:
            return _run_saicinpainting_cli(img_bgr, mask, cfg=cfg, profile=profile), "lama:saicinpainting_cli"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"saicinpainting_cli_failed: {e}")

    if runner == "unavailable":
        if (cfg.runner or "auto").strip().lower() != "auto":
            raise HTTPException(status_code=500, detail="configured_lama_runner_unavailable")
        raise HTTPException(status_code=503, detail="lama_runner_unavailable")

    raise HTTPException(status_code=400, detail=f"unknown_lama_runner: {runner}")


def _inpaint_impl(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    backend: str,
    profile: str,
    cfg: LamaServerConfig,
) -> Tuple[np.ndarray, str]:
    backend2 = (backend or "lama").lower().strip()

    if backend2 == "lama":
        return _inpaint_lama(img_bgr, mask, cfg=cfg, profile=profile)

    raise HTTPException(status_code=400, detail=f"unknown_backend: {backend2}")


@app.get("/health")
def health() -> dict:
    cfg = _server_config_from_env()
    return {
        "ok": True,
        "name": "LamaEraseServer",
        "version": "0.3.0",
        "runner": _resolve_lama_runner(cfg),
        "sam": _resolve_sam_available(cfg),
    }


@app.get("/capabilities")
def capabilities() -> dict:
    cfg = _server_config_from_env()
    return {
        "ok": True,
        "backend": "lama",
        "runner": _resolve_lama_runner(cfg),
        "profiles": ["conservative", "balanced", "aggressive"],
        "mask_refine_modes": ["heuristic", "sam_auto", "sam_required"],
        "sam": {
            "status": _resolve_sam_available(cfg),
            "model_type": cfg.sam_model_type,
            "device": cfg.sam_device,
        },
        "env": {
            "LAMA_RUNNER": cfg.runner,
            "LAMA_REPO_DIR": bool(cfg.repo_dir),
            "LAMA_MODEL_DIR": bool(cfg.model_dir),
            "LAMA_DEVICE": cfg.device,
            "LAMA_SAM_ENABLED": cfg.sam_enabled,
            "LAMA_SAM_CHECKPOINT": bool(cfg.sam_checkpoint),
            "LAMA_SAM_MODEL_TYPE": cfg.sam_model_type,
            "LAMA_SAM_DEVICE": cfg.sam_device,
        },
    }


@app.post("/refine-mask")
def refine_mask(req: RefineMaskRequest) -> dict:
    cfg = _server_config_from_env()
    sam_status = _resolve_sam_available(cfg)
    if sam_status != "available":
        raise HTTPException(status_code=503, detail=f"sam_unavailable:{sam_status}")

    img_bytes = _b64_to_bytes(req.image_b64)
    coarse_bytes = _b64_to_bytes(req.coarse_mask_b64)

    img = _decode_image_bytes(img_bytes)
    H, W = img.shape[:2]
    coarse_mask = _decode_mask_bytes(coarse_bytes, W=W, H=H)

    refined, metrics, overlay = _refine_mask_with_sam(
        img,
        coarse_mask,
        regions=list(req.regions or []),
        req=req,
        cfg=cfg,
    )

    return {
        "ok": True,
        "backend": f"sam:{cfg.sam_model_type}",
        "refined_mask_b64": base64.b64encode(_encode_mask_png(refined)).decode("utf-8"),
        "overlay_b64": base64.b64encode(_encode_png(overlay)).decode("utf-8"),
        "metrics": metrics,
    }


@app.post("/inpaint", response_model=InpaintResponse)
def inpaint(req: InpaintRequest) -> InpaintResponse:
    cfg = _server_config_from_env()

    img_bytes = _b64_to_bytes(req.image_b64)
    mask_bytes = _b64_to_bytes(req.mask_b64)

    img = _decode_image_bytes(img_bytes)
    H, W = img.shape[:2]

    if cfg.max_image_side > 0 and max(W, H) > cfg.max_image_side:
        raise HTTPException(status_code=400, detail="image_too_large")

    mask = _decode_mask_bytes(mask_bytes, W=W, H=H)

    out, actual_backend = _inpaint_impl(
        img,
        mask,
        backend=req.backend,
        profile=req.profile,
        cfg=cfg,
    )

    out_png = _encode_png(out)
    return InpaintResponse(
        image_b64=base64.b64encode(out_png).decode("utf-8"),
        backend=actual_backend,
        profile=(req.profile or "balanced"),
    )
