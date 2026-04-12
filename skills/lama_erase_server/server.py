from __future__ import annotations

import base64
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

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


class InpaintRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded PNG/JPG bytes")
    mask_b64: str = Field(..., description="Base64-encoded PNG bytes; 0=keep, 255=remove")
    backend: str = Field(default="lama", description="lama")
    profile: str = Field(default="balanced", description="conservative|balanced|aggressive")


class InpaintResponse(BaseModel):
    image_b64: str
    backend: str
    profile: str


app = FastAPI(title="LamaEraseServer", version="0.2.0")


def _server_config_from_env() -> LamaServerConfig:
    return LamaServerConfig(
        max_image_side=int(os.getenv("LAMA_SERVER_MAX_IMAGE_SIDE", "0")),
        runner=(os.getenv("LAMA_RUNNER", "auto") or "auto").strip().lower(),
        python_exe=os.getenv("LAMA_PYTHON_EXE", "python"),
        repo_dir=os.getenv("LAMA_REPO_DIR"),
        model_dir=os.getenv("LAMA_MODEL_DIR"),
        device=os.getenv("LAMA_DEVICE"),
        tmp_root=os.getenv("LAMA_TMP_ROOT"),
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
        "version": "0.2.0",
        "runner": _resolve_lama_runner(cfg),
    }


@app.get("/capabilities")
def capabilities() -> dict:
    cfg = _server_config_from_env()
    return {
        "ok": True,
        "backend": "lama",
        "runner": _resolve_lama_runner(cfg),
        "profiles": ["conservative", "balanced", "aggressive"],
        "env": {
            "LAMA_RUNNER": cfg.runner,
            "LAMA_REPO_DIR": bool(cfg.repo_dir),
            "LAMA_MODEL_DIR": bool(cfg.model_dir),
            "LAMA_DEVICE": cfg.device,
        },
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
