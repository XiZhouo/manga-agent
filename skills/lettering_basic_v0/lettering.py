from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from skills.lettering_basic_v0.mask_refinement import (
    MaskRefinementConfig,
    build_page_mask,
    coerce_has_text_box,
    safe_bbox_xywh,
)


@dataclass(frozen=True)
class LetteringConfig:
    lama_url: str = "http://127.0.0.1:8765"
    lama_backend: str = "lama"
    erase_profile: str = "balanced"
    lama_timeout: float = 60.0

    # Mask extraction heuristics
    mask_min_coverage: float = 0.01
    mask_max_coverage: float = 0.25
    mask_dilate_px: int = 2
    box_expand_px: int = 1
    embedded_mask_min_coverage: float = 0.02
    embedded_mask_max_coverage: float = 0.45
    embedded_mask_dilate_px: int = 4
    embedded_box_expand_px: int = 6
    complete_mask_close_px: int = 3
    embedded_complete_mask_close_px: int = 7
    complete_mask_dilate_px: int = 0
    embedded_complete_mask_dilate_px: int = 2

    # Typeset
    padding_ratio: float = 0.08
    embedded_padding_ratio: float = 0.03
    min_font_size: int = 10
    max_font_size_ratio: float = 0.60
    embedded_max_font_size_ratio: float = 0.72
    line_spacing: float = 1.15
    embedded_line_spacing: float = 1.00

    stroke_width: int = 2
    embedded_stroke_width: int = 3


def _b64_png(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def _img_bytes_to_pil(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _call_lama_http(*, pil_img: Image.Image, mask: np.ndarray, cfg: LetteringConfig) -> Tuple[Image.Image, Dict[str, Any]]:
    url = (cfg.lama_url or "").rstrip("/") + "/inpaint"

    img_png = _pil_to_png_bytes(pil_img)
    mask_png = _pil_to_png_bytes(Image.fromarray(mask, mode="L"))

    payload = {
        "image_b64": _b64_png(img_png),
        "mask_b64": _b64_png(mask_png),
        "backend": cfg.lama_backend,
        "profile": cfg.erase_profile,
    }

    try:
        resp = requests.post(url, json=payload, timeout=float(cfg.lama_timeout))
    except Exception as e:
        raise RuntimeError(f"lama_http_failed: {e}")

    if resp.status_code != 200:
        raise RuntimeError(f"lama_http_bad_status: {resp.status_code} {resp.text[:200]}")

    data = resp.json()
    out_b = base64.b64decode(data["image_b64"])
    return _img_bytes_to_pil(out_b), data


def _pick_font(font_size: int) -> ImageFont.FreeTypeFont:
    # Prefer user-provided font.
    env_path = os.getenv("MANGA_AGENT_FONT_PATH")
    if env_path and os.path.exists(env_path):
        return ImageFont.truetype(env_path, font_size)

    # Common fallbacks.
    candidates = [
        # Windows
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\msyh.ttf",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\arial.ttf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, font_size)

    # PIL default bitmap font (limited), but keep the pipeline alive.
    return ImageFont.load_default()


def _wrap_text(text: str, *, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]

    # Respect explicit newlines as hard breaks.
    paragraphs = [p for p in text.splitlines() if p is not None]
    if len(paragraphs) > 1:
        out_lines: List[str] = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                out_lines.append("")
                continue
            out_lines.extend(_wrap_text(p, draw=draw, font=font, max_width=max_width))
        return out_lines

    # If it has spaces, treat as word wrap; otherwise treat as char wrap.
    if " " in text:
        tokens = text.split(" ")
        lines: List[str] = []
        cur = ""
        for t in tokens:
            cand = (cur + " " + t).strip()
            w = draw.textlength(cand, font=font)
            if w <= max_width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = t
        if cur:
            lines.append(cur)
        return lines

    # CJK / no-space: char wrap
    lines = []
    cur = ""
    for ch in list(text):
        cand = cur + ch
        w = draw.textlength(cand, font=font)
        if w <= max_width or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def _fit_text_to_box(text: str, *, box_w: int, box_h: int, cfg: LetteringConfig) -> Tuple[int, List[str], bool]:
    # Returns (font_size, lines, overflow)
    dummy_img = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    draw = ImageDraw.Draw(dummy_img)

    max_fs = max(cfg.min_font_size, int(round(box_h * cfg.max_font_size_ratio)))
    min_fs = cfg.min_font_size

    best = (min_fs, [text], True)

    for fs in range(max_fs, min_fs - 1, -2):
        font = _pick_font(fs)
        lines = _wrap_text(text, draw=draw, font=font, max_width=box_w)

        # Measure total height
        ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (fs, 0)
        line_h = int(round((ascent + descent) * cfg.line_spacing))
        total_h = line_h * len(lines)

        # Measure max line width
        max_w = 0
        for ln in lines:
            w = int(round(draw.textlength(ln, font=font)))
            max_w = max(max_w, w)

        if max_w <= box_w and total_h <= box_h:
            return fs, lines, False

        best = (fs, lines, True)

    return best


@lru_cache(maxsize=1)
def _load_runtime_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parents[2] / "config" / "runtime.local.json"
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _cfg_from_env() -> LetteringConfig:
    runtime_cfg = _load_runtime_config()
    lama_cfg = runtime_cfg.get("lama") if isinstance(runtime_cfg.get("lama"), dict) else {}

    def _cfg_value(env_name: str, cfg_key: str, default: Any) -> Any:
        return os.getenv(env_name, lama_cfg.get(cfg_key, default))

    return LetteringConfig(
        lama_url=_cfg_value("MANGA_AGENT_LAMA_URL", "url", "http://127.0.0.1:8765"),
        lama_backend=_cfg_value("MANGA_AGENT_LAMA_BACKEND", "backend", "lama"),
        erase_profile=os.getenv("MANGA_AGENT_LAMA_PROFILE", "balanced"),
        lama_timeout=float(_cfg_value("MANGA_AGENT_LAMA_TIMEOUT", "timeout_sec", 60)),
        mask_min_coverage=float(os.getenv("MANGA_AGENT_MASK_MIN_COVERAGE", "0.01")),
        mask_max_coverage=float(os.getenv("MANGA_AGENT_MASK_MAX_COVERAGE", "0.25")),
        mask_dilate_px=int(os.getenv("MANGA_AGENT_MASK_DILATE_PX", "2")),
        box_expand_px=int(os.getenv("MANGA_AGENT_MASK_BOX_EXPAND_PX", "1")),
        embedded_mask_min_coverage=float(os.getenv("MANGA_AGENT_EMBEDDED_MASK_MIN_COVERAGE", "0.02")),
        embedded_mask_max_coverage=float(os.getenv("MANGA_AGENT_EMBEDDED_MASK_MAX_COVERAGE", "0.45")),
        embedded_mask_dilate_px=int(os.getenv("MANGA_AGENT_EMBEDDED_MASK_DILATE_PX", "4")),
        embedded_box_expand_px=int(os.getenv("MANGA_AGENT_EMBEDDED_BOX_EXPAND_PX", "6")),
        complete_mask_close_px=int(os.getenv("MANGA_AGENT_COMPLETE_MASK_CLOSE_PX", "3")),
        embedded_complete_mask_close_px=int(os.getenv("MANGA_AGENT_EMBEDDED_COMPLETE_MASK_CLOSE_PX", "7")),
        complete_mask_dilate_px=int(os.getenv("MANGA_AGENT_COMPLETE_MASK_DILATE_PX", "0")),
        embedded_complete_mask_dilate_px=int(os.getenv("MANGA_AGENT_EMBEDDED_COMPLETE_MASK_DILATE_PX", "2")),
        padding_ratio=float(os.getenv("MANGA_AGENT_TYPESET_PADDING_RATIO", "0.08")),
        embedded_padding_ratio=float(os.getenv("MANGA_AGENT_EMBEDDED_PADDING_RATIO", "0.03")),
        min_font_size=int(os.getenv("MANGA_AGENT_MIN_FONT_SIZE", "10")),
        max_font_size_ratio=float(os.getenv("MANGA_AGENT_MAX_FONT_SIZE_RATIO", "0.60")),
        embedded_max_font_size_ratio=float(os.getenv("MANGA_AGENT_EMBEDDED_MAX_FONT_SIZE_RATIO", "0.72")),
        line_spacing=float(os.getenv("MANGA_AGENT_LINE_SPACING", "1.15")),
        embedded_line_spacing=float(os.getenv("MANGA_AGENT_EMBEDDED_LINE_SPACING", "1.00")),
        stroke_width=int(os.getenv("MANGA_AGENT_STROKE_WIDTH", "2")),
        embedded_stroke_width=int(os.getenv("MANGA_AGENT_EMBEDDED_STROKE_WIDTH", "3")),
    )


def _mask_cfg_from_lettering(cfg: LetteringConfig) -> MaskRefinementConfig:
    return MaskRefinementConfig(
        mask_min_coverage=cfg.mask_min_coverage,
        mask_max_coverage=cfg.mask_max_coverage,
        mask_dilate_px=cfg.mask_dilate_px,
        box_expand_px=cfg.box_expand_px,
        embedded_mask_min_coverage=cfg.embedded_mask_min_coverage,
        embedded_mask_max_coverage=cfg.embedded_mask_max_coverage,
        embedded_mask_dilate_px=cfg.embedded_mask_dilate_px,
        embedded_box_expand_px=cfg.embedded_box_expand_px,
        complete_mask_close_px=cfg.complete_mask_close_px,
        embedded_complete_mask_close_px=cfg.embedded_complete_mask_close_px,
        complete_mask_dilate_px=cfg.complete_mask_dilate_px,
        embedded_complete_mask_dilate_px=cfg.embedded_complete_mask_dilate_px,
    )


def _cfg_with_overrides(cfg: LetteringConfig, overrides: Optional[Dict[str, Any]]) -> LetteringConfig:
    if not isinstance(overrides, dict) or not overrides:
        return cfg
    allowed = {
        "erase_profile",
        "mask_dilate_px",
        "box_expand_px",
        "mask_min_coverage",
        "mask_max_coverage",
        "embedded_mask_min_coverage",
        "embedded_mask_max_coverage",
        "embedded_mask_dilate_px",
        "embedded_box_expand_px",
        "complete_mask_close_px",
        "embedded_complete_mask_close_px",
        "complete_mask_dilate_px",
        "embedded_complete_mask_dilate_px",
        "padding_ratio",
        "line_spacing",
        "stroke_width",
        "max_font_size_ratio",
    }
    patch = {k: v for k, v in overrides.items() if k in allowed}
    if not patch:
        return cfg
    return replace(cfg, **patch)


def _render_mode_for_region(region: Dict[str, Any], override: Optional[str] = None) -> str:
    if override in {"bubble_fit", "art_embed"}:
        return override
    return "art_embed" if coerce_has_text_box(region.get("has_text_box")) is False else "bubble_fit"


def _cfg_for_render_mode(cfg: LetteringConfig, render_mode: str) -> LetteringConfig:
    if render_mode != "art_embed":
        return cfg
    return replace(
        cfg,
        padding_ratio=cfg.embedded_padding_ratio,
        max_font_size_ratio=cfg.embedded_max_font_size_ratio,
        line_spacing=cfg.embedded_line_spacing,
        stroke_width=cfg.embedded_stroke_width,
    )


def _colors_for_render_mode(render_mode: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    if render_mode == "art_embed":
        return (255, 255, 255), (0, 0, 0)
    return (0, 0, 0), (255, 255, 255)


def _parse_region_color(value: Any) -> Optional[Tuple[int, int, int]]:
    text = str(value or "").strip().lower()
    if not text:
        return None

    named = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (220, 20, 60),
        "blue": (30, 144, 255),
        "green": (34, 139, 34),
        "yellow": (255, 215, 0),
        "pink": (255, 105, 180),
        "purple": (138, 43, 226),
        "orange": (255, 140, 0),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "brown": (139, 69, 19),
    }
    if text in named:
        return named[text]

    if text.startswith("#") and len(text) == 7:
        try:
            return tuple(int(text[i : i + 2], 16) for i in (1, 3, 5))
        except Exception:
            return None
    if text.startswith("#") and len(text) == 4:
        try:
            return tuple(int(text[i] * 2, 16) for i in (1, 2, 3))
        except Exception:
            return None
    return None


def _colors_for_region(region: Dict[str, Any], render_mode: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    base_fill, base_stroke = _colors_for_render_mode(render_mode)
    parsed = _parse_region_color(region.get("color"))
    if parsed is None:
        return base_fill, base_stroke

    luminance = (0.299 * parsed[0]) + (0.587 * parsed[1]) + (0.114 * parsed[2])
    stroke = (0, 0, 0) if luminance >= 140 else (255, 255, 255)
    return parsed, stroke


def _assess_mask_quality(mask_metrics: Dict[str, Any], *, cfg: LetteringConfig) -> Dict[str, Any]:
    region_metrics = [x for x in (mask_metrics.get("region_metrics") or []) if isinstance(x, dict)]
    embedded_regions = int(mask_metrics.get("embedded_regions") or 0)
    boxed = embedded_regions <= 0

    mask_coverage = float(mask_metrics.get("mask_coverage") or 0.0)
    regions = max(1, int(mask_metrics.get("regions") or 0))
    under_regions = 0
    over_regions = 0

    for item in region_metrics:
        coverage = float(item.get("completed_coverage") or item.get("refined_coverage") or 0.0)
        if bool(item.get("has_text_box", True)):
            if coverage < 0.008:
                under_regions += 1
            elif coverage > 0.38:
                over_regions += 1
        else:
            if coverage < 0.018:
                under_regions += 1
            elif coverage > 0.58:
                over_regions += 1

    if boxed:
        low_page_cov = mask_coverage < 0.004
        high_page_cov = mask_coverage > 0.20
    else:
        low_page_cov = mask_coverage < 0.010
        high_page_cov = mask_coverage > 0.32

    quality = "good"
    reason = "mask coverage looks plausible"
    retry_hint: Dict[str, Any] = {}

    if low_page_cov or (under_regions / float(regions)) >= 0.5:
        quality = "under_masked"
        reason = "mask likely misses text strokes or edge pixels"
        retry_hint = {
            "profile": "aggressive",
            "mask_dilate_px": min(5, int(cfg.mask_dilate_px) + 1),
        }
        if not boxed:
            retry_hint["embedded_box_expand_px"] = min(16, int(cfg.embedded_box_expand_px) + 2)
            retry_hint["embedded_complete_mask_close_px"] = min(15, int(cfg.embedded_complete_mask_close_px) + 2)
    elif high_page_cov or (over_regions / float(regions)) >= 0.5:
        quality = "over_masked"
        reason = "mask likely swallows nearby art or shading"
        retry_hint = {
            "profile": "conservative",
            "mask_dilate_px": max(1, int(cfg.mask_dilate_px) - 1),
            "mask_max_coverage": max(0.01, float(cfg.mask_max_coverage) - 0.04),
        }
        if not boxed:
            retry_hint["embedded_box_expand_px"] = max(0, int(cfg.embedded_box_expand_px) - 2)
            retry_hint["embedded_complete_mask_close_px"] = max(1, int(cfg.embedded_complete_mask_close_px) - 2)

    return {
        "mask_quality": quality,
        "mask_quality_reason": reason,
        "mask_retry_hint": retry_hint,
        "under_masked_regions": under_regions,
        "over_masked_regions": over_regions,
    }


def erase_page_image(
    *,
    image_bytes: bytes,
    regions: List[Dict[str, Any]],
    cfg: Optional[LetteringConfig] = None,
    skill_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg2 = _cfg_with_overrides(cfg or _cfg_from_env(), skill_params)

    pil_img = _img_bytes_to_pil(image_bytes)
    mask_bundle = build_page_mask(pil_img, regions, cfg=_mask_cfg_from_lettering(cfg2))
    mask = mask_bundle["mask"]

    if np.count_nonzero(mask) == 0 and regions:
        return {
            "ok": False,
            "erased_image_bytes": None,
            "results": {},
            "metrics": {
                "erase_backend": cfg2.lama_backend,
                "erase_profile": cfg2.erase_profile,
                **(mask_bundle.get("metrics") or {}),
            },
            "error": {"code": "empty_mask", "message": "Mask refinement produced an empty page mask."},
        }

    erase_profile = cfg2.erase_profile
    mask_metrics = mask_bundle.get("metrics") or {}
    mask_quality = _assess_mask_quality(mask_metrics, cfg=cfg2)
    if erase_profile == "balanced" and int(mask_metrics.get("embedded_regions") or 0) > 0:
        erase_profile = str(mask_metrics.get("profile_suggestion") or "aggressive")

    erased, erase_response = _call_lama_http(pil_img=pil_img, mask=mask, cfg=replace(cfg2, erase_profile=erase_profile))
    actual_backend = str((erase_response or {}).get("backend") or cfg2.lama_backend)
    actual_profile = str((erase_response or {}).get("profile") or erase_profile)

    return {
        "ok": True,
        "erased_image_bytes": _pil_to_png_bytes(erased),
        "coarse_mask_bytes": _pil_to_png_bytes(Image.fromarray(mask_bundle["coarse_mask"], mode="L")),
        "mask_bytes": _pil_to_png_bytes(Image.fromarray(mask, mode="L")),
        "mask_overlay_bytes": _pil_to_png_bytes(Image.fromarray(mask_bundle["mask_overlay"], mode="RGB")),
        "results": {
            int(r.get("region_id")): {"erase_ok": True}
            for r in regions
            if str(r.get("region_id")).strip().isdigit()
        },
        "metrics": {
            "erase_backend": actual_backend,
            "erase_profile": actual_profile,
            **mask_metrics,
            **mask_quality,
        },
        "error": None,
    }


def typeset_page_image(
    *,
    image_bytes: bytes,
    regions: List[Dict[str, Any]],
    cfg: Optional[LetteringConfig] = None,
    skill_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg2 = _cfg_with_overrides(cfg or _cfg_from_env(), skill_params)
    render_mode_override = None
    if isinstance(skill_params, dict):
        v = skill_params.get("render_mode")
        if isinstance(v, str) and v in {"bubble_fit", "art_embed"}:
            render_mode_override = v

    out = _img_bytes_to_pil(image_bytes)
    W, H = out.size
    draw = ImageDraw.Draw(out)

    results: Dict[int, Dict[str, Any]] = {}

    for r in regions:
        try:
            region_id = int(r.get("region_id"))
        except Exception:
            continue

        text = str(r.get("translated_text") or r.get("tgt_text") or "").strip()
        bbox = r.get("bbox_xywh") or r.get("bbox")
        xywh = safe_bbox_xywh(bbox, W, H)
        render_mode = _render_mode_for_region(r, override=render_mode_override)
        cfg_region = _cfg_for_render_mode(cfg2, render_mode)
        fill, stroke_fill = _colors_for_region(r, render_mode)

        if not xywh:
            results[region_id] = {
                "render_ok": False,
                "overflow": False,
                "render_params": {"reason": "bad_bbox", "render_mode": render_mode},
            }
            continue

        x, y, w, h = xywh

        pad = int(round(min(w, h) * cfg_region.padding_ratio))
        inner_w = max(1, w - 2 * pad)
        inner_h = max(1, h - 2 * pad)

        if not text:
            results[region_id] = {
                "render_ok": True,
                "overflow": False,
                "render_params": {"font_size": 0, "lines": 0, "render_mode": render_mode},
            }
            continue

        fs, lines, overflow = _fit_text_to_box(text, box_w=inner_w, box_h=inner_h, cfg=cfg_region)
        font = _pick_font(fs)

        ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (fs, 0)
        line_h = int(round((ascent + descent) * cfg_region.line_spacing))
        total_h = line_h * len(lines)

        start_y = y + pad + max(0, int(round((inner_h - total_h) / 2.0)))

        for i, ln in enumerate(lines):
            yy = start_y + i * line_h
            lw = int(round(draw.textlength(ln, font=font)))
            xx = x + pad + max(0, int(round((inner_w - lw) / 2.0)))
            draw.text(
                (xx, yy),
                ln,
                fill=fill,
                font=font,
                stroke_width=int(cfg_region.stroke_width),
                stroke_fill=stroke_fill,
            )

        results[region_id] = {
            "render_ok": True,
            "overflow": bool(overflow),
            "render_params": {
                "font_size": int(fs),
                "lines": len(lines),
                "padding": int(pad),
                "render_mode": render_mode,
                "has_text_box": coerce_has_text_box(r.get("has_text_box")),
                "color": str(r.get("color") or "black"),
            },
        }

    return {
        "ok": True,
        "rendered_image_bytes": _pil_to_png_bytes(out),
        "results": results,
        "metrics": {
            "padding_ratio": cfg2.padding_ratio,
            "line_spacing": cfg2.line_spacing,
            "stroke_width": cfg2.stroke_width,
        },
        "error": None,
    }


def render_lettered_page(
    *,
    image_bytes: bytes,
    regions: List[Dict[str, Any]],
    cfg: Optional[LetteringConfig] = None,
    skill_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg2 = _cfg_with_overrides(cfg or _cfg_from_env(), skill_params)
    erased = erase_page_image(image_bytes=image_bytes, regions=regions, cfg=cfg2, skill_params=skill_params)
    if not erased.get("ok"):
        return erased

    return typeset_page_image(
        image_bytes=erased["erased_image_bytes"],
        regions=regions,
        cfg=cfg2,
        skill_params=skill_params,
    )
