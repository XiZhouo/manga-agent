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

from qa import evaluate_typeset_page
from skills.lettering_basic_v0.mask_refinement import (
    MaskRefinementConfig,
    _mask_overlay,
    build_hybrid_detect_mask,
    build_page_mask,
    constrain_remote_detect_mask,
    coerce_has_text_box,
    resolve_region_bbox_xywh,
    safe_bbox_xywh,
)


@dataclass(frozen=True)
class LetteringConfig:
    lama_url: str = "http://127.0.0.1:8765"
    lama_backend: str = "lama"
    erase_profile: str = "balanced"
    lama_timeout: float = 60.0
    mask_source_mode: str = "manga_auto_hybrid"  # heuristic|manga_auto|manga_auto_constrained|manga_auto_hybrid
    mask_refine_mode: str = "heuristic"  # heuristic|sam_auto|sam_required
    sam_timeout: float = 45.0
    sam_positive_points_per_region: int = 4
    sam_negative_points_per_region: int = 8
    sam_edge_protect_px: int = 10
    sam_mask_dilate_px: int = 0

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
    auto_mask_box_expand_px: int = 12
    auto_mask_embedded_box_expand_px: int = 28
    auto_mask_min_component_area_px: int = 40
    auto_mask_min_component_overlap_ratio: float = 0.15
    auto_mask_min_component_overlap_px: int = 24
    auto_mask_final_erode_px: int = 1
    hybrid_seed_safe_expand_px: int = 5
    hybrid_envelope_safe_expand_px: int = 40
    hybrid_growth_iterations: int = 15
    hybrid_growth_kernel_px: int = 3

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


class RemoteServiceError(RuntimeError):
    def __init__(
        self,
        *,
        service: str,
        operation: str,
        code: str,
        message: str,
        url: Optional[str] = None,
        retryable: bool = True,
        http_status: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.service = str(service or "remote_service")
        self.operation = str(operation or "request")
        self.code = str(code or "remote_service_error")
        self.message = str(message or "Remote service request failed.")
        self.url = str(url or "")
        self.retryable = bool(retryable)
        self.http_status = int(http_status) if http_status is not None else None
        self.details = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "kind": "remote_service",
            "service": self.service,
            "operation": self.operation,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.url:
            payload["url"] = self.url
        if self.http_status is not None:
            payload["http_status"] = self.http_status
        if self.details:
            payload["details"] = dict(self.details)
        return payload


def _raise_remote_http_request_error(*, url: str, operation: str, exc: Exception) -> None:
    if isinstance(exc, requests.exceptions.Timeout):
        raise RemoteServiceError(
            service="remote_erase_service",
            operation=operation,
            code=f"{operation}_timeout",
            message="Remote erase service timed out before returning a result.",
            url=url,
            retryable=True,
            details={"exception_type": type(exc).__name__, "raw_message": str(exc)},
        )
    if isinstance(exc, requests.exceptions.ConnectionError):
        raise RemoteServiceError(
            service="remote_erase_service",
            operation=operation,
            code=f"{operation}_service_unreachable",
            message="Remote erase service is unavailable or not responding.",
            url=url,
            retryable=True,
            details={"exception_type": type(exc).__name__, "raw_message": str(exc)},
        )
    raise RemoteServiceError(
        service="remote_erase_service",
        operation=operation,
        code=f"{operation}_request_failed",
        message="Remote erase service request failed before a response was received.",
        url=url,
        retryable=True,
        details={"exception_type": type(exc).__name__, "raw_message": str(exc)},
    )


def _raise_remote_http_status_error(*, url: str, operation: str, resp: requests.Response) -> None:
    status = int(resp.status_code or 0)
    code = f"{operation}_bad_status"
    message = f"Remote erase service returned HTTP {status}."
    if status in {502, 503, 504}:
        code = f"{operation}_service_unavailable"
        message = f"Remote erase service returned HTTP {status} and is currently unavailable."
    raise RemoteServiceError(
        service="remote_erase_service",
        operation=operation,
        code=code,
        message=message,
        url=url,
        retryable=True,
        http_status=status,
        details={"response_preview": resp.text[:200]},
    )


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
        _raise_remote_http_request_error(url=url, operation="inpaint", exc=e)

    if resp.status_code != 200:
        _raise_remote_http_status_error(url=url, operation="inpaint", resp=resp)

    data = resp.json()
    out_b = base64.b64decode(data["image_b64"])
    return _img_bytes_to_pil(out_b), data


def _call_auto_detect_http(*, pil_img: Image.Image, cfg: LetteringConfig) -> Dict[str, Any]:
    url = (cfg.lama_url or "").rstrip("/") + "/detect-text-mask"
    img_png = _pil_to_png_bytes(pil_img)
    payload = {
        "image_b64": _b64_png(img_png),
        "profile": cfg.erase_profile,
        "debug": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=float(cfg.lama_timeout))
    except Exception as e:
        _raise_remote_http_request_error(url=url, operation="detect_text_mask", exc=e)

    if resp.status_code != 200:
        _raise_remote_http_status_error(url=url, operation="detect_text_mask", resp=resp)

    data = resp.json()
    if not isinstance(data, dict) or not data.get("mask_b64"):
        raise RuntimeError("auto_detect_http_missing_mask")

    mask_b = base64.b64decode(data["mask_b64"])
    mask = Image.open(io.BytesIO(mask_b)).convert("L")

    heatmap = None
    overlay = None
    debug_artifacts = data.get("debug_artifacts") or {}
    if isinstance(debug_artifacts, dict):
        if debug_artifacts.get("heatmap_b64"):
            heatmap = Image.open(io.BytesIO(base64.b64decode(debug_artifacts["heatmap_b64"]))).convert("L")
        if debug_artifacts.get("overlay_b64"):
            overlay = Image.open(io.BytesIO(base64.b64decode(debug_artifacts["overlay_b64"]))).convert("RGB")

    return {
        "mask": np.array(mask, dtype=np.uint8),
        "heatmap": heatmap,
        "overlay": overlay,
        "backend": str(data.get("backend") or "mask:manga_cleaner_ocr"),
        "metrics": data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
        "raw": data,
    }


def _call_sam_refine_http(
    *,
    pil_img: Image.Image,
    coarse_mask: np.ndarray,
    regions: List[Dict[str, Any]],
    cfg: LetteringConfig,
) -> Dict[str, Any]:
    url = (cfg.lama_url or "").rstrip("/") + "/refine-mask"
    img_png = _pil_to_png_bytes(pil_img)
    coarse_png = _pil_to_png_bytes(Image.fromarray(coarse_mask, mode="L"))

    payload = {
        "image_b64": _b64_png(img_png),
        "coarse_mask_b64": _b64_png(coarse_png),
        "regions": [
            {
                "region_id": int(r.get("region_id")),
                "bbox_xyxy": r.get("bbox_xyxy"),
                "bbox_xywh": r.get("bbox_xywh") or r.get("bbox"),
                "has_text_box": r.get("has_text_box"),
            }
            for r in regions
            if str(r.get("region_id")).strip().isdigit()
        ],
        "profile": cfg.erase_profile,
        "positive_points_per_region": int(cfg.sam_positive_points_per_region),
        "negative_points_per_region": int(cfg.sam_negative_points_per_region),
        "edge_protect_px": int(cfg.sam_edge_protect_px),
        "mask_dilate_px": int(cfg.sam_mask_dilate_px),
        "multimask_output": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=float(cfg.sam_timeout))
    except Exception as e:
        _raise_remote_http_request_error(url=url, operation="refine_mask", exc=e)

    if resp.status_code != 200:
        _raise_remote_http_status_error(url=url, operation="refine_mask", resp=resp)

    data = resp.json()
    refined_b = base64.b64decode(data["refined_mask_b64"])
    refined = Image.open(io.BytesIO(refined_b)).convert("L")

    overlay_img = None
    if data.get("overlay_b64"):
        overlay_b = base64.b64decode(data["overlay_b64"])
        overlay_img = Image.open(io.BytesIO(overlay_b)).convert("RGB")

    return {
        "mask": np.array(refined, dtype=np.uint8),
        "overlay": overlay_img,
        "backend": str(data.get("backend") or "sam"),
        "metrics": data.get("metrics") if isinstance(data.get("metrics"), dict) else {},
        "raw": data,
    }


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


def _fit_text_to_box(
    text: str,
    *,
    box_w: int,
    box_h: int,
    cfg: LetteringConfig,
    max_font_size: Optional[int] = None,
) -> Tuple[int, List[str], bool]:
    # Returns (font_size, lines, overflow)
    dummy_img = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    draw = ImageDraw.Draw(dummy_img)

    max_fs = max(cfg.min_font_size, int(round(box_h * cfg.max_font_size_ratio)))
    if max_font_size is not None:
        max_fs = max(cfg.min_font_size, min(max_fs, int(max_font_size)))
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


def _split_page_mask_into_region_masks(
    mask: np.ndarray,
    *,
    regions: List[Dict[str, Any]],
    image_size: Tuple[int, int],
    assign_expand_px: int = 8,
) -> Dict[int, np.ndarray]:
    import cv2

    W, H = image_size
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if binary.shape[:2] != (H, W):
        binary = cv2.resize(binary, (W, H), interpolation=cv2.INTER_NEAREST)

    region_masks: Dict[int, np.ndarray] = {}
    region_specs: List[Tuple[int, Tuple[int, int, int, int]]] = []

    for idx, region in enumerate(regions, start=1):
        try:
            region_id = int(region.get("region_id") or idx)
        except Exception:
            continue

        bbox = region.get("bbox_xyxy") or region.get("bbox_xywh") or region.get("bbox")
        xywh = resolve_region_bbox_xywh(bbox, W=W, H=H, guide_mask=binary, expand_px=0)
        if not xywh:
            continue

        x, y, w, h = xywh
        pad = max(0, int(assign_expand_px))
        ax0 = max(0, x - pad)
        ay0 = max(0, y - pad)
        ax1 = min(W, x + w + pad)
        ay1 = min(H, y + h + pad)
        region_masks[region_id] = np.zeros((H, W), dtype=np.uint8)
        region_specs.append((region_id, (ax0, ay0, ax1, ay1)))

    if not region_specs or np.count_nonzero(binary) == 0:
        return region_masks

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels[y : y + h, x : x + w] == label
        if not np.any(component):
            continue

        best_region_id: Optional[int] = None
        best_overlap_px = -1
        for region_id, (ax0, ay0, ax1, ay1) in region_specs:
            ix0 = max(x, ax0)
            iy0 = max(y, ay0)
            ix1 = min(x + w, ax1)
            iy1 = min(y + h, ay1)
            if ix1 <= ix0 or iy1 <= iy0:
                overlap_px = 0
            else:
                overlap_px = int(np.count_nonzero(component[iy0 - y : iy1 - y, ix0 - x : ix1 - x]))
            if overlap_px > best_overlap_px:
                best_overlap_px = overlap_px
                best_region_id = region_id

        if best_region_id is None or best_overlap_px <= 0:
            continue
        target = region_masks[best_region_id]
        target[y : y + h, x : x + w][component] = 255

    return region_masks


def _build_placement_lines_from_region_mask(mask: np.ndarray) -> List[Dict[str, int]]:
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if binary.ndim != 2 or np.count_nonzero(binary) == 0:
        return []

    ys, xs = np.where(binary > 0)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    roi = binary[y0:y1, x0:x1]
    rh, rw = roi.shape
    if rh <= 0 or rw <= 0:
        return []

    kx = max(3, min(17, int(round(rw * 0.08))))
    if kx % 2 == 0:
        kx += 1
    processed = cv2.dilate(roi, np.ones((1, kx), np.uint8), iterations=1)
    row_strength = np.count_nonzero(processed > 0, axis=1).astype(np.float32)
    nonzero_rows = np.where(row_strength > 0)[0]
    if nonzero_rows.size == 0:
        return []

    start_row = int(nonzero_rows.min())
    end_row = int(nonzero_rows.max()) + 1
    profile = row_strength[start_row:end_row]
    window = 5
    smooth = np.convolve(profile, np.ones(window, dtype=np.float32) / float(window), mode="same")
    tolerance = max(10, int(round(float(smooth.max(initial=0.0)) * 0.08)))
    min_band_h = max(8, int(round(profile.shape[0] * 0.08)))

    lines: List[Tuple[int, int]] = []
    band_start = 0
    band_ref = float(smooth[0])
    for idx in range(1, len(smooth)):
        band_h = idx - band_start
        if band_h >= min_band_h and abs(float(smooth[idx]) - band_ref) >= tolerance:
            lines.append((start_row + band_start, start_row + idx))
            band_start = idx
            band_ref = float(smooth[idx])
            continue
        band_ref = ((band_ref * float(band_h)) + float(smooth[idx])) / float(band_h + 1)
    lines.append((start_row + band_start, end_row))

    placement_lines: List[Dict[str, int]] = []
    for band_y0, band_y1 in lines:
        band = processed[band_y0:band_y1, :]
        if band.size == 0 or np.count_nonzero(band) == 0:
            continue
        band_ys, band_xs = np.where(band > 0)
        left = int(band_xs.min())
        right = int(band_xs.max()) + 1
        top = int(band_y0 + band_ys.min())
        bottom = int(band_y0 + band_ys.max()) + 1
        width = max(1, right - left)
        height = max(1, bottom - top)
        if width < 8 or height < 6:
            continue

        pad_x = max(2, min(8, int(round(width * 0.04))))
        pad_y = max(1, min(4, int(round(height * 0.15))))
        x_left = x0 + max(0, left - pad_x)
        x_right = x0 + min(rw, right + pad_x)
        y_top = y0 + max(0, top - pad_y)
        y_bottom = y0 + min(rh, bottom + pad_y)

        placement_lines.append(
            {
                "x_left": int(x_left),
                "x_right": int(x_right),
                "y_top": int(y_top),
                "y_bottom": int(y_bottom),
                "x_center": int(round((x_left + x_right) / 2.0)),
                "y_center": int(round((y_top + y_bottom) / 2.0)),
                "usable_width": int(max(1, x_right - x_left)),
                "usable_height": int(max(1, y_bottom - y_top)),
            }
        )

    return placement_lines


def _wrap_text_to_widths(
    text: str,
    *,
    widths: List[int],
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
) -> Optional[List[str]]:
    text = (text or "").strip()
    if not text:
        return [""]
    if not widths:
        return None

    paragraphs = text.splitlines() or [text]
    out: List[str] = []
    width_idx = 0

    def _consume_tokens(segment: str, *, widths_local: List[int], start_idx: int) -> Tuple[Optional[List[str]], int]:
        mode_word = " " in segment
        tokens = segment.split(" ") if mode_word else list(segment)
        lines_local: List[str] = []
        cur = ""
        idx = start_idx

        for tok in tokens:
            if idx >= len(widths_local):
                return None, idx
            cand = tok if not cur else (f"{cur} {tok}" if mode_word else f"{cur}{tok}")
            if draw.textlength(cand, font=font) <= widths_local[idx] or not cur:
                if not cur and draw.textlength(tok, font=font) > widths_local[idx]:
                    return None, idx
                cur = cand
            else:
                lines_local.append(cur)
                idx += 1
                if idx >= len(widths_local) or draw.textlength(tok, font=font) > widths_local[idx]:
                    return None, idx
                cur = tok

        if cur:
            lines_local.append(cur)
            idx += 1
        return lines_local, idx

    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if para == "":
            if width_idx >= len(widths):
                return None
            out.append("")
            width_idx += 1
            continue

        wrapped, next_idx = _consume_tokens(para, widths_local=widths, start_idx=width_idx)
        if wrapped is None:
            return None
        out.extend(wrapped)
        width_idx = next_idx
        if para_idx < len(paragraphs) - 1 and width_idx >= len(widths):
            return None

    return out


def _fit_text_to_placement_lines(
    text: str,
    *,
    placement_lines: List[Dict[str, int]],
    cfg: LetteringConfig,
    max_font_size: Optional[int] = None,
) -> Tuple[int, List[str], bool, List[Dict[str, int]]]:
    if not placement_lines:
        return cfg.min_font_size, [text], True, []

    dummy_img = Image.new("RGB", (64, 64), (255, 255, 255))
    draw = ImageDraw.Draw(dummy_img)
    ordered = sorted(
        [dict(item) for item in placement_lines if isinstance(item, dict)],
        key=lambda item: (int(item.get("y_center", 0)), int(item.get("x_center", 0))),
    )
    if not ordered:
        return cfg.min_font_size, [text], True, []

    heights = [int(item.get("usable_height", 0)) for item in ordered if int(item.get("usable_height", 0)) > 0]
    if heights:
        median_h = float(np.median(np.asarray(heights, dtype=np.float32)))
        min_keep_h = max(10, int(round(median_h * 0.45)))
        filtered = [item for item in ordered if int(item.get("usable_height", 0)) >= min_keep_h]
        if filtered:
            ordered = filtered

    max_h = max(int(item.get("usable_height", 0)) for item in ordered) if ordered else 0
    max_fs = max(cfg.min_font_size, min(int(round(max_h * 1.15)), int(round(max_h)), int(round(512 * cfg.max_font_size_ratio))))
    if max_font_size is not None:
        max_fs = max(cfg.min_font_size, min(max_fs, int(max_font_size)))
    best: Tuple[int, List[str], bool, List[Dict[str, int]]] = (cfg.min_font_size, [text], True, ordered[:1])

    for fs in range(max_fs, cfg.min_font_size - 1, -2):
        font = _pick_font(fs)
        ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (fs, 0)
        line_h = int(round((ascent + descent) * cfg.line_spacing))
        candidate_slots: List[Dict[str, int]] = []
        for slot in ordered:
            usable_h = int(slot.get("usable_height", 0))
            if usable_h + 2 < line_h:
                continue
            split_count = max(1, min(4, usable_h // max(1, line_h)))
            if split_count <= 1:
                candidate_slots.append(slot)
                continue
            y_top = int(slot.get("y_top", slot.get("y_center", 0)))
            y_bottom = int(slot.get("y_bottom", slot.get("y_center", 0)))
            span = max(1, y_bottom - y_top)
            for i in range(split_count):
                sub_top = int(round(y_top + (span * i) / float(split_count)))
                sub_bottom = int(round(y_top + (span * (i + 1)) / float(split_count)))
                candidate_slots.append(
                    {
                        **slot,
                        "y_top": sub_top,
                        "y_bottom": sub_bottom,
                        "y_center": int(round((sub_top + sub_bottom) / 2.0)),
                        "usable_height": int(max(1, sub_bottom - sub_top)),
                    }
                )
        if not candidate_slots:
            continue

        best_candidate: Optional[Tuple[List[str], List[Dict[str, int]], Tuple[float, float]]] = None
        page_mid = float(candidate_slots[0]["y_center"] + candidate_slots[-1]["y_center"]) / 2.0
        for line_count in range(len(candidate_slots), 0, -1):
            for start in range(0, len(candidate_slots) - line_count + 1):
                slots = candidate_slots[start : start + line_count]
                widths = [int(slot.get("usable_width", 1)) for slot in slots]
                wrapped = _wrap_text_to_widths(text, widths=widths, draw=draw, font=font)
                if wrapped is None or len(wrapped) != line_count:
                    continue
                line_widths = [float(draw.textlength(line, font=font)) for line in wrapped]
                if any(line_widths[i] > widths[i] for i in range(line_count)):
                    continue

                slot_mid = float(slots[0]["y_center"] + slots[-1]["y_center"]) / 2.0
                score = (float(sum(widths) - sum(line_widths)), abs(slot_mid - page_mid))
                if best_candidate is None or score < best_candidate[2]:
                    best_candidate = (wrapped, slots, score)

            if best_candidate is not None:
                lines, slots, _ = best_candidate
                return fs, lines, False, slots

        best = (fs, [text], True, candidate_slots[:1])

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
        mask_source_mode=os.getenv("MANGA_AGENT_MASK_SOURCE_MODE", "manga_auto_hybrid"),
        mask_refine_mode=os.getenv("MANGA_AGENT_MASK_REFINE_MODE", "heuristic"),
        sam_timeout=float(_cfg_value("MANGA_AGENT_SAM_TIMEOUT", "sam_timeout_sec", 45)),
        sam_positive_points_per_region=int(os.getenv("MANGA_AGENT_SAM_POS_POINTS", "4")),
        sam_negative_points_per_region=int(os.getenv("MANGA_AGENT_SAM_NEG_POINTS", "8")),
        sam_edge_protect_px=int(os.getenv("MANGA_AGENT_SAM_EDGE_PROTECT_PX", "10")),
        sam_mask_dilate_px=int(os.getenv("MANGA_AGENT_SAM_MASK_DILATE_PX", "0")),
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
        auto_mask_box_expand_px=int(os.getenv("MANGA_AGENT_AUTO_MASK_BOX_EXPAND_PX", "12")),
        auto_mask_embedded_box_expand_px=int(os.getenv("MANGA_AGENT_AUTO_MASK_EMBEDDED_BOX_EXPAND_PX", "28")),
        auto_mask_min_component_area_px=int(os.getenv("MANGA_AGENT_AUTO_MASK_MIN_COMPONENT_AREA_PX", "40")),
        auto_mask_min_component_overlap_ratio=float(
            os.getenv("MANGA_AGENT_AUTO_MASK_MIN_COMPONENT_OVERLAP_RATIO", "0.15")
        ),
        auto_mask_min_component_overlap_px=int(os.getenv("MANGA_AGENT_AUTO_MASK_MIN_COMPONENT_OVERLAP_PX", "24")),
        auto_mask_final_erode_px=int(os.getenv("MANGA_AGENT_AUTO_MASK_FINAL_ERODE_PX", "1")),
        hybrid_seed_safe_expand_px=int(os.getenv("MANGA_AGENT_HYBRID_SEED_SAFE_EXPAND_PX", "5")),
        hybrid_envelope_safe_expand_px=int(os.getenv("MANGA_AGENT_HYBRID_ENVELOPE_SAFE_EXPAND_PX", "40")),
        hybrid_growth_iterations=int(os.getenv("MANGA_AGENT_HYBRID_GROWTH_ITERATIONS", "15")),
        hybrid_growth_kernel_px=int(os.getenv("MANGA_AGENT_HYBRID_GROWTH_KERNEL_PX", "3")),
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
        auto_mask_box_expand_px=cfg.auto_mask_box_expand_px,
        auto_mask_embedded_box_expand_px=cfg.auto_mask_embedded_box_expand_px,
        auto_mask_min_component_area_px=cfg.auto_mask_min_component_area_px,
        auto_mask_min_component_overlap_ratio=cfg.auto_mask_min_component_overlap_ratio,
        auto_mask_min_component_overlap_px=cfg.auto_mask_min_component_overlap_px,
        auto_mask_final_erode_px=cfg.auto_mask_final_erode_px,
        hybrid_seed_safe_expand_px=cfg.hybrid_seed_safe_expand_px,
        hybrid_envelope_safe_expand_px=cfg.hybrid_envelope_safe_expand_px,
        hybrid_growth_iterations=cfg.hybrid_growth_iterations,
        hybrid_growth_kernel_px=cfg.hybrid_growth_kernel_px,
    )


def _cfg_with_overrides(cfg: LetteringConfig, overrides: Optional[Dict[str, Any]]) -> LetteringConfig:
    if not isinstance(overrides, dict) or not overrides:
        return cfg
    allowed = {
        "erase_profile",
        "mask_source_mode",
        "mask_refine_mode",
        "sam_timeout",
        "sam_positive_points_per_region",
        "sam_negative_points_per_region",
        "sam_edge_protect_px",
        "sam_mask_dilate_px",
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
        "auto_mask_box_expand_px",
        "auto_mask_embedded_box_expand_px",
        "auto_mask_min_component_area_px",
        "auto_mask_min_component_overlap_ratio",
        "auto_mask_min_component_overlap_px",
        "auto_mask_final_erode_px",
        "hybrid_seed_safe_expand_px",
        "hybrid_envelope_safe_expand_px",
        "hybrid_growth_iterations",
        "hybrid_growth_kernel_px",
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
    auto_detect_bundle: Optional[Dict[str, Any]] = None
    mask_source_mode = str(cfg2.mask_source_mode or "heuristic").strip().lower()

    if mask_source_mode in {"manga_auto", "manga_auto_constrained", "manga_auto_hybrid"}:
        auto_detect_resp = _call_auto_detect_http(pil_img=pil_img, cfg=cfg2)
        raw_detect_mask = (auto_detect_resp.get("mask") > 0).astype(np.uint8) * 255
        if mask_source_mode == "manga_auto_hybrid":
            heatmap_img = auto_detect_resp.get("heatmap")
            heatmap_arr = np.array(heatmap_img, dtype=np.uint8) if heatmap_img is not None else None
            if heatmap_arr is None or np.count_nonzero(heatmap_arr) == 0:
                constrained = constrain_remote_detect_mask(
                    raw_detect_mask,
                    regions,
                    image_size=pil_img.size,
                    cfg=_mask_cfg_from_lettering(cfg2),
                )
                final_mask = constrained["mask"]
                strategy = "manga_auto_constrained_fallback_no_heatmap"
                extra_metrics = {"auto_constraint_metrics": constrained.get("metrics") or {}}
            else:
                hybrid = build_hybrid_detect_mask(
                    raw_detect_mask,
                    heatmap_arr,
                    regions,
                    image_size=pil_img.size,
                    cfg=_mask_cfg_from_lettering(cfg2),
                )
                final_mask = hybrid["mask"]
                strategy = "manga_auto_hybrid"
                extra_metrics = {
                    "auto_hybrid_metrics": hybrid.get("metrics") or {},
                }

            overlay_img = _mask_overlay(pil_img, final_mask)
            mask_bundle = {
                "coarse_mask": raw_detect_mask,
                "mask": final_mask,
                "mask_overlay": np.array(overlay_img),
                "metrics": {
                    "regions": len(regions),
                    "boxed_regions": sum(1 for r in regions if coerce_has_text_box(r.get("has_text_box")) is not False),
                    "embedded_regions": sum(1 for r in regions if coerce_has_text_box(r.get("has_text_box")) is False),
                    "image_pixels": int(pil_img.size[0] * pil_img.size[1]),
                    "coarse_mask_pixels": int(np.count_nonzero(raw_detect_mask)),
                    "mask_pixels": int(np.count_nonzero(final_mask)),
                    "coarse_mask_coverage": round(float(np.count_nonzero(raw_detect_mask)) / float(max(1, raw_detect_mask.size)), 6),
                    "mask_coverage": round(float(np.count_nonzero(final_mask)) / float(max(1, final_mask.size)), 6),
                    "mask_strategy": strategy,
                    "profile_suggestion": "balanced",
                    "region_metrics": [],
                    "auto_detect_backend": auto_detect_resp.get("backend"),
                    "auto_detect_metrics": auto_detect_resp.get("metrics") or {},
                    **extra_metrics,
                },
            }
            auto_detect_bundle = {
                "raw_mask": raw_detect_mask,
                "final_mask": final_mask,
                "heatmap": auto_detect_resp.get("heatmap"),
                "overlay": auto_detect_resp.get("overlay"),
            }
        elif mask_source_mode == "manga_auto_constrained":
            constrained = constrain_remote_detect_mask(
                raw_detect_mask,
                regions,
                image_size=pil_img.size,
                cfg=_mask_cfg_from_lettering(cfg2),
            )
            final_mask = constrained["mask"]
            overlay_img = _mask_overlay(pil_img, final_mask)
            mask_bundle = {
                "coarse_mask": raw_detect_mask,
                "mask": final_mask,
                "mask_overlay": np.array(overlay_img),
                "metrics": {
                    "regions": len(regions),
                    "boxed_regions": sum(1 for r in regions if coerce_has_text_box(r.get("has_text_box")) is not False),
                    "embedded_regions": sum(1 for r in regions if coerce_has_text_box(r.get("has_text_box")) is False),
                    "image_pixels": int(pil_img.size[0] * pil_img.size[1]),
                    "coarse_mask_pixels": int(np.count_nonzero(raw_detect_mask)),
                    "mask_pixels": int(np.count_nonzero(final_mask)),
                    "coarse_mask_coverage": round(float(np.count_nonzero(raw_detect_mask)) / float(max(1, raw_detect_mask.size)), 6),
                    "mask_coverage": round(float(np.count_nonzero(final_mask)) / float(max(1, final_mask.size)), 6),
                    "mask_strategy": "manga_auto_constrained",
                    "profile_suggestion": "balanced",
                    "region_metrics": [],
                    "auto_detect_backend": auto_detect_resp.get("backend"),
                    "auto_detect_metrics": auto_detect_resp.get("metrics") or {},
                    "auto_constraint_metrics": constrained.get("metrics") or {},
                },
            }
            auto_detect_bundle = {
                "raw_mask": raw_detect_mask,
                "final_mask": final_mask,
                "heatmap": auto_detect_resp.get("heatmap"),
                "overlay": auto_detect_resp.get("overlay"),
            }
        else:
            overlay_img = auto_detect_resp.get("overlay") or _mask_overlay(pil_img, raw_detect_mask)
            mask_bundle = {
                "coarse_mask": raw_detect_mask,
                "mask": raw_detect_mask,
                "mask_overlay": np.array(overlay_img),
                "metrics": {
                    "regions": len(regions),
                    "boxed_regions": sum(1 for r in regions if coerce_has_text_box(r.get("has_text_box")) is not False),
                    "embedded_regions": sum(1 for r in regions if coerce_has_text_box(r.get("has_text_box")) is False),
                    "image_pixels": int(pil_img.size[0] * pil_img.size[1]),
                    "coarse_mask_pixels": int(np.count_nonzero(raw_detect_mask)),
                    "mask_pixels": int(np.count_nonzero(raw_detect_mask)),
                    "coarse_mask_coverage": round(float(np.count_nonzero(raw_detect_mask)) / float(max(1, raw_detect_mask.size)), 6),
                    "mask_coverage": round(float(np.count_nonzero(raw_detect_mask)) / float(max(1, raw_detect_mask.size)), 6),
                    "mask_strategy": "manga_auto_raw",
                    "profile_suggestion": "balanced",
                    "region_metrics": [],
                    "auto_detect_backend": auto_detect_resp.get("backend"),
                    "auto_detect_metrics": auto_detect_resp.get("metrics") or {},
                },
            }
            auto_detect_bundle = {
                "raw_mask": raw_detect_mask,
                "final_mask": raw_detect_mask,
                "heatmap": auto_detect_resp.get("heatmap"),
                "overlay": auto_detect_resp.get("overlay"),
            }
    else:
        mask_bundle = build_page_mask(pil_img, regions, cfg=_mask_cfg_from_lettering(cfg2))

    coarse_mask = mask_bundle["coarse_mask"]
    mask = mask_bundle["mask"]
    overlay_img = Image.fromarray(mask_bundle["mask_overlay"], mode="RGB")
    sam_metrics: Dict[str, Any] = {"sam_refine_status": "skipped"}

    if np.count_nonzero(mask) == 0 and regions:
        return {
            "ok": False,
            "erased_image_bytes": None,
            "results": {},
            "metrics": {
                "erase_backend": cfg2.lama_backend,
                "erase_profile": cfg2.erase_profile,
                "mask_source_mode": mask_source_mode,
                **(mask_bundle.get("metrics") or {}),
            },
            "error": {"code": "empty_mask", "message": "Mask refinement produced an empty page mask."},
        }

    erase_profile = cfg2.erase_profile
    mask_metrics = mask_bundle.get("metrics") or {}
    mask_quality = _assess_mask_quality(mask_metrics, cfg=cfg2)
    if erase_profile == "balanced" and int(mask_metrics.get("embedded_regions") or 0) > 0:
        erase_profile = str(mask_metrics.get("profile_suggestion") or "aggressive")

    refine_mode = str(cfg2.mask_refine_mode or "heuristic").strip().lower()
    if refine_mode in {"sam_auto", "sam_required"} and regions and np.count_nonzero(coarse_mask) > 0:
        try:
            sam_resp = _call_sam_refine_http(pil_img=pil_img, coarse_mask=coarse_mask, regions=regions, cfg=cfg2)
            refined_mask = (sam_resp.get("mask") > 0).astype(np.uint8) * 255
            if np.count_nonzero(refined_mask) > 0:
                mask = refined_mask
                overlay_img = sam_resp.get("overlay") or overlay_img
                sam_metrics = {
                    "sam_refine_status": "used",
                    "sam_refine_backend": sam_resp.get("backend"),
                    "sam_mask_pixels": int(np.count_nonzero(refined_mask)),
                    "sam_mask_coverage": round(float(np.count_nonzero(refined_mask)) / float(max(1, refined_mask.size)), 6),
                    "sam_region_metrics": (sam_resp.get("metrics") or {}).get("region_metrics") or [],
                }
            else:
                sam_metrics = {"sam_refine_status": "empty"}
                if refine_mode == "sam_required":
                    return {
                        "ok": False,
                        "erased_image_bytes": None,
                        "results": {},
                        "metrics": {**mask_metrics, **mask_quality, **sam_metrics},
                        "error": {"code": "sam_empty_mask", "message": "SAM refinement returned an empty mask."},
                    }
        except Exception as e:
            if isinstance(e, RemoteServiceError):
                sam_metrics = {"sam_refine_status": "failed", "sam_refine_error": e.to_dict()}
            else:
                sam_metrics = {"sam_refine_status": "failed", "sam_refine_error": str(e)}
            if refine_mode == "sam_required":
                return {
                    "ok": False,
                    "erased_image_bytes": None,
                    "results": {},
                    "metrics": {**mask_metrics, **mask_quality, **sam_metrics},
                    "error": (
                        e.to_dict()
                        if isinstance(e, RemoteServiceError)
                        else {"code": "sam_refine_failed", "message": str(e)}
                    ),
                }

    region_masks = _split_page_mask_into_region_masks(
        mask,
        regions=regions,
        image_size=pil_img.size,
        assign_expand_px=max(6, int(cfg2.hybrid_seed_safe_expand_px)),
    )
    region_layouts: Dict[int, Dict[str, Any]] = {}
    for idx, region in enumerate(regions, start=1):
        try:
            region_id = int(region.get("region_id") or idx)
        except Exception:
            continue
        region_mask = region_masks.get(region_id)
        placement_lines = _build_placement_lines_from_region_mask(region_mask) if region_mask is not None else []
        region_layouts[region_id] = {
            "placement_source": "region_mask_lines" if placement_lines else "bbox_fallback",
            "placement_lines": placement_lines,
            "placement_line_count": len(placement_lines),
            "placement_mask_pixels": int(np.count_nonzero(region_mask)) if region_mask is not None else 0,
        }

    erased, erase_response = _call_lama_http(pil_img=pil_img, mask=mask, cfg=replace(cfg2, erase_profile=erase_profile))
    actual_backend = str((erase_response or {}).get("backend") or cfg2.lama_backend)
    actual_profile = str((erase_response or {}).get("profile") or erase_profile)

    out: Dict[str, Any] = {
        "ok": True,
        "erased_image_bytes": _pil_to_png_bytes(erased),
        "coarse_mask_bytes": _pil_to_png_bytes(Image.fromarray(mask_bundle["coarse_mask"], mode="L")),
        "mask_bytes": _pil_to_png_bytes(Image.fromarray(mask, mode="L")),
        "mask_overlay_bytes": _pil_to_png_bytes(overlay_img),
        "results": {
            int(r.get("region_id")): {
                "erase_ok": True,
                **(region_layouts.get(int(r.get("region_id"))) or {}),
            }
            for r in regions
            if str(r.get("region_id")).strip().isdigit()
        },
        "metrics": {
            "erase_backend": actual_backend,
            "erase_profile": actual_profile,
            "mask_source_mode": mask_source_mode,
            **mask_metrics,
            **mask_quality,
            **sam_metrics,
            "final_mask_pixels": int(np.count_nonzero(mask)),
            "final_mask_coverage": round(float(np.count_nonzero(mask)) / float(max(1, mask.size)), 6),
        },
        "error": None,
    }

    if auto_detect_bundle is not None:
        out["auto_detect_mask_bytes"] = _pil_to_png_bytes(Image.fromarray(auto_detect_bundle["raw_mask"], mode="L"))
        if auto_detect_bundle.get("heatmap") is not None:
            out["auto_detect_heatmap_bytes"] = _pil_to_png_bytes(auto_detect_bundle["heatmap"])

    return out


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

    page_font_candidates: List[int] = []
    for r in regions:
        try:
            region_id = int(r.get("region_id"))
        except Exception:
            continue
        text = str(r.get("translated_text") or r.get("tgt_text") or "").strip()
        if not text:
            continue
        render_mode = _render_mode_for_region(r, override=render_mode_override)
        if render_mode != "bubble_fit":
            continue
        cfg_region = _cfg_for_render_mode(cfg2, render_mode)
        placement_lines = [x for x in (r.get("placement_lines") or []) if isinstance(x, dict)]
        bbox = r.get("bbox_xywh") or r.get("bbox_xyxy") or r.get("bbox")
        xywh = resolve_region_bbox_xywh(bbox, W=W, H=H)
        if placement_lines:
            fs, _, overflow, _ = _fit_text_to_placement_lines(text, placement_lines=placement_lines, cfg=cfg_region)
            if not overflow:
                page_font_candidates.append(int(fs))
                continue
        if xywh:
            _, _, w, h = xywh
            pad = int(round(min(w, h) * cfg_region.padding_ratio))
            inner_w = max(1, w - 2 * pad)
            inner_h = max(1, h - 2 * pad)
            fs, _, overflow = _fit_text_to_box(text, box_w=inner_w, box_h=inner_h, cfg=cfg_region)
            if not overflow:
                page_font_candidates.append(int(fs))

    page_uniform_font_size: Optional[int] = None
    if page_font_candidates:
        sorted_sizes = sorted(page_font_candidates)
        page_uniform_font_size = int(round(float(np.median(np.asarray(sorted_sizes, dtype=np.float32)))))

    results: Dict[int, Dict[str, Any]] = {}

    for r in regions:
        try:
            region_id = int(r.get("region_id"))
        except Exception:
            continue

        text = str(r.get("translated_text") or r.get("tgt_text") or "").strip()
        bbox = r.get("bbox_xywh") or r.get("bbox_xyxy") or r.get("bbox")
        placement_lines = [x for x in (r.get("placement_lines") or []) if isinstance(x, dict)]
        xywh = resolve_region_bbox_xywh(bbox, W=W, H=H)
        render_mode = _render_mode_for_region(r, override=render_mode_override)
        cfg_region = _cfg_for_render_mode(cfg2, render_mode)
        fill, stroke_fill = _colors_for_region(r, render_mode)
        target_font_size = page_uniform_font_size if render_mode == "bubble_fit" and page_uniform_font_size else None

        if not xywh and not placement_lines:
            results[region_id] = {
                "render_ok": False,
                "overflow": False,
                "render_params": {"reason": "bad_bbox", "render_mode": render_mode},
            }
            continue

        if not text:
            results[region_id] = {
                "render_ok": True,
                "overflow": False,
                "render_params": {
                    "font_size": 0,
                    "lines": 0,
                    "render_mode": render_mode,
                    "placement_source": "placement_lines" if placement_lines else "bbox",
                    "line_layouts": [],
                },
            }
            continue

        fs = cfg_region.min_font_size
        lines: List[str] = [text]
        overflow = True
        font = _pick_font(fs)
        ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (fs, 0)
        line_h = int(round((ascent + descent) * cfg_region.line_spacing))
        placement_source = "bbox"
        used_slots: List[Dict[str, int]] = []
        pad = 0
        line_layouts: List[Dict[str, Any]] = []

        if placement_lines:
            fs, lines, overflow, used_slots = _fit_text_to_placement_lines(
                text,
                placement_lines=placement_lines,
                cfg=cfg_region,
                max_font_size=target_font_size,
            )
            placement_source = "placement_lines"
            font = _pick_font(fs)
            ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (fs, 0)
            line_h = int(round((ascent + descent) * cfg_region.line_spacing))

            if overflow or not used_slots or len(lines) != len(used_slots):
                placement_source = "bbox_fallback"
                used_slots = []

        if not placement_lines or placement_source == "bbox_fallback":
            assert xywh is not None
            x, y, w, h = xywh
            pad = int(round(min(w, h) * cfg_region.padding_ratio))
            inner_w = max(1, w - 2 * pad)
            inner_h = max(1, h - 2 * pad)
            fs, lines, overflow = _fit_text_to_box(
                text,
                box_w=inner_w,
                box_h=inner_h,
                cfg=cfg_region,
                max_font_size=target_font_size,
            )
            font = _pick_font(fs)
            ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (fs, 0)
            line_h = int(round((ascent + descent) * cfg_region.line_spacing))
            total_h = line_h * len(lines)
            start_y = y + pad + max(0, int(round((inner_h - total_h) / 2.0)))
            for i, ln in enumerate(lines):
                yy = start_y + i * line_h
                lw = int(round(draw.textlength(ln, font=font)))
                xx = x + pad + max(0, int(round((inner_w - lw) / 2.0)))
                line_layouts.append(
                    {
                        "text": ln,
                        "draw_x": int(xx),
                        "draw_y": int(yy),
                        "text_width": int(lw),
                        "line_height": int(line_h),
                        "usable_width": int(inner_w),
                        "usable_height": int(line_h),
                        "x_left": int(x + pad),
                        "x_right": int(x + pad + inner_w),
                        "y_top": int(yy),
                        "y_bottom": int(yy + line_h),
                        "x_center": int(round(xx + (lw / 2.0))),
                        "y_center": int(round(yy + (line_h / 2.0))),
                    }
                )
                draw.text(
                    (xx, yy),
                    ln,
                    fill=fill,
                    font=font,
                    stroke_width=int(cfg_region.stroke_width),
                    stroke_fill=stroke_fill,
                )
        else:
            band_top = min(int(slot.get("y_top", slot.get("y_center", 0))) for slot in used_slots)
            band_bottom = max(int(slot.get("y_bottom", slot.get("y_center", 0))) for slot in used_slots)
            total_h = line_h * len(lines)
            start_y = band_top + max(0, int(round(((band_bottom - band_top) - total_h) / 2.0)))
            for i, (slot, ln) in enumerate(zip(used_slots, lines)):
                lw = int(round(draw.textlength(ln, font=font)))
                xx = int(round(int(slot.get("x_center", 0)) - (lw / 2.0)))
                yy = int(round(start_y + (i * line_h)))
                line_layouts.append(
                    {
                        "text": ln,
                        "draw_x": int(xx),
                        "draw_y": int(yy),
                        "text_width": int(lw),
                        "line_height": int(line_h),
                        "usable_width": int(slot.get("usable_width", 0)),
                        "usable_height": int(slot.get("usable_height", 0)),
                        "x_left": int(slot.get("x_left", xx)),
                        "x_right": int(slot.get("x_right", xx + lw)),
                        "y_top": int(slot.get("y_top", yy)),
                        "y_bottom": int(slot.get("y_bottom", yy + line_h)),
                        "x_center": int(slot.get("x_center", xx + (lw / 2.0))),
                        "y_center": int(slot.get("y_center", yy + (line_h / 2.0))),
                    }
                )
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
                "placement_source": placement_source,
                "placement_line_count": len(placement_lines),
                "placement_lines_used": len(used_slots),
                "page_uniform_font_size": int(page_uniform_font_size) if page_uniform_font_size else None,
                "line_layouts": line_layouts,
            },
        }

    qa_report = evaluate_typeset_page(regions=regions, results=results, metrics={})
    for region_id, report in ((qa_report.get("metrics") or {}).get("region_reports") or {}).items():
        rid = int(region_id)
        if rid in results and isinstance(report, dict):
            results[rid]["qa"] = report

    return {
        "ok": True,
        "rendered_image_bytes": _pil_to_png_bytes(out),
        "results": results,
        "metrics": {
            "padding_ratio": cfg2.padding_ratio,
            "line_spacing": cfg2.line_spacing,
            "stroke_width": cfg2.stroke_width,
            "page_uniform_font_size": int(page_uniform_font_size) if page_uniform_font_size else None,
            "qa": qa_report,
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

    enriched_regions: List[Dict[str, Any]] = []
    erase_results = erased.get("results") or {}
    for item in regions:
        try:
            region_id = int(item.get("region_id"))
        except Exception:
            enriched_regions.append(dict(item))
            continue
        merged = dict(item)
        if isinstance(erase_results.get(region_id), dict):
            merged.update(
                {
                    "placement_lines": erase_results[region_id].get("placement_lines") or item.get("placement_lines"),
                    "placement_source": erase_results[region_id].get("placement_source") or item.get("placement_source"),
                }
            )
        enriched_regions.append(merged)

    return typeset_page_image(
        image_bytes=erased["erased_image_bytes"],
        regions=enriched_regions,
        cfg=cfg2,
        skill_params=skill_params,
    )
