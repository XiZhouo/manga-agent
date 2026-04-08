from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskRefinementConfig:
    mask_min_coverage: float = 0.01
    mask_max_coverage: float = 0.25
    mask_dilate_px: int = 2

    embedded_mask_min_coverage: float = 0.02
    embedded_mask_max_coverage: float = 0.45
    embedded_mask_dilate_px: int = 4

    open_kernel_px: int = 3
    close_kernel_px: int = 3
    min_component_area_ratio: float = 0.0008
    max_component_area_ratio: float = 0.35


def safe_bbox_xywh(bbox: List[int], W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in bbox]
    except Exception:
        return None

    if w <= 1 or h <= 1:
        return None

    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    w = max(1, min(W - x, w))
    h = max(1, min(H - y, h))
    return (x, y, w, h)


def coerce_has_text_box(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "boxed", "bubble", "caption", "narration"}:
        return True
    if text in {"false", "0", "no", "n", "embedded", "art", "art_integrated"}:
        return False
    return None


def _coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def _kernel(px: int) -> np.ndarray:
    size = max(1, int(px))
    return np.ones((size, size), np.uint8)


def _select_candidate(
    candidates: List[np.ndarray],
    *,
    has_text_box: bool,
    cfg: MaskRefinementConfig,
) -> np.ndarray:
    target = 0.08 if has_text_box else 0.16
    cov_min = cfg.mask_min_coverage if has_text_box else cfg.embedded_mask_min_coverage
    cov_max = cfg.mask_max_coverage if has_text_box else cfg.embedded_mask_max_coverage

    ranked: List[Tuple[int, float, float]] = []
    for idx, cand in enumerate(candidates):
        cov = _coverage(cand)
        in_range = 0.0 if cov_min <= cov <= cov_max else 1.0
        ranked.append((idx, in_range, abs(cov - target)))

    ranked.sort(key=lambda item: (item[1], item[2]))
    return candidates[ranked[0][0]]


def _filter_components(mask: np.ndarray, *, has_text_box: bool, cfg: MaskRefinementConfig) -> np.ndarray:
    import cv2

    if np.count_nonzero(mask) == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    roi_area = float(mask.shape[0] * mask.shape[1])
    min_area = max(4.0, roi_area * cfg.min_component_area_ratio)
    max_ratio = cfg.max_component_area_ratio if has_text_box else min(0.60, cfg.max_component_area_ratio * 1.8)
    max_area = roi_area * max_ratio

    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        out[labels == label] = 255
    return out


def refine_region_mask(
    roi_rgb: np.ndarray,
    *,
    has_text_box: bool,
    cfg: MaskRefinementConfig,
) -> Dict[str, Any]:
    import cv2

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    dark = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    light = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_otsu = 255 - otsu

    coarse = _select_candidate([dark, light, otsu, inv_otsu], has_text_box=has_text_box, cfg=cfg)
    coarse = (coarse > 0).astype(np.uint8) * 255

    if has_text_box:
        refined = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, _kernel(cfg.open_kernel_px), iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, _kernel(max(2, cfg.close_kernel_px - 1)), iterations=1)
        dilate_px = cfg.mask_dilate_px
    else:
        refined = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, _kernel(max(3, cfg.close_kernel_px + 1)), iterations=1)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, _kernel(max(2, cfg.open_kernel_px - 1)), iterations=1)
        dilate_px = max(cfg.mask_dilate_px + 1, cfg.embedded_mask_dilate_px)

    refined = _filter_components(refined, has_text_box=has_text_box, cfg=cfg)
    if np.count_nonzero(refined) == 0:
        refined = coarse.copy()

    if dilate_px > 0:
        refined = cv2.dilate(refined, _kernel(dilate_px + 1), iterations=1)

    refined = (refined > 0).astype(np.uint8) * 255

    return {
        "coarse_mask": coarse,
        "refined_mask": refined,
        "metrics": {
            "coarse_coverage": round(_coverage(coarse), 6),
            "refined_coverage": round(_coverage(refined), 6),
            "mask_dilate_px": int(dilate_px),
            "has_text_box": bool(has_text_box),
        },
    }


def _mask_overlay(pil_img: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.array(pil_img).astype(np.float32)
    alpha = (mask > 0).astype(np.float32)[..., None] * 0.45
    tint = np.zeros_like(rgb)
    tint[..., 0] = 255.0
    tint[..., 1] = 64.0
    out = rgb * (1.0 - alpha) + tint * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def build_page_mask(
    pil_img: Image.Image,
    regions: List[Dict[str, Any]],
    *,
    cfg: MaskRefinementConfig,
) -> Dict[str, Any]:
    W, H = pil_img.size
    img_rgb = np.array(pil_img)

    coarse_page_mask = np.zeros((H, W), dtype=np.uint8)
    refined_page_mask = np.zeros((H, W), dtype=np.uint8)
    region_metrics: List[Dict[str, Any]] = []
    embedded_regions = 0

    for r in regions:
        bbox = r.get("bbox_xywh") or r.get("bbox")
        xywh = safe_bbox_xywh(bbox, W, H)
        if not xywh:
            continue
        x, y, w, h = xywh

        roi = img_rgb[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        has_text_box = coerce_has_text_box(r.get("has_text_box"))
        boxed = has_text_box is not False
        if not boxed:
            embedded_regions += 1

        refined = refine_region_mask(roi, has_text_box=boxed, cfg=cfg)
        coarse_roi = refined["coarse_mask"]
        refined_roi = refined["refined_mask"]

        coarse_page_mask[y : y + h, x : x + w] = np.maximum(coarse_page_mask[y : y + h, x : x + w], coarse_roi)
        refined_page_mask[y : y + h, x : x + w] = np.maximum(refined_page_mask[y : y + h, x : x + w], refined_roi)

        region_metrics.append(
            {
                "region_id": r.get("region_id"),
                "has_text_box": boxed,
                **(refined.get("metrics") or {}),
            }
        )

    profile_suggestion = "aggressive" if embedded_regions > 0 else "balanced"
    overlay = _mask_overlay(pil_img, refined_page_mask)

    return {
        "coarse_mask": coarse_page_mask,
        "mask": refined_page_mask,
        "mask_overlay": np.array(overlay),
        "metrics": {
            "regions": len(region_metrics),
            "embedded_regions": embedded_regions,
            "coarse_mask_pixels": int(np.count_nonzero(coarse_page_mask)),
            "mask_pixels": int(np.count_nonzero(refined_page_mask)),
            "profile_suggestion": profile_suggestion,
            "region_metrics": region_metrics,
        },
    }
