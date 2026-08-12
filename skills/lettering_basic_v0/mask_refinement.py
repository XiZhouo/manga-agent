from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskRefinementConfig:
    mask_min_coverage: float = 0.01
    mask_max_coverage: float = 0.25
    mask_dilate_px: int = 2
    box_expand_px: int = 1

    embedded_mask_min_coverage: float = 0.02
    embedded_mask_max_coverage: float = 0.45
    embedded_mask_dilate_px: int = 4
    embedded_box_expand_px: int = 6

    open_kernel_px: int = 3
    close_kernel_px: int = 3
    complete_mask_close_px: int = 3
    embedded_complete_mask_close_px: int = 7
    complete_mask_dilate_px: int = 0
    embedded_complete_mask_dilate_px: int = 2
    fill_holes: bool = True
    min_component_area_ratio: float = 0.0008
    max_component_area_ratio: float = 0.35

    # Remote auto-detect constraint pass
    auto_mask_box_expand_px: int = 12
    auto_mask_embedded_box_expand_px: int = 28
    auto_mask_min_component_area_px: int = 40
    auto_mask_min_component_overlap_ratio: float = 0.15
    auto_mask_min_component_overlap_px: int = 24
    auto_mask_final_erode_px: int = 1

    # Heatmap-constrained v2 pass.
    # Use VLM bbox only to assign/clip heatmap seeds; do not rasterize bbox rectangles into the final mask.
    heatmap_threshold: int = 1
    heatmap_component_assign_expand_px: int = 8
    heatmap_safe_expand_px: int = 3
    heatmap_min_component_area_px: int = 12
    heatmap_min_component_overlap_px: int = 8
    heatmap_min_component_overlap_ratio: float = 0.02
    heatmap_close_px: int = 2
    heatmap_dilate_px: int = 0
    heatmap_max_growth_ratio: float = 1.35

    # DC-Nightwing tuned default:
    # grow a heatmap seed inside the raw manga-cleaner envelope until residue is low
    # without jumping straight to the full raw detect blob.
    hybrid_seed_safe_expand_px: int = 5
    hybrid_envelope_safe_expand_px: int = 40
    hybrid_growth_iterations: int = 15
    hybrid_growth_kernel_px: int = 3


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


def safe_bbox_xyxy(bbox: List[int], W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    except Exception:
        return None

    if x1 <= x0 or y1 <= y0:
        return None

    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(x0 + 1, min(W, x1))
    y1 = max(y0 + 1, min(H, y1))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def expand_bbox_xywh(xywh: Tuple[int, int, int, int], W: int, H: int, expand_px: int) -> Tuple[int, int, int, int]:
    x, y, w, h = xywh
    pad = max(0, int(expand_px))
    if pad <= 0:
        return xywh

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def resolve_region_bbox_xywh(
    bbox: Any,
    *,
    W: int,
    H: int,
    guide_mask: Optional[np.ndarray] = None,
    expand_px: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    candidates: List[Tuple[int, int, int, int]] = []
    if isinstance(bbox, list) and len(bbox) == 4:
        xywh = safe_bbox_xywh(bbox, W, H)
        if xywh:
            candidates.append(xywh)
        xyxy = safe_bbox_xyxy(bbox, W, H)
        if xyxy and xyxy not in candidates:
            candidates.append(xyxy)

    if not candidates:
        return None
    if len(candidates) == 1 or guide_mask is None or guide_mask.size == 0 or np.count_nonzero(guide_mask) == 0:
        chosen = candidates[0]
        return expand_bbox_xywh(chosen, W, H, expand_px) if expand_px > 0 else chosen

    best = candidates[0]
    best_score = float("-inf")
    binary = (guide_mask > 0).astype(np.uint8)
    for cand in candidates:
        x, y, w, h = expand_bbox_xywh(cand, W, H, expand_px) if expand_px > 0 else cand
        region = binary[y : y + h, x : x + w]
        overlap = float(np.count_nonzero(region))
        area = float(max(1, w * h))
        score = overlap / np.sqrt(area)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    return best


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


def _fill_component_holes(mask: np.ndarray) -> np.ndarray:
    import cv2

    if np.count_nonzero(mask) == 0:
        return mask

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    for contour in contours:
        cv2.drawContours(out, [contour], -1, 255, thickness=-1)
    return out


def refine_mask(
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


def complete_mask(
    mask: np.ndarray,
    *,
    has_text_box: bool,
    cfg: MaskRefinementConfig,
) -> Dict[str, Any]:
    import cv2

    if np.count_nonzero(mask) == 0:
        return {
            "completed_mask": mask,
            "metrics": {
                "completion_applied": False,
                "completed_coverage": 0.0,
                "complete_mask_close_px": 0,
                "complete_mask_dilate_px": 0,
            },
        }

    close_px = cfg.complete_mask_close_px if has_text_box else cfg.embedded_complete_mask_close_px
    dilate_px = cfg.complete_mask_dilate_px if has_text_box else cfg.embedded_complete_mask_dilate_px

    completed = mask.copy()
    if close_px > 0:
        completed = cv2.morphologyEx(completed, cv2.MORPH_CLOSE, _kernel(close_px), iterations=1)

    if cfg.fill_holes:
        completed = _fill_component_holes(completed)

    completed = _filter_components(completed, has_text_box=has_text_box, cfg=cfg)
    if np.count_nonzero(completed) == 0:
        completed = mask.copy()

    if dilate_px > 0:
        completed = cv2.dilate(completed, _kernel(dilate_px + 1), iterations=1)

    completed = (completed > 0).astype(np.uint8) * 255
    return {
        "completed_mask": completed,
        "metrics": {
            "completion_applied": (close_px > 0) or bool(cfg.fill_holes) or (dilate_px > 0),
            "completed_coverage": round(_coverage(completed), 6),
            "complete_mask_close_px": int(close_px),
            "complete_mask_dilate_px": int(dilate_px),
        },
    }


def refine_region_mask(
    roi_rgb: np.ndarray,
    *,
    has_text_box: bool,
    cfg: MaskRefinementConfig,
) -> Dict[str, Any]:
    refined = refine_mask(roi_rgb, has_text_box=has_text_box, cfg=cfg)
    completed = complete_mask(refined["refined_mask"], has_text_box=has_text_box, cfg=cfg)

    return {
        "coarse_mask": refined["coarse_mask"],
        "refined_mask": completed["completed_mask"],
        "metrics": {
            **(refined.get("metrics") or {}),
            **(completed.get("metrics") or {}),
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


def _rect_overlap_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return int((ix2 - ix1) * (iy2 - iy1))


def constrain_remote_detect_mask(
    detect_mask: np.ndarray,
    regions: List[Dict[str, Any]],
    *,
    image_size: Tuple[int, int],
    cfg: MaskRefinementConfig,
) -> Dict[str, Any]:
    import cv2

    W, H = image_size
    raw_mask = (detect_mask > 0).astype(np.uint8) * 255
    if raw_mask.shape[:2] != (H, W):
        raw_mask = cv2.resize(raw_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    if np.count_nonzero(raw_mask) == 0 or not regions:
        return {
            "mask": raw_mask,
            "gate_mask": np.zeros((H, W), dtype=np.uint8),
            "metrics": {
                "constraint_applied": False,
                "raw_mask_pixels": int(np.count_nonzero(raw_mask)),
                "constrained_mask_pixels": int(np.count_nonzero(raw_mask)),
                "raw_mask_coverage": round(_coverage(raw_mask), 6),
                "constrained_mask_coverage": round(_coverage(raw_mask), 6),
                "kept_components": 0,
                "dropped_components": 0,
                "boxed_gate_regions": 0,
                "embedded_gate_regions": 0,
            },
        }

    gate_mask = np.zeros((H, W), dtype=np.uint8)
    boxed_gate_mask = np.zeros((H, W), dtype=np.uint8)
    embedded_gate_mask = np.zeros((H, W), dtype=np.uint8)
    gate_regions: List[Dict[str, Any]] = []
    boxed_gate_regions = 0
    embedded_gate_regions = 0

    for region in regions:
        bbox = region.get("bbox_xyxy") or region.get("bbox_xywh") or region.get("bbox")
        boxed = coerce_has_text_box(region.get("has_text_box")) is not False
        expand_px = cfg.auto_mask_box_expand_px if boxed else cfg.auto_mask_embedded_box_expand_px
        xywh = resolve_region_bbox_xywh(bbox, W=W, H=H, guide_mask=raw_mask, expand_px=expand_px)
        if not xywh:
            continue

        x, y, w, h = xywh
        gate_mask[y : y + h, x : x + w] = 255
        if boxed:
            boxed_gate_mask[y : y + h, x : x + w] = 255
            boxed_gate_regions += 1
        else:
            embedded_gate_mask[y : y + h, x : x + w] = 255
            embedded_gate_regions += 1

        gate_regions.append(
            {
                "region_id": region.get("region_id"),
                "has_text_box": boxed,
                "expanded_bbox_xywh": [x, y, w, h],
                "expand_px": int(expand_px),
            }
        )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((raw_mask > 0).astype(np.uint8), connectivity=8)
    constrained_boxed = np.zeros_like(raw_mask)
    constrained_embedded = np.zeros_like(raw_mask)
    kept_components = 0
    dropped_components = 0
    component_metrics: List[Dict[str, Any]] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        component = (labels[y : y + h, x : x + w] == label)
        if area < max(1, int(cfg.auto_mask_min_component_area_px)):
            dropped_components += 1
            component_metrics.append({"label": label, "area": area, "kept": False, "reason": "small_component"})
            continue

        boxed_overlap = int(np.count_nonzero(component & (boxed_gate_mask[y : y + h, x : x + w] > 0)))
        embedded_overlap = int(np.count_nonzero(component & (embedded_gate_mask[y : y + h, x : x + w] > 0)))
        best_overlap = max(boxed_overlap, embedded_overlap)
        overlap_ratio = float(best_overlap) / float(max(1, area))
        keep = best_overlap >= int(cfg.auto_mask_min_component_overlap_px) or overlap_ratio >= float(
            cfg.auto_mask_min_component_overlap_ratio
        )

        if not keep:
            dropped_components += 1
            component_metrics.append(
                {
                    "label": label,
                    "area": area,
                    "kept": False,
                    "reason": "low_region_overlap",
                    "best_overlap_px": best_overlap,
                    "best_overlap_ratio": round(overlap_ratio, 6),
                }
            )
            continue

        target_gate = boxed_gate_mask if boxed_overlap >= embedded_overlap else embedded_gate_mask
        clipped = component & (target_gate[y : y + h, x : x + w] > 0)
        if not np.any(clipped):
            dropped_components += 1
            component_metrics.append(
                {
                    "label": label,
                    "area": area,
                    "kept": False,
                    "reason": "empty_after_clipping",
                    "best_overlap_px": best_overlap,
                }
            )
            continue

        if target_gate is boxed_gate_mask:
            constrained_boxed[y : y + h, x : x + w][clipped] = 255
            component_kind = "boxed"
        else:
            constrained_embedded[y : y + h, x : x + w][clipped] = 255
            component_kind = "embedded"

        kept_components += 1
        component_metrics.append(
            {
                "label": label,
                "area": area,
                "kept": True,
                "kind": component_kind,
                "best_overlap_px": best_overlap,
                "best_overlap_ratio": round(overlap_ratio, 6),
            }
        )

    if cfg.auto_mask_final_erode_px > 0 and np.count_nonzero(constrained_boxed) > 0:
        constrained_boxed = cv2.erode(
            constrained_boxed,
            _kernel(max(1, int(cfg.auto_mask_final_erode_px))),
            iterations=1,
        )

    constrained = np.maximum(constrained_boxed, constrained_embedded)
    constrained = (constrained > 0).astype(np.uint8) * 255

    return {
        "mask": constrained,
        "gate_mask": gate_mask,
        "boxed_gate_mask": boxed_gate_mask,
        "embedded_gate_mask": embedded_gate_mask,
        "metrics": {
            "constraint_applied": True,
            "raw_mask_pixels": int(np.count_nonzero(raw_mask)),
            "constrained_mask_pixels": int(np.count_nonzero(constrained)),
            "raw_mask_coverage": round(_coverage(raw_mask), 6),
            "constrained_mask_coverage": round(_coverage(constrained), 6),
            "kept_components": kept_components,
            "dropped_components": dropped_components,
            "boxed_gate_regions": int(boxed_gate_regions),
            "embedded_gate_regions": int(embedded_gate_regions),
            "auto_mask_box_expand_px": int(cfg.auto_mask_box_expand_px),
            "auto_mask_embedded_box_expand_px": int(cfg.auto_mask_embedded_box_expand_px),
            "auto_mask_min_component_area_px": int(cfg.auto_mask_min_component_area_px),
            "auto_mask_min_component_overlap_ratio": float(cfg.auto_mask_min_component_overlap_ratio),
            "auto_mask_min_component_overlap_px": int(cfg.auto_mask_min_component_overlap_px),
            "auto_mask_final_erode_px": int(cfg.auto_mask_final_erode_px),
            "gate_regions": gate_regions,
            "component_metrics": component_metrics,
        },
    }


def constrain_remote_detect_heatmap(
    heatmap: np.ndarray,
    regions: List[Dict[str, Any]],
    *,
    image_size: Tuple[int, int],
    cfg: MaskRefinementConfig,
) -> Dict[str, Any]:
    import cv2

    W, H = image_size
    raw_heatmap = np.asarray(heatmap)
    if raw_heatmap.ndim == 3:
        raw_heatmap = raw_heatmap[..., 0]
    raw_heatmap = raw_heatmap.astype(np.uint8)
    if raw_heatmap.shape[:2] != (H, W):
        raw_heatmap = cv2.resize(raw_heatmap, (W, H), interpolation=cv2.INTER_NEAREST)

    seed_mask = (raw_heatmap >= max(0, int(cfg.heatmap_threshold))).astype(np.uint8) * 255
    if np.count_nonzero(seed_mask) == 0 or not regions:
        return {
            "mask": seed_mask,
            "seed_mask": seed_mask,
            "metrics": {
                "constraint_applied": False,
                "raw_heatmap_pixels": int(np.count_nonzero(seed_mask)),
                "constrained_mask_pixels": int(np.count_nonzero(seed_mask)),
                "raw_heatmap_coverage": round(_coverage(seed_mask), 6),
                "constrained_mask_coverage": round(_coverage(seed_mask), 6),
                "total_components": 0,
                "assigned_components": 0,
                "dropped_components": 0,
                "region_metrics": [],
            },
        }

    region_specs: List[Dict[str, Any]] = []
    region_seed_masks: Dict[int, np.ndarray] = {}

    for idx, region in enumerate(regions, start=1):
        bbox = region.get("bbox_xyxy") or region.get("bbox_xywh") or region.get("bbox")
        boxed = coerce_has_text_box(region.get("has_text_box")) is not False
        base_xywh = resolve_region_bbox_xywh(bbox, W=W, H=H, guide_mask=seed_mask, expand_px=0)
        if not base_xywh:
            continue

        assign_expand_px = int(cfg.heatmap_component_assign_expand_px)
        safe_expand_px = int(cfg.heatmap_safe_expand_px if boxed else max(cfg.heatmap_safe_expand_px + 2, cfg.embedded_box_expand_px))
        assign_xywh = expand_bbox_xywh(base_xywh, W, H, assign_expand_px)
        safe_xywh = expand_bbox_xywh(base_xywh, W, H, safe_expand_px)
        region_id = int(region.get("region_id") or idx)

        region_specs.append(
            {
                "region_id": region_id,
                "has_text_box": boxed,
                "base_xywh": base_xywh,
                "assign_xywh": assign_xywh,
                "safe_xywh": safe_xywh,
            }
        )
        region_seed_masks[region_id] = np.zeros((H, W), dtype=np.uint8)

    if not region_specs:
        return {
            "mask": np.zeros((H, W), dtype=np.uint8),
            "seed_mask": seed_mask,
            "metrics": {
                "constraint_applied": False,
                "raw_heatmap_pixels": int(np.count_nonzero(seed_mask)),
                "constrained_mask_pixels": 0,
                "raw_heatmap_coverage": round(_coverage(seed_mask), 6),
                "constrained_mask_coverage": 0.0,
                "total_components": 0,
                "assigned_components": 0,
                "dropped_components": 0,
                "region_metrics": [],
            },
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((seed_mask > 0).astype(np.uint8), connectivity=8)
    total_components = max(0, num_labels - 1)
    assigned_components = 0
    dropped_components = 0
    component_metrics: List[Dict[str, Any]] = []

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(1, int(cfg.heatmap_min_component_area_px)):
            dropped_components += 1
            component_metrics.append({"label": label, "area": area, "kept": False, "reason": "small_component"})
            continue

        component = (labels[y : y + h, x : x + w] == label)
        best_spec: Optional[Dict[str, Any]] = None
        best_overlap_px = 0
        best_overlap_ratio = 0.0
        for spec in region_specs:
            ax, ay, aw, ah = spec["assign_xywh"]
            ix0 = max(x, ax)
            iy0 = max(y, ay)
            ix1 = min(x + w, ax + aw)
            iy1 = min(y + h, ay + ah)
            overlap_px = 0
            if ix1 > ix0 and iy1 > iy0:
                comp_crop = component[iy0 - y : iy1 - y, ix0 - x : ix1 - x]
                overlap_px = int(np.count_nonzero(comp_crop))
            overlap_ratio = float(overlap_px) / float(max(1, area))
            if overlap_px > best_overlap_px or (overlap_px == best_overlap_px and overlap_ratio > best_overlap_ratio):
                best_overlap_px = overlap_px
                best_overlap_ratio = overlap_ratio
                best_spec = spec

        if (
            best_spec is None
            or (
                best_overlap_px < int(cfg.heatmap_min_component_overlap_px)
                and best_overlap_ratio < float(cfg.heatmap_min_component_overlap_ratio)
            )
        ):
            dropped_components += 1
            component_metrics.append(
                {
                    "label": label,
                    "area": area,
                    "kept": False,
                    "reason": "low_region_overlap",
                    "best_overlap_px": int(best_overlap_px),
                    "best_overlap_ratio": round(best_overlap_ratio, 6),
                }
            )
            continue

        target = region_seed_masks[int(best_spec["region_id"])]
        target[y : y + h, x : x + w][component] = 255
        assigned_components += 1
        component_metrics.append(
            {
                "label": label,
                "area": area,
                "kept": True,
                "region_id": int(best_spec["region_id"]),
                "best_overlap_px": int(best_overlap_px),
                "best_overlap_ratio": round(best_overlap_ratio, 6),
            }
        )

    final_mask = np.zeros((H, W), dtype=np.uint8)
    region_metrics: List[Dict[str, Any]] = []

    for spec in region_specs:
        region_id = int(spec["region_id"])
        seed = region_seed_masks[region_id]
        seed_pixels = int(np.count_nonzero(seed))
        if seed_pixels == 0:
            region_metrics.append(
                {
                    "region_id": region_id,
                    "has_text_box": bool(spec["has_text_box"]),
                    "seed_pixels": 0,
                    "grown_pixels": 0,
                    "growth_ratio": 0.0,
                    "fallback_used": False,
                    "assigned_component_count": 0,
                    "base_bbox_xywh": list(spec["base_xywh"]),
                    "assign_bbox_xywh": list(spec["assign_xywh"]),
                    "safe_bbox_xywh": list(spec["safe_xywh"]),
                }
            )
            continue

        ys, xs = np.where(seed > 0)
        seed_x0 = int(xs.min())
        seed_y0 = int(ys.min())
        seed_x1 = int(xs.max()) + 1
        seed_y1 = int(ys.max()) + 1

        sx, sy, sw, sh = spec["safe_xywh"]
        process_x0 = max(0, min(seed_x0, sx))
        process_y0 = max(0, min(seed_y0, sy))
        process_x1 = min(W, max(seed_x1, sx + sw))
        process_y1 = min(H, max(seed_y1, sy + sh))
        ax = process_x0
        ay = process_y0
        aw = max(1, process_x1 - process_x0)
        ah = max(1, process_y1 - process_y0)
        roi = seed[ay : ay + ah, ax : ax + aw]
        seed_local = (roi > 0).astype(np.uint8) * 255
        grown = seed_local.copy()

        if cfg.heatmap_close_px > 1 and np.count_nonzero(grown) > 0:
            grown = cv2.morphologyEx(grown, cv2.MORPH_CLOSE, _kernel(int(cfg.heatmap_close_px)), iterations=1)
        if cfg.heatmap_dilate_px > 0 and np.count_nonzero(grown) > 0:
            grown = cv2.dilate(grown, _kernel(int(cfg.heatmap_dilate_px) + 1), iterations=1)

        safe_mask_roi = np.zeros_like(grown)
        rx0 = max(0, sx - ax)
        ry0 = max(0, sy - ay)
        rx1 = min(aw, sx + sw - ax)
        ry1 = min(ah, sy + sh - ay)
        safe_mask_roi[ry0:ry1, rx0:rx1] = 255

        # Keep the original assigned heatmap components intact.
        # Only constrain newly added pixels so VLM bbox controls component count,
        # not the final mask silhouette.
        added = ((grown > 0) & ~(seed_local > 0)).astype(np.uint8) * 255
        added = ((added > 0) & (safe_mask_roi > 0)).astype(np.uint8) * 255
        grown = np.maximum(seed_local, added)

        grown_pixels = int(np.count_nonzero(grown))
        growth_ratio = float(grown_pixels) / float(max(1, seed_pixels))
        fallback_used = False
        if growth_ratio > float(cfg.heatmap_max_growth_ratio):
            fallback_used = True
            grown = seed_local.copy()
            grown_pixels = int(np.count_nonzero(grown))
            growth_ratio = float(grown_pixels) / float(max(1, seed_pixels))

        final_mask[ay : ay + ah, ax : ax + aw] = np.maximum(final_mask[ay : ay + ah, ax : ax + aw], grown)

        assigned_component_count = sum(1 for item in component_metrics if item.get("region_id") == region_id and item.get("kept"))
        region_metrics.append(
            {
                "region_id": region_id,
                "has_text_box": bool(spec["has_text_box"]),
                "seed_pixels": seed_pixels,
                "grown_pixels": grown_pixels,
                "growth_ratio": round(growth_ratio, 6),
                "fallback_used": fallback_used,
                "assigned_component_count": int(assigned_component_count),
                "base_bbox_xywh": list(spec["base_xywh"]),
                "assign_bbox_xywh": list(spec["assign_xywh"]),
                "process_bbox_xywh": [ax, ay, aw, ah],
                "safe_bbox_xywh": list(spec["safe_xywh"]),
            }
        )

    final_mask = (final_mask > 0).astype(np.uint8) * 255
    return {
        "mask": final_mask,
        "seed_mask": seed_mask,
        "metrics": {
            "constraint_applied": True,
            "raw_heatmap_pixels": int(np.count_nonzero(seed_mask)),
            "constrained_mask_pixels": int(np.count_nonzero(final_mask)),
            "raw_heatmap_coverage": round(_coverage(seed_mask), 6),
            "constrained_mask_coverage": round(_coverage(final_mask), 6),
            "total_components": int(total_components),
            "assigned_components": int(assigned_components),
            "dropped_components": int(dropped_components),
            "heatmap_threshold": int(cfg.heatmap_threshold),
            "heatmap_component_assign_expand_px": int(cfg.heatmap_component_assign_expand_px),
            "heatmap_safe_expand_px": int(cfg.heatmap_safe_expand_px),
            "heatmap_min_component_area_px": int(cfg.heatmap_min_component_area_px),
            "heatmap_min_component_overlap_px": int(cfg.heatmap_min_component_overlap_px),
            "heatmap_min_component_overlap_ratio": float(cfg.heatmap_min_component_overlap_ratio),
            "heatmap_close_px": int(cfg.heatmap_close_px),
            "heatmap_dilate_px": int(cfg.heatmap_dilate_px),
            "heatmap_max_growth_ratio": float(cfg.heatmap_max_growth_ratio),
            "region_metrics": region_metrics,
            "component_metrics": component_metrics,
        },
    }


def build_hybrid_detect_mask(
    detect_mask: np.ndarray,
    heatmap: np.ndarray,
    regions: List[Dict[str, Any]],
    *,
    image_size: Tuple[int, int],
    cfg: MaskRefinementConfig,
) -> Dict[str, Any]:
    import cv2

    seed_cfg = replace(
        cfg,
        heatmap_safe_expand_px=int(cfg.hybrid_seed_safe_expand_px),
        heatmap_close_px=0,
        heatmap_dilate_px=0,
        heatmap_max_growth_ratio=max(float(cfg.heatmap_max_growth_ratio), 10.0),
    )
    envelope_cfg = replace(
        cfg,
        heatmap_safe_expand_px=int(cfg.hybrid_envelope_safe_expand_px),
        heatmap_close_px=0,
        heatmap_dilate_px=0,
        heatmap_max_growth_ratio=max(float(cfg.heatmap_max_growth_ratio), 10.0),
    )

    seed_bundle = constrain_remote_detect_heatmap(heatmap, regions, image_size=image_size, cfg=seed_cfg)
    envelope_bundle = constrain_remote_detect_heatmap(detect_mask, regions, image_size=image_size, cfg=envelope_cfg)

    seed_mask = (seed_bundle["mask"] > 0).astype(np.uint8) * 255
    envelope_mask = (envelope_bundle["mask"] > 0).astype(np.uint8) * 255
    grown = seed_mask.copy()
    iterations = max(0, int(cfg.hybrid_growth_iterations))
    kernel = _kernel(max(1, int(cfg.hybrid_growth_kernel_px)))

    for _ in range(iterations):
        grown = cv2.dilate(grown, kernel, iterations=1)
        grown = ((grown > 0) & (envelope_mask > 0)).astype(np.uint8) * 255

    return {
        "mask": grown,
        "seed_mask": seed_mask,
        "envelope_mask": envelope_mask,
        "metrics": {
            "mask_strategy": "manga_auto_hybrid",
            "tuning_comment": "DC-Nightwing",
            "hybrid_growth_iterations": iterations,
            "hybrid_growth_kernel_px": int(cfg.hybrid_growth_kernel_px),
            "hybrid_seed_safe_expand_px": int(cfg.hybrid_seed_safe_expand_px),
            "hybrid_envelope_safe_expand_px": int(cfg.hybrid_envelope_safe_expand_px),
            "seed_mask_pixels": int(np.count_nonzero(seed_mask)),
            "envelope_mask_pixels": int(np.count_nonzero(envelope_mask)),
            "mask_pixels": int(np.count_nonzero(grown)),
            "seed_mask_coverage": round(_coverage(seed_mask), 6),
            "envelope_mask_coverage": round(_coverage(envelope_mask), 6),
            "mask_coverage": round(_coverage(grown), 6),
            "seed_metrics": seed_bundle.get("metrics") or {},
            "envelope_metrics": envelope_bundle.get("metrics") or {},
        },
    }


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
        bbox = r.get("bbox_xyxy") or r.get("bbox_xywh") or r.get("bbox")
        xywh = resolve_region_bbox_xywh(bbox, W=W, H=H)
        if not xywh:
            continue

        has_text_box = coerce_has_text_box(r.get("has_text_box"))
        boxed = has_text_box is not False
        if not boxed:
            embedded_regions += 1

        expand_px = cfg.box_expand_px if boxed else cfg.embedded_box_expand_px
        x, y, w, h = expand_bbox_xywh(xywh, W, H, expand_px)
        roi = img_rgb[y : y + h, x : x + w]
        if roi.size == 0:
            continue

        refined = refine_region_mask(roi, has_text_box=boxed, cfg=cfg)
        coarse_roi = refined["coarse_mask"]
        refined_roi = refined["refined_mask"]

        coarse_page_mask[y : y + h, x : x + w] = np.maximum(coarse_page_mask[y : y + h, x : x + w], coarse_roi)
        refined_page_mask[y : y + h, x : x + w] = np.maximum(refined_page_mask[y : y + h, x : x + w], refined_roi)

        region_metrics.append(
            {
                "region_id": r.get("region_id"),
                "has_text_box": boxed,
                "mask_strategy": "boxed_refine" if boxed else "embedded_refine_complete",
                "original_bbox_xywh": list(xywh),
                "expanded_bbox_xywh": [x, y, w, h],
                "box_expand_px": int(expand_px),
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
            "boxed_regions": len(region_metrics) - embedded_regions,
            "embedded_regions": embedded_regions,
            "image_pixels": int(W * H),
            "coarse_mask_pixels": int(np.count_nonzero(coarse_page_mask)),
            "mask_pixels": int(np.count_nonzero(refined_page_mask)),
            "coarse_mask_coverage": round(_coverage(coarse_page_mask), 6),
            "mask_coverage": round(_coverage(refined_page_mask), 6),
            "mask_strategy": "embedded_refine_complete" if embedded_regions > 0 else "boxed_refine",
            "profile_suggestion": profile_suggestion,
            "region_metrics": region_metrics,
        },
    }
