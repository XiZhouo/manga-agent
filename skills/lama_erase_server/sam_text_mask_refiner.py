from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SAMTextMaskRefineConfig:
    positive_points_per_region: int = 4
    negative_points_per_region: int = 8
    edge_protect_px: int = 10
    min_coarse_recall: float = 0.72
    boxed_target_area_ratio: float = 0.16
    embedded_target_area_ratio: float = 0.28
    boxed_max_area_ratio: float = 0.45
    embedded_max_area_ratio: float = 0.70
    mask_dilate_px: int = 0


def xywh_to_xyxy(box: List[int]) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in box]
    except Exception:
        return None
    if w <= 1 or h <= 1:
        return None
    return (x, y, x + w, y + h)


def normalize_bbox_xyxy(box: List[int], *, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    except Exception:
        return None
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def region_bbox_xyxy(region: Dict[str, Any], *, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    box = region.get("bbox_xyxy")
    if isinstance(box, list) and len(box) == 4:
        return normalize_bbox_xyxy(box, width=width, height=height)
    box = region.get("bbox_xywh") or region.get("bbox")
    xyxy = xywh_to_xyxy(box) if isinstance(box, list) else None
    if xyxy is None:
        return None
    return normalize_bbox_xyxy(list(xyxy), width=width, height=height)


def _component_centers(mask_u8: np.ndarray, *, limit: int) -> List[Tuple[int, int]]:
    import cv2

    if limit <= 0 or np.count_nonzero(mask_u8) == 0:
        return []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    comps: List[Tuple[float, int, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        cx, cy = centroids[label]
        comps.append((float(area), int(round(cx)), int(round(cy))))
    comps.sort(key=lambda item: item[0], reverse=True)
    return [(x, y) for _, x, y in comps[:limit]]


def _sample_mask_points(mask_u8: np.ndarray, *, limit: int) -> List[Tuple[int, int]]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or limit <= 0:
        return []
    points: List[Tuple[int, int]] = []
    qs = np.linspace(0.15, 0.85, num=min(limit, 4))
    for q in qs:
        tx = float(np.quantile(xs, q))
        ty = float(np.quantile(ys, q))
        dist = ((xs.astype(np.float32) - tx) ** 2) + ((ys.astype(np.float32) - ty) ** 2)
        idx = int(np.argmin(dist))
        points.append((int(xs[idx]), int(ys[idx])))
    return points


def _dedupe_points(points: List[Tuple[int, int]], *, limit: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    for pt in points:
        if pt in seen:
            continue
        seen.add(pt)
        out.append(pt)
        if len(out) >= limit:
            break
    return out


def build_region_prompts(
    *,
    coarse_mask: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    has_text_box: bool,
    cfg: SAMTextMaskRefineConfig,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox_xyxy
    roi_mask = (coarse_mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255

    pos_points = _component_centers(roi_mask, limit=cfg.positive_points_per_region)
    if len(pos_points) < cfg.positive_points_per_region:
        pos_points.extend(
            _sample_mask_points(
                roi_mask,
                limit=max(0, cfg.positive_points_per_region - len(pos_points)),
            )
        )
    if not pos_points:
        pos_points = [((x2 - x1) // 2, (y2 - y1) // 2)]

    inset = max(2, min(cfg.edge_protect_px, max(2, min((x2 - x1) // 4, (y2 - y1) // 4))))
    neg_local = [
        (inset, inset),
        ((x2 - x1) // 2, inset),
        (max(1, (x2 - x1) - inset - 1), inset),
        (inset, (y2 - y1) // 2),
        (max(1, (x2 - x1) - inset - 1), (y2 - y1) // 2),
        (inset, max(1, (y2 - y1) - inset - 1)),
        ((x2 - x1) // 2, max(1, (y2 - y1) - inset - 1)),
        (max(1, (x2 - x1) - inset - 1), max(1, (y2 - y1) - inset - 1)),
    ]
    if has_text_box:
        neg_points = neg_local
    else:
        neg_points = neg_local[: max(4, cfg.negative_points_per_region // 2)]

    pos_global = _dedupe_points([(x1 + px, y1 + py) for px, py in pos_points], limit=cfg.positive_points_per_region)
    neg_global = _dedupe_points([(x1 + px, y1 + py) for px, py in neg_points], limit=cfg.negative_points_per_region)

    return {
        "positive_points": pos_global,
        "negative_points": neg_global,
        "positive_labels": [1] * len(pos_global),
        "negative_labels": [0] * len(neg_global),
    }


def _edge_ring_mask(shape: Tuple[int, int], bbox_xyxy: Tuple[int, int, int, int], edge_px: int) -> np.ndarray:
    H, W = shape
    x1, y1, x2, y2 = bbox_xyxy
    ring = np.zeros((H, W), dtype=np.uint8)
    if edge_px <= 0:
        return ring
    ring[y1:y2, x1:x2] = 1
    inner_x1 = min(x2, x1 + edge_px)
    inner_y1 = min(y2, y1 + edge_px)
    inner_x2 = max(x1, x2 - edge_px)
    inner_y2 = max(y1, y2 - edge_px)
    if inner_x2 > inner_x1 and inner_y2 > inner_y1:
        ring[inner_y1:inner_y2, inner_x1:inner_x2] = 0
    return ring


def score_sam_candidate(
    *,
    candidate_mask: np.ndarray,
    coarse_mask: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    has_text_box: bool,
    cfg: SAMTextMaskRefineConfig,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox_xyxy
    bbox_area = max(1.0, float((x2 - x1) * (y2 - y1)))
    cand = (candidate_mask > 0).astype(np.uint8)
    coarse = (coarse_mask > 0).astype(np.uint8)

    cand_bbox = cand[y1:y2, x1:x2]
    coarse_bbox = coarse[y1:y2, x1:x2]
    cand_count = int(np.count_nonzero(cand_bbox))
    coarse_count = int(np.count_nonzero(coarse_bbox))
    inter = int(np.count_nonzero((cand_bbox > 0) & (coarse_bbox > 0)))

    coarse_recall = float(inter) / float(coarse_count) if coarse_count > 0 else 1.0
    area_ratio = float(cand_count) / bbox_area

    ring = _edge_ring_mask(cand.shape, bbox_xyxy, max(1, cfg.edge_protect_px))
    ring_total = int(np.count_nonzero(ring))
    ring_hit = int(np.count_nonzero((cand > 0) & (ring > 0)))
    edge_overlap = float(ring_hit) / float(ring_total) if ring_total > 0 else 0.0

    target_area = cfg.boxed_target_area_ratio if has_text_box else cfg.embedded_target_area_ratio
    max_area = cfg.boxed_max_area_ratio if has_text_box else cfg.embedded_max_area_ratio
    area_penalty = abs(area_ratio - target_area)
    oversize_penalty = max(0.0, area_ratio - max_area)
    recall_penalty = max(0.0, cfg.min_coarse_recall - coarse_recall)

    score = (
        (4.5 * coarse_recall)
        - (2.8 * edge_overlap)
        - (2.2 * area_penalty)
        - (4.0 * oversize_penalty)
        - (6.0 * recall_penalty)
    )

    return {
        "score": round(float(score), 6),
        "coarse_recall": round(float(coarse_recall), 6),
        "area_ratio": round(float(area_ratio), 6),
        "edge_overlap": round(float(edge_overlap), 6),
        "oversize_penalty": round(float(oversize_penalty), 6),
        "recall_penalty": round(float(recall_penalty), 6),
        "candidate_pixels": cand_count,
    }


def choose_best_sam_candidate(
    *,
    candidate_masks: List[np.ndarray],
    candidate_scores: Optional[List[float]],
    coarse_mask: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    has_text_box: bool,
    cfg: SAMTextMaskRefineConfig,
) -> Dict[str, Any]:
    best_index = -1
    best_metrics: Dict[str, Any] = {}
    best_mask: Optional[np.ndarray] = None

    for idx, mask in enumerate(candidate_masks):
        metrics = score_sam_candidate(
            candidate_mask=mask,
            coarse_mask=coarse_mask,
            bbox_xyxy=bbox_xyxy,
            has_text_box=has_text_box,
            cfg=cfg,
        )
        if candidate_scores and idx < len(candidate_scores):
            metrics["sam_score"] = round(float(candidate_scores[idx]), 6)
            metrics["score"] = round(float(metrics["score"]) + (0.15 * float(candidate_scores[idx])), 6)
        if best_index < 0 or float(metrics["score"]) > float(best_metrics.get("score", -1e9)):
            best_index = idx
            best_metrics = metrics
            best_mask = mask

    return {
        "index": best_index,
        "mask": best_mask,
        "metrics": best_metrics,
    }
