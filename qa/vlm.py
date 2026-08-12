from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .base import build_qa_report


def _has_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            return True
    return False


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def _evaluate_bbox_geometry(
    *,
    regions: List[Dict[str, Any]],
    image_size: Optional[List[int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not (isinstance(image_size, list) and len(image_size) == 2 and all(isinstance(x, int) and x > 0 for x in image_size)):
        return [], {
            "boxed_region_count": 0,
            "bbox_hard_fail_count": 0,
            "bbox_soft_warn_count": 0,
        }

    page_w, page_h = int(image_size[0]), int(image_size[1])
    page_area = float(page_w * page_h)
    boxed_infos: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    hard_fail_count = 0
    soft_warn_count = 0

    for region in regions:
        bbox = region.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            x, y, w, h = [int(round(float(v))) for v in bbox]
        except Exception:
            continue
        if w <= 0 or h <= 0:
            issues.append(
                {
                    "region_id": region.get("region_id"),
                    "code": "invalid_bbox",
                    "severity": "high",
                    "message": "region bbox has non-positive width or height",
                }
            )
            hard_fail_count += 1
            continue

        area = float(w * h)
        area_ratio = area / page_area if page_area > 0 else 0.0
        src = str(region.get("src_text") or "").strip()
        tgt = str(region.get("tgt_text") or "").strip()
        char_count = max(len(src), len(tgt))
        chars_per_10k = (char_count * 10000.0 / area) if area > 0 else 0.0
        boxed = region.get("has_text_box") is not False

        if boxed:
            boxed_infos.append(
                {
                    "region_id": region.get("region_id"),
                    "bbox": [x, y, w, h],
                    "area_ratio": area_ratio,
                    "char_count": char_count,
                    "chars_per_10k": chars_per_10k,
                }
            )

            if area_ratio >= 0.05 and char_count <= 80 and chars_per_10k < 3.0:
                issues.append(
                    {
                        "region_id": region.get("region_id"),
                        "code": "bbox_too_large_for_text",
                        "severity": "high",
                        "message": (
                            f"boxed bbox is too large for its text "
                            f"(area={area_ratio:.3f}, chars={char_count}, density={chars_per_10k:.2f}/10k)"
                        ),
                    }
                )
                hard_fail_count += 1
            elif area_ratio >= 0.035 and char_count <= 48 and chars_per_10k < 2.2:
                issues.append(
                    {
                        "region_id": region.get("region_id"),
                        "code": "bbox_looks_loose",
                        "severity": "medium",
                        "message": (
                            f"boxed bbox looks loose "
                            f"(area={area_ratio:.3f}, chars={char_count}, density={chars_per_10k:.2f}/10k)"
                        ),
                    }
                )
                soft_warn_count += 1

    for idx in range(len(boxed_infos)):
        for jdx in range(idx + 1, len(boxed_infos)):
            a = boxed_infos[idx]
            b = boxed_infos[jdx]
            iou = _bbox_iou(a["bbox"], b["bbox"])
            if iou < 0.55:
                continue
            if (
                a["area_ratio"] >= 0.04
                and b["area_ratio"] >= 0.04
                and a["chars_per_10k"] < 4.0
                and b["chars_per_10k"] < 4.0
            ):
                issues.append(
                    {
                        "region_id": a["region_id"],
                        "related_region_id": b["region_id"],
                        "code": "overlapping_oversized_bboxes",
                        "severity": "high",
                        "message": f"oversized boxed bboxes overlap heavily (IoU={iou:.2f})",
                    }
                )
                hard_fail_count += 1

    return issues, {
        "boxed_region_count": len(boxed_infos),
        "bbox_hard_fail_count": hard_fail_count,
        "bbox_soft_warn_count": soft_warn_count,
    }


def evaluate_vlm_page(
    *,
    page_type: str,
    regions: List[Dict[str, Any]],
    target_language: str,
    image_size: Optional[List[int]] = None,
    detector_found_text: Optional[bool] = None,
    extra_warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    clean_page_type = str(page_type or "content").strip().lower()
    if clean_page_type in {"cover", "info"}:
        return build_qa_report(
            stage="vlm",
            status="good",
            score=1.0,
            metrics={
                "page_type": clean_page_type,
                "region_count": 0,
                "skipped_text_pipeline": True,
            },
        )

    issues: List[Dict[str, Any]] = []
    warnings = [str(x) for x in (extra_warnings or []) if str(x).strip()]
    region_count = len(regions)
    boxed_regions = sum(1 for r in regions if r.get("has_text_box") is not False)
    embedded_regions = max(0, region_count - boxed_regions)

    if region_count == 0:
        if detector_found_text is True:
            issues.append(
                {
                    "code": "zero_regions_but_detector_found_text",
                    "severity": "high",
                    "message": "VLM returned zero regions but local detector still found text.",
                }
            )
            return build_qa_report(
                stage="vlm",
                status="bad",
                score=0.12,
                metrics={
                    "region_count": 0,
                    "boxed_regions": 0,
                    "embedded_regions": 0,
                    "detector_found_text": True,
                    "warnings": warnings,
                },
                issues=issues,
                retry_hint={"preferred_action": "retry_vlm_page"},
            )

        return build_qa_report(
            stage="vlm",
            status="good",
            score=0.92,
            metrics={
                "region_count": 0,
                "boxed_regions": 0,
                "embedded_regions": 0,
                "detector_found_text": bool(detector_found_text),
                "warnings": warnings,
            },
        )

    geometry_issues, geometry_metrics = _evaluate_bbox_geometry(regions=regions, image_size=image_size)
    issues.extend(geometry_issues)

    src_nonempty = 0
    tgt_nonempty = 0
    tgt_cjk = 0
    empty_src_ids: List[int] = []
    empty_tgt_ids: List[int] = []

    for region in regions:
        region_id = int(region.get("region_id") or 0)
        src = str(region.get("src_text") or "").strip()
        tgt = str(region.get("tgt_text") or "").strip()
        if src:
            src_nonempty += 1
        else:
            empty_src_ids.append(region_id)
        if tgt:
            tgt_nonempty += 1
        else:
            empty_tgt_ids.append(region_id)
        if tgt and _has_cjk(tgt):
            tgt_cjk += 1

    src_ratio = src_nonempty / float(max(region_count, 1))
    tgt_ratio = tgt_nonempty / float(max(region_count, 1))
    cjk_ratio = (tgt_cjk / float(tgt_nonempty)) if tgt_nonempty else 0.0

    if region_count >= 3 and src_ratio < 0.34:
        issues.append(
            {
                "code": "too_many_empty_src_text",
                "severity": "high",
                "message": f"only {src_nonempty}/{region_count} regions have non-empty source text",
                "region_ids": empty_src_ids,
            }
        )

    if region_count >= 3 and tgt_ratio < 0.34:
        issues.append(
            {
                "code": "too_many_empty_tgt_text",
                "severity": "high",
                "message": f"only {tgt_nonempty}/{region_count} regions have non-empty translated text",
                "region_ids": empty_tgt_ids,
            }
        )

    if str(target_language or "").strip().lower() == "zh" and tgt_nonempty >= 3 and cjk_ratio < 0.34:
        issues.append(
            {
                "code": "target_language_mismatch",
                "severity": "high",
                "message": f"translated text looks insufficiently Chinese ({tgt_cjk}/{tgt_nonempty} regions contain CJK)",
            }
        )

    high_issues = [item for item in issues if item.get("severity") == "high"]
    medium_issues = [item for item in issues if item.get("severity") == "medium"]
    status = "good"
    if high_issues:
        status = "bad"
    elif geometry_metrics.get("bbox_soft_warn_count", 0) >= max(2, boxed_regions // 2) or medium_issues:
        status = "retryable"

    retry_hint: Dict[str, Any] = {}
    if status in {"retryable", "bad"}:
        retry_hint["preferred_action"] = "retry_vlm_page"

    score = 1.0
    score -= min(0.7, 0.24 * len(high_issues))
    score -= min(0.18, 0.08 * len(medium_issues))
    score -= min(0.12, max(0.0, 0.5 - src_ratio))
    score -= min(0.12, max(0.0, 0.5 - tgt_ratio))
    if str(target_language or "").strip().lower() == "zh" and tgt_nonempty:
        score -= min(0.12, max(0.0, 0.5 - cjk_ratio))
    score = max(0.0, score)

    return build_qa_report(
        stage="vlm",
        status=status,
        score=score,
        metrics={
            "page_type": clean_page_type,
            "region_count": region_count,
            "boxed_regions": boxed_regions,
            "embedded_regions": embedded_regions,
            "src_nonempty": src_nonempty,
            "tgt_nonempty": tgt_nonempty,
            "tgt_cjk": tgt_cjk,
            "src_nonempty_ratio": round(src_ratio, 4),
            "tgt_nonempty_ratio": round(tgt_ratio, 4),
            "tgt_cjk_ratio": round(cjk_ratio, 4) if tgt_nonempty else None,
            "detector_found_text": detector_found_text,
            "warnings": warnings,
            **geometry_metrics,
        },
        issues=issues,
        retry_hint=retry_hint,
    )
