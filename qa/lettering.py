from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .base import build_qa_report


def _line_gap_stats(line_layouts: List[Dict[str, Any]]) -> Tuple[float, float]:
    ordered = sorted(
        [item for item in line_layouts if isinstance(item, dict)],
        key=lambda item: (float(item.get("draw_y", 0.0)), float(item.get("draw_x", 0.0))),
    )
    if len(ordered) < 2:
        return 0.0, 0.0
    gaps: List[float] = []
    for prev, cur in zip(ordered[:-1], ordered[1:]):
        prev_bottom = float(prev.get("draw_y", 0.0)) + float(prev.get("line_height", 0.0))
        cur_top = float(cur.get("draw_y", 0.0))
        gaps.append(max(0.0, cur_top - prev_bottom))
    if not gaps:
        return 0.0, 0.0
    arr = np.asarray(gaps, dtype=np.float32)
    return float(arr.mean()), float(arr.std() / max(arr.mean(), 1e-6))


def evaluate_typeset_page(
    *,
    regions: List[Dict[str, Any]],
    results: Dict[int, Dict[str, Any]],
    metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics = dict(metrics or {})
    region_inputs: Dict[int, Dict[str, Any]] = {}
    for item in regions:
        try:
            region_inputs[int(item.get("region_id"))] = item
        except Exception:
            continue

    page_font_sizes: List[int] = []
    region_reports: Dict[int, Dict[str, Any]] = {}
    issues: List[Dict[str, Any]] = []
    fill_ratios: List[float] = []
    spacing_cvs: List[float] = []
    bad_regions = 0

    for region_id, result in results.items():
        if not isinstance(result, dict):
            continue
        render_params = result.get("render_params") or {}
        line_layouts = [x for x in (render_params.get("line_layouts") or []) if isinstance(x, dict)]
        placement_source = str(render_params.get("placement_source") or "bbox")
        font_size = int(render_params.get("font_size") or 0)
        line_count = int(render_params.get("lines") or len(line_layouts) or 0)
        placement_line_count = int(render_params.get("placement_line_count") or 0)
        placement_lines_used = int(render_params.get("placement_lines_used") or 0)
        render_mode = str(render_params.get("render_mode") or "bubble_fit")
        overflow = bool(result.get("overflow"))

        if render_mode == "bubble_fit" and font_size > 0:
            page_font_sizes.append(font_size)

        used_area = 0.0
        text_area = 0.0
        for item in line_layouts:
            used_area += float(max(1.0, float(item.get("usable_width", 0.0))) * max(1.0, float(item.get("usable_height", 0.0))))
            text_area += float(max(1.0, float(item.get("text_width", 0.0))) * max(1.0, float(item.get("line_height", 0.0))))
        fill_ratio = float(text_area / max(used_area, 1.0)) if line_layouts else 0.0
        fill_ratios.append(fill_ratio)

        _, line_spacing_cv = _line_gap_stats(line_layouts)
        spacing_cvs.append(line_spacing_cv)

        slot_utilization = (
            float(placement_lines_used) / float(max(placement_line_count, 1))
            if placement_source == "placement_lines" and placement_line_count > 0
            else 1.0
        )

        region_issues: List[Dict[str, Any]] = []
        if overflow:
            region_issues.append(
                {
                    "code": "overflow",
                    "severity": "high",
                    "message": "text overflow remains after layout",
                }
            )
        if placement_source == "placement_lines" and line_count > 1 and fill_ratio < 0.42:
            region_issues.append(
                {
                    "code": "fill_ratio_low",
                    "severity": "medium",
                    "message": f"text only fills {round(fill_ratio, 3)} of the writable area",
                }
            )
        if placement_source == "placement_lines" and placement_line_count > 1 and line_count >= 2 and slot_utilization < 0.6:
            region_issues.append(
                {
                    "code": "slot_utilization_low",
                    "severity": "medium",
                    "message": f"only used {placement_lines_used}/{placement_line_count} writable slots",
                }
            )
        if line_count >= 3 and line_spacing_cv > 0.18:
            region_issues.append(
                {
                    "code": "line_spacing_inconsistent",
                    "severity": "medium",
                    "message": f"line spacing variation is high ({round(line_spacing_cv, 3)})",
                }
            )

        if region_issues:
            bad_regions += 1
            issues.extend([{"region_id": int(region_id), **item} for item in region_issues])

        region_reports[int(region_id)] = build_qa_report(
            stage="typeset_region",
            status="retryable" if region_issues else "good",
            score=max(0.0, 1.0 - (0.22 * len(region_issues))),
            metrics={
                "font_size": font_size,
                "line_count": line_count,
                "fill_ratio": round(fill_ratio, 4),
                "slot_utilization": round(slot_utilization, 4),
                "line_spacing_cv": round(line_spacing_cv, 4),
                "placement_source": placement_source,
                "placement_line_count": placement_line_count,
                "placement_lines_used": placement_lines_used,
            },
            issues=region_issues,
            retry_hint={"preferred_action": "retry_typeset"} if region_issues else {},
        )

    page_font_uniformity_cv = 0.0
    page_uniform_font_size = None
    if page_font_sizes:
        arr = np.asarray(page_font_sizes, dtype=np.float32)
        page_uniform_font_size = float(np.median(arr))
        page_font_uniformity_cv = float(arr.std() / max(arr.mean(), 1e-6))
        if len(arr) >= 3 and page_font_uniformity_cv > 0.14:
            issues.append(
                {
                    "code": "font_size_uniformity_low",
                    "severity": "medium",
                    "message": f"page font size variation is high ({round(page_font_uniformity_cv, 3)})",
                }
            )

    mean_fill_ratio = float(np.mean(np.asarray(fill_ratios, dtype=np.float32))) if fill_ratios else 0.0
    max_spacing_cv = float(np.max(np.asarray(spacing_cvs, dtype=np.float32))) if spacing_cvs else 0.0

    retry_hint: Dict[str, Any] = {}
    if any(item.get("code") == "line_spacing_inconsistent" for item in issues):
        retry_hint.setdefault("preferred_action", "retry_typeset")
        retry_hint.setdefault("skill_params", {})
        retry_hint["skill_params"]["line_spacing"] = 1.1
    if any(item.get("code") == "fill_ratio_low" for item in issues):
        retry_hint.setdefault("preferred_action", "retry_typeset")
        retry_hint.setdefault("skill_params", {})
        retry_hint["skill_params"]["padding_ratio"] = 0.05

    status = "good"
    if any(item.get("severity") == "high" for item in issues):
        status = "bad"
    elif issues:
        status = "retryable"

    score = 1.0
    score -= min(0.6, 0.16 * len(issues))
    score -= min(0.2, max(0.0, 0.45 - mean_fill_ratio))
    score -= min(0.15, page_font_uniformity_cv)
    score = max(0.0, score)

    return build_qa_report(
        stage="typeset",
        status=status,
        score=score,
        metrics={
            "region_count": len(region_reports),
            "bad_region_count": int(bad_regions),
            "mean_fill_ratio": round(mean_fill_ratio, 4),
            "max_line_spacing_cv": round(max_spacing_cv, 4),
            "page_font_uniformity_cv": round(page_font_uniformity_cv, 4),
            "page_uniform_font_size": int(round(page_uniform_font_size)) if page_uniform_font_size else None,
            "region_reports": region_reports,
        },
        issues=issues,
        retry_hint=retry_hint,
    )
