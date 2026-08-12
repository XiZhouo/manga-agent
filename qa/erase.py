from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import build_qa_report


def evaluate_erase_page(
    *,
    page_type: str,
    metrics: Optional[Dict[str, Any]] = None,
    results: Optional[Dict[int, Dict[str, Any]]] = None,
    expected_region_count: Optional[int] = None,
) -> Dict[str, Any]:
    clean_page_type = str(page_type or "content").strip().lower()
    raw_metrics = dict(metrics or {})
    raw_results = results or {}

    service_error = raw_metrics.get("service_error") if isinstance(raw_metrics.get("service_error"), dict) else None

    if clean_page_type in {"cover", "info"}:
        return build_qa_report(
            stage="erase",
            status="good",
            score=1.0,
            metrics={"page_type": clean_page_type, "skipped_text_pipeline": True},
        )

    if service_error:
        error_code = str(service_error.get("code") or "remote_service_error").strip()
        error_message = str(service_error.get("message") or "Remote erase service is unavailable.").strip()
        return build_qa_report(
            stage="erase",
            status="blocked",
            score=0.0,
            metrics={
                "page_type": clean_page_type,
                "erase_backend": raw_metrics.get("erase_backend"),
                "failure_kind": raw_metrics.get("failure_kind") or "remote_service",
                "failure_stage": raw_metrics.get("failure_stage") or "erase_remote_call",
                "service_error": service_error,
            },
            issues=[
                {
                    "code": error_code,
                    "severity": "high",
                    "message": error_message,
                }
            ],
            retry_hint={
                "preferred_action": "ask_user",
                "operator_action": "restart_remote_erase_service",
                "retryable": bool(service_error.get("retryable", True)),
            },
        )

    issues: List[Dict[str, Any]] = []
    region_reports: Dict[int, Dict[str, Any]] = {}
    zero_mask_regions = 0
    zero_line_regions = 0

    for raw_region_id, payload in raw_results.items():
        if not isinstance(payload, dict):
            continue
        try:
            region_id = int(raw_region_id)
        except Exception:
            continue

        placement_source = str(payload.get("placement_source") or "unknown")
        placement_line_count = int(payload.get("placement_line_count") or 0)
        placement_mask_pixels = int(payload.get("placement_mask_pixels") or 0)

        region_issues: List[Dict[str, Any]] = []
        if placement_mask_pixels <= 0:
            zero_mask_regions += 1
            region_issues.append(
                {
                    "code": "region_mask_empty",
                    "severity": "high",
                    "message": "region received no writable mask pixels after erase masking",
                }
            )
        elif placement_line_count <= 0:
            zero_line_regions += 1
            region_issues.append(
                {
                    "code": "placement_lines_missing",
                    "severity": "medium",
                    "message": "region mask exists but did not yield writable placement lines",
                }
            )

        region_reports[region_id] = build_qa_report(
            stage="erase_region",
            status="bad" if any(x.get("severity") == "high" for x in region_issues) else ("retryable" if region_issues else "good"),
            score=max(0.0, 1.0 - (0.28 * len(region_issues))),
            metrics={
                "placement_source": placement_source,
                "placement_line_count": placement_line_count,
                "placement_mask_pixels": placement_mask_pixels,
            },
            issues=region_issues,
            retry_hint={"preferred_action": "retry_erase"} if region_issues else {},
        )
        if region_issues:
            issues.extend([{"region_id": region_id, **item} for item in region_issues])

    mask_quality = str(raw_metrics.get("mask_quality") or "").strip().lower()
    mask_quality_reason = str(raw_metrics.get("mask_quality_reason") or "").strip()
    if mask_quality == "under_masked":
        issues.append(
            {
                "code": "under_masked",
                "severity": "high",
                "message": mask_quality_reason or "mask likely misses text strokes or edge pixels",
            }
        )
    elif mask_quality == "over_masked":
        issues.append(
            {
                "code": "over_masked",
                "severity": "high",
                "message": mask_quality_reason or "mask likely swallows nearby art or bubble edges",
            }
        )

    sam_status = str(raw_metrics.get("sam_refine_status") or "").strip().lower()
    if sam_status in {"failed", "empty"}:
        issues.append(
            {
                "code": f"sam_refine_{sam_status}",
                "severity": "medium",
                "message": f"SAM refinement status is {sam_status}",
            }
        )

    expected = int(expected_region_count or 0)
    actual_regions = len(region_reports)
    if expected > 0 and actual_regions and actual_regions < expected:
        issues.append(
            {
                "code": "missing_region_layouts",
                "severity": "medium",
                "message": f"only {actual_regions}/{expected} regions produced erase placement metadata",
            }
        )

    high_issues = [item for item in issues if item.get("severity") == "high"]
    medium_issues = [item for item in issues if item.get("severity") == "medium"]

    status = "good"
    if high_issues:
        status = "bad"
    elif medium_issues:
        status = "retryable"

    retry_hint = dict(raw_metrics.get("mask_retry_hint") or {})
    if status in {"retryable", "bad"}:
        retry_hint = {"preferred_action": "retry_erase", **retry_hint}

    placement_ready_regions = max(0, actual_regions - zero_mask_regions - zero_line_regions)
    placement_ready_ratio = (
        placement_ready_regions / float(max(actual_regions, 1))
        if actual_regions > 0
        else 0.0
    )

    final_mask_coverage = raw_metrics.get("final_mask_coverage")
    if final_mask_coverage is None:
        final_mask_coverage = raw_metrics.get("mask_coverage")

    score = 1.0
    score -= min(0.72, 0.24 * len(high_issues))
    score -= min(0.18, 0.08 * len(medium_issues))
    score -= min(0.10, max(0.0, 0.75 - placement_ready_ratio))
    score = max(0.0, score)

    return build_qa_report(
        stage="erase",
        status=status,
        score=score,
        metrics={
            "page_type": clean_page_type,
            "expected_region_count": expected or actual_regions,
            "region_count": actual_regions,
            "placement_ready_regions": placement_ready_regions,
            "placement_ready_ratio": round(placement_ready_ratio, 4),
            "zero_mask_regions": zero_mask_regions,
            "zero_line_regions": zero_line_regions,
            "erase_backend": raw_metrics.get("erase_backend"),
            "erase_profile": raw_metrics.get("erase_profile"),
            "mask_source_mode": raw_metrics.get("mask_source_mode"),
            "mask_strategy": raw_metrics.get("mask_strategy"),
            "mask_quality": raw_metrics.get("mask_quality"),
            "mask_quality_reason": raw_metrics.get("mask_quality_reason"),
            "mask_coverage": raw_metrics.get("mask_coverage"),
            "final_mask_coverage": final_mask_coverage,
            "sam_refine_status": raw_metrics.get("sam_refine_status"),
            "region_reports": region_reports,
        },
        issues=issues,
        retry_hint=retry_hint,
    )
