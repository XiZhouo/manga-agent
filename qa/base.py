from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_qa_report(
    *,
    stage: str,
    status: str,
    score: float,
    metrics: Optional[Dict[str, Any]] = None,
    issues: Optional[List[Dict[str, Any]]] = None,
    retry_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "stage": str(stage),
        "status": str(status),
        "score": round(max(0.0, min(1.0, float(score))), 4),
        "metrics": dict(metrics or {}),
        "issues": list(issues or []),
        "retry_hint": dict(retry_hint or {}),
    }
