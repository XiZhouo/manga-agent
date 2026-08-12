from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _unique_nonempty(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        norm = _norm(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _score_entry(entry: Dict[str, Any], queries: List[str]) -> float:
    src = _norm(str(entry.get("src_text") or ""))
    tgt = _norm(str(entry.get("tgt_text") or ""))
    aliases = [_norm(x) for x in (entry.get("aliases_src") or []) if str(x).strip()]
    tags = [_norm(x) for x in (entry.get("tags") or []) if str(x).strip()]
    situation = entry.get("situation") if isinstance(entry.get("situation"), dict) else {}
    situation_values = [_norm(str(v)) for v in situation.values() if str(v).strip()]
    keys = [src, tgt] + aliases + tags + situation_values

    if not queries:
        score = 0.0
        if entry.get("locked"):
            score += 5.0
        score += 1.0 if str(entry.get("source") or "") == "user_revision" else 0.0
        score += 0.5 if str(entry.get("scope") or "") == "project" else 0.0
        score += 0.5 if str(entry.get("kind") or "").strip() in {"failure_pattern", "successful_strategy"} else 0.0
        return score

    best = 0.0
    for q in queries:
        if not q:
            continue
        for key in keys:
            if not key:
                continue
            if key == q:
                best = max(best, 12.0)
            elif key in q or q in key:
                best = max(best, 8.0)
            else:
                best = max(best, 4.0 * SequenceMatcher(None, key, q).ratio())

    if entry.get("locked"):
        best += 2.0
    return best


def _top_entries(entries: List[Dict[str, Any]], *, queries: List[str], limit: int) -> List[Dict[str, Any]]:
    ranked: List[tuple[float, Dict[str, Any]]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        score = _score_entry(entry, queries)
        if queries and score < 3.0:
            continue
        ranked.append((score, entry))

    ranked.sort(key=lambda item: (-item[0], str(item[1].get("updated_at") or ""), str(item[1].get("entry_id") or "")))
    return [dict(item[1]) for item in ranked[:limit]]


def _format_term_hints(*, glossary_hits: List[Dict[str, Any]], tm_hits: List[Dict[str, Any]], limit: int = 10) -> str:
    lines: List[str] = []
    for item in (glossary_hits + tm_hits)[:limit]:
        src = str(item.get("src_text") or "").strip()
        tgt = str(item.get("tgt_text") or "").strip()
        if not src and not tgt:
            continue
        marker = " [locked]" if item.get("locked") else ""
        if src and tgt:
            lines.append(f"- {src} -> {tgt}{marker}")
        else:
            lines.append(f"- {tgt or src}{marker}")
    return "\n".join(lines).strip()


def _neighbor_context(job, page, *, limit_regions: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for offset in (-1, 1):
        neighbor = job.pages.get(page.page_index + offset)
        if not neighbor:
            continue
        added = 0
        for region_id in sorted(neighbor.regions.keys()):
            region = neighbor.regions[region_id]
            src = str(region.ocr_text or "").strip()
            tgt = str(region.translated_text or "").strip()
            if not src and not tgt:
                continue
            out.append(
                {
                    "page_index": neighbor.page_index,
                    "region_id": region_id,
                    "src_text": src,
                    "tgt_text": tgt,
                }
            )
            added += 1
            if added >= limit_regions:
                break
    return out


def _neighbor_context_for_page_indices(job, page_indices: List[int], *, limit_regions: int = 8) -> List[Dict[str, Any]]:
    if not page_indices:
        return []

    ordered = sorted(int(x) for x in page_indices)
    out: List[Dict[str, Any]] = []
    for neighbor_index in (ordered[0] - 1, ordered[-1] + 1):
        neighbor = job.pages.get(neighbor_index)
        if not neighbor:
            continue
        added = 0
        for region_id in sorted(neighbor.regions.keys()):
            region = neighbor.regions[region_id]
            src = str(region.ocr_text or "").strip()
            tgt = str(region.translated_text or "").strip()
            if not src and not tgt:
                continue
            out.append(
                {
                    "page_index": neighbor.page_index,
                    "region_id": region_id,
                    "src_text": src,
                    "tgt_text": tgt,
                }
            )
            added += 1
            if added >= limit_regions:
                break
    return out


def _window_page_examples(pages: List[Any], *, limit_pages: int = 6, limit_regions: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for page in pages[:limit_pages]:
        page_regions = []
        for region_id in sorted(page.regions.keys()):
            region = page.regions[region_id]
            src = str(region.ocr_text or "").strip()
            tgt = str(region.translated_text or "").strip()
            if not src and not tgt:
                continue
            page_regions.append(
                {
                    "region_id": region_id,
                    "src_text": src,
                    "tgt_text": tgt,
                }
            )
            if len(page_regions) >= limit_regions:
                break
        out.append({"page_index": page.page_index, "regions": page_regions})
    return out


def _format_page_context(
    *,
    style_hints: List[Dict[str, Any]],
    project_rules: List[Dict[str, Any]],
    neighbors: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    if project_rules:
        lines.append("Project rules:")
        for item in project_rules[:6]:
            src = str(item.get("src_text") or "").strip()
            tgt = str(item.get("tgt_text") or "").strip()
            note = f"{src} -> {tgt}" if src and tgt else (tgt or src)
            if note:
                lines.append(f"- {note}")
    if style_hints:
        lines.append("Style hints:")
        for item in style_hints[:6]:
            src = str(item.get("src_text") or "").strip()
            tgt = str(item.get("tgt_text") or "").strip()
            note = f"{src}: {tgt}" if src and tgt else (tgt or src)
            if note:
                lines.append(f"- {note}")
    if neighbors:
        lines.append("Neighbor page examples:")
        for item in neighbors[:6]:
            src = str(item.get("src_text") or "").strip()
            tgt = str(item.get("tgt_text") or "").strip()
            if src or tgt:
                lines.append(f"- p{item.get('page_index')} r{item.get('region_id')}: {src} => {tgt}")
    return "\n".join(lines).strip()


def _entry_list(snapshot: Dict[str, Any], bucket: str) -> List[Dict[str, Any]]:
    return [x for x in (snapshot.get(bucket, {}) or {}).get("entries", []) if isinstance(x, dict)]


def _recent_entries(entries: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    ordered = sorted(entries, key=lambda x: str(x.get("updated_at") or ""))
    if limit <= 0:
        return ordered
    return ordered[-limit:]


def _format_experience_hints(entries: List[Dict[str, Any]], *, limit: int = 6) -> str:
    lines: List[str] = []
    for item in entries[:limit]:
        kind = str(item.get("kind") or item.get("type") or "experience").strip()
        note = str(item.get("tgt_text") or item.get("note") or "").strip()
        action = str(item.get("recommended_action") or "").strip()
        if not note and not action:
            continue
        tail = f" -> {action}" if action else ""
        lines.append(f"- [{kind}] {note}{tail}")
    return "\n".join(lines).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _page_issue_codes(page_summary: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    if not isinstance(page_summary, dict):
        return codes

    keys = (
        "page_error_code",
        "vlm_quality_reason",
        "vlm_judge_reason_code",
        "erase_mask_quality",
        "erase_qa_status",
        "typeset_qa_status",
        "vlm_qa_status",
    )
    for key in keys:
        value = str(page_summary.get(key) or "").strip().lower()
        if value:
            codes.append(value)

    for qa_key in ("vlm_qa", "erase_qa", "typeset_qa"):
        qa = page_summary.get(qa_key) if isinstance(page_summary.get(qa_key), dict) else {}
        for item in qa.get("issues") or []:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip().lower()
            if code:
                codes.append(code)

    return _unique_nonempty(codes)


def _candidate_stage_names(page_summary: Dict[str, Any]) -> List[str]:
    if not isinstance(page_summary, dict):
        return []

    stage = str(page_summary.get("stage") or "").strip().lower()
    page_error_stage = str(page_summary.get("page_error_stage") or "").strip().lower()
    candidates = [stage, page_error_stage]

    if stage == "translated":
        candidates.extend(["vlm", "translation"])
    elif stage == "erased":
        candidates.append("erase")
    elif stage == "typesetting":
        candidates.extend(["typeset", "lettering"])
        erase_qa_status = str(page_summary.get("erase_qa_status") or "").strip().lower()
        erase_mask_quality = str(page_summary.get("erase_mask_quality") or "").strip().lower()
        embedded_ratio = _safe_float(page_summary.get("embedded_ratio"))
        if erase_qa_status in {"retryable", "bad", "blocked"} or erase_mask_quality in {"under_masked", "over_masked"} or embedded_ratio >= 0.5:
            candidates.append("erase")
    elif stage == "failed":
        if page_error_stage:
            candidates.append(page_error_stage)
        if page_error_stage == "erase":
            candidates.append("erase")
        if page_error_stage == "typeset":
            candidates.extend(["typeset", "lettering"])
        if page_error_stage == "vlm":
            candidates.extend(["vlm", "translation"])

    return _unique_nonempty(candidates)


def _skill_name_for_action_or_stage(action: str, stage: str) -> Optional[str]:
    action = str(action or "").strip().lower()
    stage = str(stage or "").strip().lower()

    action_map = {
        "retry_erase": "lama_eraser",
        "retry_typeset": "lettering",
        "retry_vlm_page": "vlm_page_translate",
        "retry_translation": "vlm_translation_revision",
        "run_revision": "vlm_translation_revision",
    }
    if action in action_map:
        return action_map[action]

    stage_map = {
        "erase": "lama_eraser",
        "typeset": "lettering",
        "lettering": "lettering",
        "vlm": "vlm_page_translate",
        "translation": "vlm_translation_revision",
        "revision": "vlm_translation_revision",
    }
    return stage_map.get(stage)


def _proactive_success_stage_names(page_summary: Dict[str, Any]) -> List[str]:
    if not isinstance(page_summary, dict):
        return []

    stage = str(page_summary.get("stage") or "").strip().lower()
    page_error_stage = str(page_summary.get("page_error_stage") or "").strip().lower()

    if stage == "translated":
        return ["erase", "typeset", "lettering"]
    if stage == "erased":
        return ["typeset", "lettering"]
    if stage == "typesetting":
        return ["typeset", "lettering"]
    if stage == "failed" and page_error_stage == "erase":
        return ["erase"]
    if stage == "failed" and page_error_stage == "typeset":
        return ["typeset", "lettering"]
    return []


def _build_proactive_success_params(experience_hits: List[Dict[str, Any]], *, page_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    summary = page_summary if isinstance(page_summary, dict) else {}
    desired_stages = set(_proactive_success_stage_names(summary))
    if not desired_stages:
        return {"skill_param_suggestions": {}, "matched_entries": [], "summary": ""}

    page_type = str(summary.get("page_type") or "").strip().lower()
    page_codes = set(_page_issue_codes(summary))
    by_skill: Dict[str, Dict[str, Any]] = {}

    for item in experience_hits:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind != "successful_strategy":
            continue
        situation = item.get("situation") if isinstance(item.get("situation"), dict) else {}
        stage = str(situation.get("stage") or "").strip().lower()
        if stage not in desired_stages:
            continue

        skill_name = _skill_name_for_action_or_stage("", stage)
        params = item.get("skill_params") if isinstance(item.get("skill_params"), dict) else {}
        if not skill_name or not params:
            continue

        repeat_count = max(1, _safe_int(item.get("repeat_count"), 1))
        confidence = max(0.0, min(1.0, _safe_float(item.get("confidence"), 0.0)))
        score = 1.0 + min(4, repeat_count) * 0.45 + confidence * 1.2

        situation_page_type = str(situation.get("page_type") or "").strip().lower()
        if page_type and situation_page_type and page_type == situation_page_type:
            score += 0.8

        situation_codes = {
            str(situation.get("error_code") or "").strip().lower(),
            str(situation.get("qa_status") or "").strip().lower(),
        }
        situation_codes.update(
            {
                str(x).strip().lower()
                for x in (situation.get("issue_codes") or [])
                if str(x).strip()
            }
        )
        if page_codes and (page_codes & situation_codes):
            score += 0.6

        prev = by_skill.get(skill_name)
        if prev is None or score > _safe_float(prev.get("score")):
            by_skill[skill_name] = {
                "score": score,
                "params": dict(params),
                "note": str(item.get("tgt_text") or item.get("note") or "").strip(),
                "stage": stage,
                "repeat_count": repeat_count,
            }

    suggestions = {skill: dict(data.get("params") or {}) for skill, data in by_skill.items() if isinstance(data.get("params"), dict)}
    matched_entries = [
        {
            "skill_name": skill,
            "stage": data.get("stage"),
            "score": round(_safe_float(data.get("score")), 3),
            "repeat_count": _safe_int(data.get("repeat_count")),
            "note": str(data.get("note") or "").strip(),
        }
        for skill, data in sorted(by_skill.items(), key=lambda item: (-_safe_float(item[1].get("score")), item[0]))
    ]
    summary_lines = [f"- {item.get('skill_name')}: {item.get('note')}" for item in matched_entries[:3] if str(item.get("note") or "").strip()]

    return {
        "skill_param_suggestions": suggestions,
        "matched_entries": matched_entries,
        "summary": "\n".join(summary_lines).strip(),
    }


def _build_experience_policy(experience_hits: List[Dict[str, Any]], *, page_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not experience_hits:
        return {
            "top_action": None,
            "top_action_score": 0.0,
            "candidate_actions": [],
            "suggested_skill_params": {},
            "summary": "",
        }

    summary = page_summary if isinstance(page_summary, dict) else {}
    candidate_stages = set(_candidate_stage_names(summary))
    page_type = str(summary.get("page_type") or "").strip().lower()
    page_codes = set(_page_issue_codes(summary))

    supported_actions = {
        "ask_user",
        "retry_erase",
        "retry_typeset",
        "retry_vlm_page",
        "retry_translation",
        "run_revision",
        "continue",
    }
    action_stats: Dict[str, Dict[str, Any]] = {}

    for item in experience_hits:
        if not isinstance(item, dict):
            continue
        action = str(item.get("recommended_action") or "").strip().lower()
        if not action or action not in supported_actions:
            continue

        situation = item.get("situation") if isinstance(item.get("situation"), dict) else {}
        situation_stage = str(situation.get("stage") or "").strip().lower()
        situation_page_type = str(situation.get("page_type") or "").strip().lower()
        situation_error_code = str(situation.get("error_code") or "").strip().lower()
        situation_qa_status = str(situation.get("qa_status") or "").strip().lower()
        situation_issue_codes = {
            str(x).strip().lower()
            for x in (situation.get("issue_codes") or [])
            if str(x).strip()
        }

        repeat_count = max(1, _safe_int(item.get("repeat_count"), 1))
        confidence = max(0.0, min(1.0, _safe_float(item.get("confidence"), 0.0)))
        score = 1.0 + min(4, repeat_count) * 0.5 + confidence * 1.5

        if candidate_stages and situation_stage in candidate_stages:
            score += 2.0
        elif situation_stage:
            score -= 0.4

        if page_type and situation_page_type and situation_page_type == page_type:
            score += 0.8

        if situation_error_code and situation_error_code in page_codes:
            score += 1.2
        if situation_qa_status and situation_qa_status in page_codes:
            score += 0.8
        if situation_issue_codes and (situation_issue_codes & page_codes):
            score += 0.8

        if str(item.get("kind") or "").strip().lower() == "successful_strategy":
            score += 0.3

        bucket = action_stats.setdefault(
            action,
            {
                "recommended_action": action,
                "score": 0.0,
                "support_count": 0,
                "repeat_count_total": 0,
                "notes": [],
                "skill_params": {},
            },
        )
        bucket["score"] += score
        bucket["support_count"] += 1
        bucket["repeat_count_total"] += repeat_count

        note = str(item.get("tgt_text") or item.get("note") or "").strip()
        if note and note not in bucket["notes"]:
            bucket["notes"].append(note)

        skill_name = _skill_name_for_action_or_stage(action, situation_stage)
        params = item.get("skill_params") if isinstance(item.get("skill_params"), dict) else {}
        if skill_name and params:
            merged = bucket["skill_params"].setdefault(skill_name, {})
            for key, value in params.items():
                if key not in merged:
                    merged[key] = value

    if not action_stats:
        return {
            "top_action": None,
            "top_action_score": 0.0,
            "candidate_actions": [],
            "suggested_skill_params": {},
            "summary": "",
        }

    candidate_actions = sorted(
        action_stats.values(),
        key=lambda item: (-_safe_float(item.get("score")), -_safe_int(item.get("repeat_count_total")), str(item.get("recommended_action") or "")),
    )
    top = candidate_actions[0]
    notes = top.get("notes") or []
    summary_lines = [f"- {str(x).strip()}" for x in notes[:3] if str(x).strip()]

    return {
        "top_action": top.get("recommended_action"),
        "top_action_score": round(_safe_float(top.get("score")), 3),
        "candidate_actions": [
            {
                "recommended_action": item.get("recommended_action"),
                "score": round(_safe_float(item.get("score")), 3),
                "support_count": _safe_int(item.get("support_count")),
                "repeat_count_total": _safe_int(item.get("repeat_count_total")),
                "notes": list(item.get("notes") or [])[:3],
            }
            for item in candidate_actions[:5]
        ],
        "suggested_skill_params": dict(top.get("skill_params") or {}),
        "summary": "\n".join(summary_lines).strip(),
    }


class RetrievalContextBuilder:
    """Build lightweight retrieval context from local memory buckets."""

    ROUTES: Dict[str, Dict[str, Any]] = {
        "chat": {
            "buckets": ["brain_memory", "style_memory", "project_memory"],
            "mode": "recent_chat_plus_preferences",
        },
        "job_user_message": {
            "buckets": ["brain_memory", "glossary", "translation_memory", "style_memory", "project_memory", "experience_memory"],
            "mode": "task_planning",
        },
        "progress": {
            "buckets": ["brain_memory", "project_memory", "experience_memory"],
            "mode": "progress_resume",
        },
        "vlm": {
            "buckets": ["glossary", "translation_memory", "style_memory", "project_memory"],
            "mode": "preinject_only",
        },
        "translation": {
            "buckets": ["glossary", "translation_memory", "style_memory", "project_memory"],
            "mode": "postalign_only",
        },
        "revision": {
            "buckets": ["glossary", "translation_memory", "style_memory", "project_memory", "brain_memory"],
            "mode": "postalign_window",
        },
        "technical_decision": {
            "buckets": ["experience_memory", "brain_memory", "project_memory"],
            "mode": "technical_decision",
        },
    }

    def route_for_scenario(self, scenario: str) -> Dict[str, Any]:
        clean = str(scenario or "job_user_message").strip().lower()
        route = self.ROUTES.get(clean) or self.ROUTES["job_user_message"]
        return {
            "scenario": clean,
            "mode": route["mode"],
            "buckets": list(route["buckets"]),
        }

    def build_for_vlm(self, job, page, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        route = self.route_for_scenario("vlm")
        glossary_entries = _entry_list(snapshot, "glossary")
        tm_entries = _entry_list(snapshot, "translation_memory")
        style_entries = _entry_list(snapshot, "style_memory")
        project_entries = _entry_list(snapshot, "project_memory")
        neighbors = _neighbor_context(job, page)

        locked_glossary = [x for x in glossary_entries if bool(x.get("locked"))]
        locked_tm = [x for x in tm_entries if bool(x.get("locked"))]
        glossary_hits = _top_entries(locked_glossary or glossary_entries, queries=[], limit=8)
        tm_hits = _top_entries(locked_tm, queries=[], limit=4)
        style_hints = _top_entries(style_entries, queries=[], limit=4)
        project_rules = _top_entries(project_entries, queries=[], limit=6)
        page_context = _format_page_context(style_hints=style_hints, project_rules=project_rules, neighbors=neighbors)

        return {
            "route": route,
            "rag_mode": "preinject_only",
            "glossary_hits": glossary_hits,
            "translation_memory_hits": tm_hits,
            "style_hints": style_hints,
            "project_rules": project_rules,
            "neighbor_examples": neighbors[:3],
            "term_hints": _format_term_hints(glossary_hits=glossary_hits, tm_hits=tm_hits, limit=8),
            "page_context": page_context,
        }

    def build_for_translation(self, job, page, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        route = self.route_for_scenario("translation")
        src_queries = [str(r.ocr_text or "") for r in page.regions.values() if str(r.ocr_text or "").strip()]
        tgt_queries = [str(r.translated_text or "") for r in page.regions.values() if str(r.translated_text or "").strip()]
        queries = _unique_nonempty(src_queries + tgt_queries)
        glossary_entries = _entry_list(snapshot, "glossary")
        tm_entries = _entry_list(snapshot, "translation_memory")
        style_entries = _entry_list(snapshot, "style_memory")
        project_entries = _entry_list(snapshot, "project_memory")
        neighbors = _neighbor_context(job, page)

        glossary_hits = _top_entries(glossary_entries, queries=queries, limit=12)
        tm_hits = _top_entries(tm_entries, queries=queries, limit=12)
        style_hints = _top_entries(style_entries, queries=[], limit=6)
        project_rules = _top_entries(project_entries, queries=[], limit=6)
        page_context = _format_page_context(style_hints=style_hints, project_rules=project_rules, neighbors=neighbors)

        return {
            "route": route,
            "rag_mode": "postalign_only",
            "glossary_hits": glossary_hits,
            "translation_memory_hits": tm_hits,
            "style_hints": style_hints,
            "project_rules": project_rules,
            "neighbor_examples": neighbors[:6],
            "term_hints": _format_term_hints(glossary_hits=glossary_hits, tm_hits=tm_hits, limit=10),
            "page_context": page_context,
        }

    def build_for_revision(self, job, pages: List[Any], snapshot: Dict[str, Any], *, document_range: Dict[str, Any] | None = None) -> Dict[str, Any]:
        route = self.route_for_scenario("revision")
        pages = [p for p in pages if p is not None]
        if not pages:
            return {
                "route": route,
                "rag_mode": "postalign_window",
                "document_range": document_range or {},
                "target_page_indices": [],
                "glossary_hits": [],
                "translation_memory_hits": [],
                "style_hints": [],
                "project_rules": [],
                "neighbor_examples": [],
                "target_page_examples": [],
                "term_hints": "",
                "page_context": "",
                "window_context": "",
            }

        src_queries: List[str] = []
        tgt_queries: List[str] = []
        for page in pages:
            src_queries.extend([str(r.ocr_text or "") for r in page.regions.values() if str(r.ocr_text or "").strip()])
            tgt_queries.extend([str(r.translated_text or "") for r in page.regions.values() if str(r.translated_text or "").strip()])
        queries = _unique_nonempty(src_queries + tgt_queries)

        glossary_entries = _entry_list(snapshot, "glossary")
        tm_entries = _entry_list(snapshot, "translation_memory")
        style_entries = _entry_list(snapshot, "style_memory")
        project_entries = _entry_list(snapshot, "project_memory")

        glossary_hits = _top_entries(glossary_entries, queries=queries, limit=16)
        tm_hits = _top_entries(tm_entries, queries=queries, limit=16)
        style_hints = _top_entries(style_entries, queries=[], limit=6)
        project_rules = _top_entries(project_entries, queries=[], limit=6)
        target_page_indices = [int(page.page_index) for page in pages]
        neighbors = _neighbor_context_for_page_indices(job, target_page_indices)
        target_examples = _window_page_examples(sorted(pages, key=lambda p: p.page_index))
        page_context = _format_page_context(style_hints=style_hints, project_rules=project_rules, neighbors=neighbors)

        range_obj = document_range or {
            "start_page_index": min(target_page_indices),
            "end_page_index": max(target_page_indices),
            "page_count": len(target_page_indices),
            "page_indices": sorted(target_page_indices),
            "label": f"{min(target_page_indices)}-{max(target_page_indices)}",
        }
        window_lines = [
            f"Revision window: {range_obj.get('label')}",
            f"Target pages: {', '.join(str(x) for x in sorted(target_page_indices))}",
        ]
        if target_examples:
            window_lines.append("Window page examples:")
            for item in target_examples[:6]:
                page_index = item.get("page_index")
                regions = item.get("regions") or []
                sample = "; ".join(
                    f"r{r.get('region_id')}: {str(r.get('src_text') or '').strip()} => {str(r.get('tgt_text') or '').strip()}"
                    for r in regions[:3]
                    if str(r.get("src_text") or "").strip() or str(r.get("tgt_text") or "").strip()
                )
                if sample:
                    window_lines.append(f"- p{page_index}: {sample}")

        return {
            "route": route,
            "rag_mode": "postalign_window",
            "document_range": range_obj,
            "target_page_indices": sorted(target_page_indices),
            "glossary_hits": glossary_hits,
            "translation_memory_hits": tm_hits,
            "style_hints": style_hints,
            "project_rules": project_rules,
            "neighbor_examples": neighbors[:8],
            "target_page_examples": target_examples,
            "term_hints": _format_term_hints(glossary_hits=glossary_hits, tm_hits=tm_hits, limit=12),
            "page_context": page_context,
            "window_context": "\n".join(window_lines).strip(),
        }

    def build_for_chat(self, job, *, user_message: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        route = self.route_for_scenario("chat")
        queries = [_norm(str(user_message or ""))]
        brain_entries = _entry_list(snapshot, "brain_memory")
        style_entries = _entry_list(snapshot, "style_memory")
        project_entries = _entry_list(snapshot, "project_memory")

        relevant_brain = _top_entries(brain_entries, queries=queries, limit=6)
        recent_brain = _recent_entries(brain_entries, limit=6)
        style_hints = _top_entries(style_entries, queries=[], limit=4)
        project_rules = _top_entries(project_entries, queries=[], limit=4)

        lines: List[str] = []
        if project_rules:
            lines.append("Project rules:")
            for item in project_rules[:4]:
                src = str(item.get("src_text") or "").strip()
                tgt = str(item.get("tgt_text") or "").strip()
                note = f"{src}: {tgt}" if src and tgt else (tgt or src)
                if note:
                    lines.append(f"- {note}")
        if style_hints:
            lines.append("Style hints:")
            for item in style_hints[:4]:
                src = str(item.get("src_text") or "").strip()
                tgt = str(item.get("tgt_text") or "").strip()
                note = f"{src}: {tgt}" if src and tgt else (tgt or src)
                if note:
                    lines.append(f"- {note}")
        if relevant_brain:
            lines.append("Relevant recent brain memory:")
            for item in relevant_brain[:4]:
                msg = str(item.get("src_text") or "").strip()
                summary = str(item.get("tgt_text") or "").strip()
                if msg or summary:
                    lines.append(f"- user: {msg} | summary: {summary}")
        elif recent_brain:
            lines.append("Recent brain memory:")
            for item in recent_brain[-4:]:
                msg = str(item.get("src_text") or "").strip()
                summary = str(item.get("tgt_text") or "").strip()
                if msg or summary:
                    lines.append(f"- user: {msg} | summary: {summary}")

        return {
            "route": route,
            "brain_memory_hits": relevant_brain,
            "recent_brain_events": recent_brain,
            "style_hints": style_hints,
            "project_rules": project_rules,
            "chat_summary": "\n".join(lines).strip(),
        }

    def build_for_progress(self, job, *, user_message: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        route = self.route_for_scenario("progress")
        queries = [_norm(str(user_message or ""))]
        brain_entries = _entry_list(snapshot, "brain_memory")
        project_entries = _entry_list(snapshot, "project_memory")
        experience_entries = _entry_list(snapshot, "experience_memory")

        relevant_brain = _top_entries(brain_entries, queries=queries, limit=8)
        recent_brain = _recent_entries(brain_entries, limit=8)
        project_rules = _top_entries(project_entries, queries=[], limit=6)
        experience_hits = _top_entries(experience_entries, queries=queries, limit=6)

        lines: List[str] = []
        if project_rules:
            lines.append("Project rules:")
            for item in project_rules[:4]:
                src = str(item.get("src_text") or "").strip()
                tgt = str(item.get("tgt_text") or "").strip()
                note = f"{src}: {tgt}" if src and tgt else (tgt or src)
                if note:
                    lines.append(f"- {note}")
        if relevant_brain or recent_brain:
            lines.append("Recent job memory:")
            for item in (relevant_brain or recent_brain)[-6:]:
                msg = str(item.get("src_text") or "").strip()
                summary = str(item.get("tgt_text") or "").strip()
                if msg or summary:
                    lines.append(f"- user: {msg} | summary: {summary}")
        if experience_hits:
            lines.append("Relevant experience:")
            hint_text = _format_experience_hints(experience_hits, limit=4)
            if hint_text:
                lines.append(hint_text)

        return {
            "route": route,
            "brain_memory_hits": relevant_brain,
            "recent_brain_events": recent_brain,
            "project_rules": project_rules,
            "experience_hits": experience_hits,
            "progress_summary": "\n".join(lines).strip(),
        }

    def build_for_technical_decision(
        self,
        job,
        *,
        user_message: str,
        snapshot: Dict[str, Any],
        page_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        route = self.route_for_scenario("technical_decision")
        queries = [_norm(str(user_message or ""))]
        experience_entries = _entry_list(snapshot, "experience_memory")
        brain_entries = _entry_list(snapshot, "brain_memory")
        project_entries = _entry_list(snapshot, "project_memory")

        if isinstance(page_summary, dict):
            for key in (
                "page_error_code",
                "page_error_message",
                "erase_backend",
                "erase_mask_quality",
                "page_type",
                "stage",
                "page_error_stage",
                "vlm_quality_reason",
                "vlm_judge_reason_code",
                "erase_qa_status",
                "typeset_qa_status",
                "vlm_qa_status",
            ):
                value = str(page_summary.get(key) or "").strip()
                if value:
                    queries.append(_norm(value))
            for code in _page_issue_codes(page_summary):
                queries.append(code)

        experience_hits = _top_entries(experience_entries, queries=_unique_nonempty(queries), limit=8)
        relevant_brain = _top_entries(brain_entries, queries=_unique_nonempty(queries), limit=6)
        project_rules = _top_entries(project_entries, queries=[], limit=4)
        experience_policy = _build_experience_policy(experience_hits, page_summary=page_summary)
        proactive_success = _build_proactive_success_params(experience_hits, page_summary=page_summary)

        lines: List[str] = []
        if experience_hits:
            lines.append("Relevant experience memory:")
            hint_text = _format_experience_hints(experience_hits, limit=6)
            if hint_text:
                lines.append(hint_text)
        if experience_policy.get("top_action"):
            lines.append(
                f"Experience policy: prefer {experience_policy.get('top_action')} (score={experience_policy.get('top_action_score')})."
            )
            policy_summary = str(experience_policy.get("summary") or "").strip()
            if policy_summary:
                lines.append(policy_summary)
        if (proactive_success.get("skill_param_suggestions") or {}):
            lines.append("Reusable successful skill params are available.")
            proactive_summary = str(proactive_success.get("summary") or "").strip()
            if proactive_summary:
                lines.append(proactive_summary)
        if relevant_brain:
            lines.append("Relevant recent brain memory:")
            for item in relevant_brain[:4]:
                msg = str(item.get("src_text") or "").strip()
                summary = str(item.get("tgt_text") or "").strip()
                if msg or summary:
                    lines.append(f"- user: {msg} | summary: {summary}")
        if project_rules:
            lines.append("Project rules:")
            for item in project_rules[:4]:
                src = str(item.get("src_text") or "").strip()
                tgt = str(item.get("tgt_text") or "").strip()
                note = f"{src}: {tgt}" if src and tgt else (tgt or src)
                if note:
                    lines.append(f"- {note}")

        return {
            "route": route,
            "experience_hits": experience_hits,
            "experience_policy": experience_policy,
            "proactive_success_params": proactive_success,
            "brain_memory_hits": relevant_brain,
            "project_rules": project_rules,
            "technical_summary": "\n".join(lines).strip(),
        }

    def build_for_brain(self, job, *, user_message: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        route = self.route_for_scenario("job_user_message")
        queries = [_norm(str(user_message or ""))]
        glossary_entries = _entry_list(snapshot, "glossary")
        style_entries = _entry_list(snapshot, "style_memory")
        project_entries = _entry_list(snapshot, "project_memory")
        brain_entries = _entry_list(snapshot, "brain_memory")
        tm_entries = _entry_list(snapshot, "translation_memory")
        experience_entries = _entry_list(snapshot, "experience_memory")

        relevant_brain = _top_entries(brain_entries, queries=queries, limit=8)
        recent_brain = _recent_entries(brain_entries, limit=6)
        glossary_hits = _top_entries(glossary_entries, queries=queries, limit=8)
        tm_hits = _top_entries(tm_entries, queries=queries, limit=8)
        style_hints = _top_entries(style_entries, queries=[], limit=6)
        project_rules = _top_entries(project_entries, queries=[], limit=6)
        experience_hits = _top_entries(experience_entries, queries=queries, limit=6)
        if not experience_hits:
            experience_hits = _recent_entries(experience_entries, limit=4)

        lines: List[str] = []
        if project_rules:
            lines.append("Project rules:")
            for item in project_rules[:6]:
                src = str(item.get("src_text") or "").strip()
                tgt = str(item.get("tgt_text") or "").strip()
                note = f"{src}: {tgt}" if src and tgt else (tgt or src)
                if note:
                    lines.append(f"- {note}")
        if style_hints:
            lines.append("Style hints:")
            for item in style_hints[:6]:
                src = str(item.get("src_text") or "").strip()
                tgt = str(item.get("tgt_text") or "").strip()
                note = f"{src}: {tgt}" if src and tgt else (tgt or src)
                if note:
                    lines.append(f"- {note}")
        if glossary_hits or tm_hits:
            lines.append("Relevant terminology:")
            for item in (glossary_hits + tm_hits)[:8]:
                src = str(item.get("src_text") or "").strip()
                tgt = str(item.get("tgt_text") or "").strip()
                if src or tgt:
                    lines.append(f"- {src} -> {tgt}")
        if relevant_brain:
            lines.append("Similar prior user intents:")
            for item in relevant_brain[:6]:
                msg = str(item.get("src_text") or "").strip()
                summary = str(item.get("tgt_text") or "").strip()
                if msg or summary:
                    lines.append(f"- user: {msg} | summary: {summary}")
        elif recent_brain:
            lines.append("Recent user intents:")
            for item in recent_brain[-4:]:
                msg = str(item.get("src_text") or "").strip()
                summary = str(item.get("tgt_text") or "").strip()
                if msg or summary:
                    lines.append(f"- user: {msg} | summary: {summary}")
        if experience_hits:
            lines.append("Relevant technical memory:")
            hint_text = _format_experience_hints(experience_hits, limit=4)
            if hint_text:
                lines.append(hint_text)

        return {
            "route": route,
            "brain_memory_hits": relevant_brain,
            "recent_brain_events": recent_brain,
            "glossary_hits": glossary_hits,
            "translation_memory_hits": tm_hits,
            "style_hints": style_hints,
            "project_rules": project_rules,
            "experience_hits": experience_hits,
            "brain_summary": "\n".join(lines).strip(),
        }
