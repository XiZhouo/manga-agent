from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List


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
    aliases = [_norm(x) for x in (entry.get("aliases_src") or []) if str(x).strip()]
    keys = [src] + aliases

    if not queries:
        score = 0.0
        if entry.get("locked"):
            score += 5.0
        score += 1.0 if str(entry.get("source") or "") == "user_revision" else 0.0
        score += 0.5 if str(entry.get("scope") or "") == "project" else 0.0
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


class RetrievalContextBuilder:
    """Build lightweight retrieval context from local memory buckets."""

    def build_for_vlm(self, job, page, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        glossary_entries = [x for x in (snapshot.get("glossary", {}) or {}).get("entries", []) if isinstance(x, dict)]
        tm_entries = [x for x in (snapshot.get("translation_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        style_entries = [x for x in (snapshot.get("style_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        project_entries = [x for x in (snapshot.get("project_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        neighbors = _neighbor_context(job, page)

        locked_glossary = [x for x in glossary_entries if bool(x.get("locked"))]
        locked_tm = [x for x in tm_entries if bool(x.get("locked"))]
        glossary_hits = _top_entries(locked_glossary or glossary_entries, queries=[], limit=8)
        tm_hits = _top_entries(locked_tm, queries=[], limit=4)
        style_hints = _top_entries(style_entries, queries=[], limit=4)
        project_rules = _top_entries(project_entries, queries=[], limit=6)
        page_context = _format_page_context(style_hints=style_hints, project_rules=project_rules, neighbors=neighbors)

        return {
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
        src_queries = [str(r.ocr_text or "") for r in page.regions.values() if str(r.ocr_text or "").strip()]
        tgt_queries = [str(r.translated_text or "") for r in page.regions.values() if str(r.translated_text or "").strip()]
        queries = _unique_nonempty(src_queries + tgt_queries)
        glossary_entries = [x for x in (snapshot.get("glossary", {}) or {}).get("entries", []) if isinstance(x, dict)]
        tm_entries = [x for x in (snapshot.get("translation_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        style_entries = [x for x in (snapshot.get("style_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        project_entries = [x for x in (snapshot.get("project_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        neighbors = _neighbor_context(job, page)

        glossary_hits = _top_entries(glossary_entries, queries=queries, limit=12)
        tm_hits = _top_entries(tm_entries, queries=queries, limit=12)
        style_hints = _top_entries(style_entries, queries=[], limit=6)
        project_rules = _top_entries(project_entries, queries=[], limit=6)
        page_context = _format_page_context(style_hints=style_hints, project_rules=project_rules, neighbors=neighbors)

        return {
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
        pages = [p for p in pages if p is not None]
        if not pages:
            return {
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

        glossary_entries = [x for x in (snapshot.get("glossary", {}) or {}).get("entries", []) if isinstance(x, dict)]
        tm_entries = [x for x in (snapshot.get("translation_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        style_entries = [x for x in (snapshot.get("style_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        project_entries = [x for x in (snapshot.get("project_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]

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

    def build_for_brain(self, job, *, user_message: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        queries = [_norm(str(user_message or ""))]
        glossary_entries = [x for x in (snapshot.get("glossary", {}) or {}).get("entries", []) if isinstance(x, dict)]
        style_entries = [x for x in (snapshot.get("style_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        project_entries = [x for x in (snapshot.get("project_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        brain_entries = [x for x in (snapshot.get("brain_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]
        tm_entries = [x for x in (snapshot.get("translation_memory", {}) or {}).get("entries", []) if isinstance(x, dict)]

        relevant_brain = _top_entries(brain_entries, queries=queries, limit=8)
        recent_brain = sorted(brain_entries, key=lambda x: str(x.get("updated_at") or ""))[-6:]
        glossary_hits = _top_entries(glossary_entries, queries=queries, limit=8)
        tm_hits = _top_entries(tm_entries, queries=queries, limit=8)
        style_hints = _top_entries(style_entries, queries=[], limit=6)
        project_rules = _top_entries(project_entries, queries=[], limit=6)

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

        return {
            "brain_memory_hits": relevant_brain,
            "recent_brain_events": recent_brain,
            "glossary_hits": glossary_hits,
            "translation_memory_hits": tm_hits,
            "style_hints": style_hints,
            "project_rules": project_rules,
            "brain_summary": "\n".join(lines).strip(),
        }
