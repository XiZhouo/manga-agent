from __future__ import annotations

import os
import re
import time
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional, Tuple, List, Any, Dict

from .state import (
    JobState,
    PageState,
    RegionState,
    JobStatus,
    PageStage,
    RegionStage,
    AgentAction,
    ActionType,
)
from .store import StateStore
from .decision_engine import DecisionEngine
from .tools import Tools
from .brain import DecisionBrain
from .memory_store import LocalMemoryStore, utc_iso
from .retrieval import RetrievalContextBuilder
from .skill_specs import (
    build_skill_context_for_brain,
    sanitize_job_strategy_suggestions,
    sanitize_skill_param_suggestions,
    sanitize_strategy_suggestions,
)


class SimpleMemory:
    """
    极简 memory：
    负责：
    - 更新用户偏好
    - 维护本地 JSON memory buckets
    - 为翻译/VLM 组装 retrieval context
    """

    def __init__(self) -> None:
        self.store = LocalMemoryStore()
        self.retrieval = RetrievalContextBuilder()

    def initialize_job(self, job: JobState) -> None:
        self.store.ensure_initialized(job)

    def snapshot(self, job: JobState) -> Dict[str, Any]:
        return self.store.snapshot(job)

    def update_preferences(self, job: JobState, user_message: str) -> None:
        if "中文" in user_message:
            job.user_preferences["target_language"] = "zh"
        if "英文" in user_message:
            job.user_preferences["target_language"] = "en"
        if "更夸张" in user_message:
            job.user_preferences["sfx_style"] = "exaggerated"
            self.store.add_style_hint(job, message=user_message, key="sfx_style", value="exaggerated")
        if "术语统一" in user_message:
            job.user_preferences["strict_glossary"] = True
            self.store.add_project_rule(job, message=user_message, key="strict_glossary", value="enabled")

    def build_vlm_retrieval(self, job: JobState, page: PageState) -> Dict[str, Any]:
        return self.retrieval.build_for_vlm(job, page, self.snapshot(job))

    def build_translation_retrieval(self, job: JobState, page: PageState) -> Dict[str, Any]:
        return self.retrieval.build_for_translation(job, page, self.snapshot(job))

    def build_revision_retrieval(
        self,
        job: JobState,
        *,
        pages: List[PageState],
        document_range: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.retrieval.build_for_revision(job, pages, self.snapshot(job), document_range=document_range)

    def build_brain_context(self, job: JobState, *, user_message: str) -> Dict[str, Any]:
        return self.retrieval.build_for_brain(job, user_message=user_message, snapshot=self.snapshot(job))

    def record_brain_event(
        self,
        job: JobState,
        *,
        user_message: str,
        plan: Optional[Dict[str, Any]] = None,
        note: Optional[str] = None,
    ) -> None:
        summary_parts: List[str] = []
        if isinstance(plan, dict) and plan:
            intent = str(plan.get("intent") or "unknown")
            scope = str(plan.get("scope") or "page_only")
            ref = plan.get("reference_page_number")
            from_text = str(plan.get("from_text") or "").strip()
            to_text = str(plan.get("to_text") or "").strip()
            if intent:
                summary_parts.append(f"intent={intent}")
            if scope:
                summary_parts.append(f"scope={scope}")
            if ref is not None:
                summary_parts.append(f"ref={ref}")
            if from_text and to_text:
                summary_parts.append(f"patch={from_text}->{to_text}")
        if note:
            summary_parts.append(str(note))
        summary = "; ".join(summary_parts) if summary_parts else "user_message"
        event_type = str((plan or {}).get("intent") or "message")
        extra = {"plan": plan or {}, "note": note}
        self.store.append_brain_event(
            job,
            event_type=event_type,
            message=user_message,
            summary=summary,
            extra=extra,
        )

    def record_revision_updates(
        self,
        job: JobState,
        *,
        page_index_to_regions: Dict[int, Dict[int, RegionState]],
        patches: List[Dict[str, Any]],
        glossary_updates: List[Dict[str, Any]],
        glossary_type: Optional[str] = None,
    ) -> None:
        glossary_entries: List[Dict[str, Any]] = []
        for item in glossary_updates:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry.setdefault("scope", "project")
            entry.setdefault("source", "user_revision")
            entry.setdefault("locked", bool(entry.get("locked")))
            glossary_entries.append(entry)

        if glossary_entries:
            self.store.upsert_entries(job, "glossary", glossary_entries)

        tm_entries: List[Dict[str, Any]] = []
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            try:
                pi = int(patch.get("page_index"))
                rid = int(patch.get("region_id"))
            except Exception:
                continue
            region = (page_index_to_regions.get(pi) or {}).get(rid)
            if not region:
                continue
            src_text = str(region.ocr_text or "").strip()
            tgt_text = str(patch.get("new_tgt") or "").strip()
            if not src_text or not tgt_text:
                continue
            tm_entries.append(
                {
                    "type": str(glossary_type or "phrase"),
                    "src_text": src_text,
                    "tgt_text": tgt_text,
                    "scope": "project",
                    "source": "user_revision",
                    "locked": False,
                    "page_index": pi,
                    "region_id": rid,
                    "expected_old_tgt": patch.get("expected_old_tgt"),
                    "why": patch.get("why"),
                    "updated_at": utc_iso(),
                }
            )

        if tm_entries:
            self.store.upsert_entries(job, "translation_memory", tm_entries)

        self.store.sync_job_glossary(job)


class ComicAgent:
    """
    中央 Agent / Orchestrator。

    职责：
    1. 创建任务
    2. 读取状态
    3. 调用决策引擎决定下一步
    4. 执行动作（调用 tools）
    5. 更新状态
    6. 支持用户反馈后局部重算
    """

    def __init__(
        self,
        store: StateStore,
        tools: Tools,
        decision_engine: DecisionEngine,
        memory: Optional[SimpleMemory] = None,
        brain: Optional[DecisionBrain] = None,
    ) -> None:
        self.store = store
        self.tools = tools
        self.decision_engine = decision_engine
        self.memory = memory or SimpleMemory()
        self.brain = brain or DecisionBrain()

    def _resolve_plan_page_index(self, job: JobState, plan: Dict[str, Any]) -> Optional[int]:
        page_index = None
        ref_num = plan.get("reference_page_number")
        try:
            ref_num = int(ref_num) if ref_num is not None else None
        except Exception:
            ref_num = None

        if ref_num is not None:
            if ref_num == 0 and 0 in job.pages:
                page_index = 0
            elif (ref_num - 1) in job.pages:
                page_index = ref_num - 1
            elif ref_num in job.pages:
                page_index = ref_num
        return page_index

    def _revision_query_matches(self, *, query: str, from_side: str, region: RegionState) -> bool:
        q = " ".join(str(query or "").strip().lower().split())
        if not q:
            return False

        src = " ".join(str(region.ocr_text or "").strip().lower().split())
        tgt = " ".join(str(region.translated_text or "").strip().lower().split())
        haystacks = []
        if from_side == "src":
            haystacks = [src]
        elif from_side == "tgt":
            haystacks = [tgt]
        else:
            haystacks = [src, tgt]

        for text in haystacks:
            if not text:
                continue
            if q == text or q in text or text in q:
                return True
            if SequenceMatcher(None, q, text).ratio() >= 0.92:
                return True
        return False

    def _select_revision_seed_pages(self, job: JobState, action: AgentAction, plan: Dict[str, Any], revision_mode: str) -> List[int]:
        if revision_mode == "auto_rag_revision":
            explicit = plan.get("target_page_indices")
            if isinstance(explicit, list):
                selected = []
                for item in explicit:
                    try:
                        pi = int(item)
                    except Exception:
                        continue
                    if pi in job.pages:
                        selected.append(pi)
                if selected:
                    return sorted(dict.fromkeys(selected))
            return [action.page_index]

        strategy = plan.get("strategy_suggestions") if isinstance(plan.get("strategy_suggestions"), dict) else {}
        scope = str((strategy or {}).get("revision_scope") or plan.get("scope") or "page_only").strip().lower()
        ref_page_index = self._resolve_plan_page_index(job, plan)
        if ref_page_index is None:
            ref_page_index = action.page_index

        if scope != "global":
            return [ref_page_index]

        query = str(plan.get("from_text") or "").strip()
        from_side = str(plan.get("from_side") or "auto").strip().lower()
        selected: List[int] = []
        if ref_page_index in job.pages:
            selected.append(ref_page_index)

        for page_index in sorted(job.pages.keys()):
            if page_index == ref_page_index:
                continue
            if not query:
                continue
            page = job.pages[page_index]
            if any(self._revision_query_matches(query=query, from_side=from_side, region=region) for region in page.regions.values()):
                selected.append(page_index)

        env_limit = max(1, int(os.getenv("MANGA_AGENT_REVISION_PRESELECT_PAGES", "24")))
        suggested_limit = (strategy or {}).get("revision_page_limit")
        try:
            max_pages = min(env_limit, max(1, int(suggested_limit))) if suggested_limit is not None else env_limit
        except Exception:
            max_pages = env_limit
        if len(selected) > max_pages:
            selected = selected[:max_pages]

        return selected or [ref_page_index]

    def _select_revision_target_pages(self, job: JobState, action: AgentAction, plan: Dict[str, Any], revision_mode: str) -> List[int]:
        seed_pages = self._select_revision_seed_pages(job, action, plan, revision_mode)
        strategy = plan.get("strategy_suggestions") if isinstance(plan.get("strategy_suggestions"), dict) else {}
        scope = str((strategy or {}).get("revision_scope") or plan.get("scope") or "page_only").strip().lower()

        if revision_mode != "auto_rag_revision" and scope != "global":
            ref_page_index = seed_pages[0] if seed_pages else action.page_index
            ref_page = job.pages.get(ref_page_index)
            batch_range = getattr(ref_page, "vlm_batch_range", None) if ref_page is not None else None
            if isinstance(batch_range, dict) and isinstance(batch_range.get("page_indices"), list):
                selected = []
                for item in batch_range.get("page_indices") or []:
                    try:
                        pi = int(item)
                    except Exception:
                        continue
                    if pi in job.pages:
                        selected.append(pi)
                if selected:
                    return sorted(dict.fromkeys(selected))
            return seed_pages

        if revision_mode == "auto_rag_revision":
            return seed_pages

        env_limit = max(1, int(os.getenv("MANGA_AGENT_REVISION_PRESELECT_PAGES", "24")))
        suggested_limit = (strategy or {}).get("revision_page_limit")
        try:
            max_pages = min(env_limit, max(1, int(suggested_limit))) if suggested_limit is not None else env_limit
        except Exception:
            max_pages = env_limit

        expanded: List[int] = []
        for page_index in seed_pages:
            page_state = job.pages.get(page_index)
            batch_range = getattr(page_state, "vlm_batch_range", None) if page_state is not None else None
            if isinstance(batch_range, dict) and isinstance(batch_range.get("page_indices"), list):
                for item in batch_range.get("page_indices") or []:
                    try:
                        pi = int(item)
                    except Exception:
                        continue
                    if pi in job.pages:
                        expanded.append(pi)
            elif page_index in job.pages:
                expanded.append(page_index)

        expanded = sorted(dict.fromkeys(expanded))
        if len(expanded) > max_pages:
            expanded = expanded[:max_pages]

        return expanded or seed_pages

    def _select_revision_apply_pages(self, job: JobState, action: AgentAction, plan: Dict[str, Any], revision_mode: str) -> List[int]:
        if revision_mode == "auto_rag_revision":
            return self._select_revision_target_pages(job, action, plan, revision_mode)

        strategy = plan.get("strategy_suggestions") if isinstance(plan.get("strategy_suggestions"), dict) else {}
        scope = str((strategy or {}).get("revision_scope") or plan.get("scope") or "page_only").strip().lower()
        if scope != "global":
            ref_page_index = self._resolve_plan_page_index(job, plan)
            if ref_page_index is None:
                ref_page_index = action.page_index
            return [ref_page_index]
        return self._select_revision_seed_pages(job, action, plan, revision_mode)

    def _target_pages_for_plan(self, job: JobState, plan: Dict[str, Any]) -> List[int]:
        page_index = self._resolve_plan_page_index(job, plan)
        scope = str(plan.get("scope") or "page_only").strip().lower()
        if scope == "global":
            return sorted(job.pages.keys())
        if page_index is not None and page_index in job.pages:
            return [page_index]
        return [sorted(job.pages.keys())[0]] if job.pages else []

    def _merge_page_skill_params(self, page: PageState, skill_name: str, params: Dict[str, Any]) -> None:
        merged = dict(page.skill_param_suggestions or {})
        prev = merged.get(skill_name) or {}
        merged[skill_name] = {**prev, **params}
        page.skill_param_suggestions = merged

    def _apply_page_profile_presets(self, page: PageState, profile: str) -> None:
        profile_norm = str(profile or "").strip().lower()
        if profile_norm == "art_integrated":
            self._merge_page_skill_params(page, "lama_eraser", {"profile": "aggressive", "mask_dilate_px": 4})
            self._merge_page_skill_params(page, "lettering", {"render_mode": "art_embed", "padding_ratio": 0.06, "stroke_width": 3})
        elif profile_norm == "boxed_dialogue":
            self._merge_page_skill_params(page, "lama_eraser", {"profile": "balanced", "mask_dilate_px": 2})
            self._merge_page_skill_params(page, "lettering", {"render_mode": "bubble_fit", "padding_ratio": 0.08, "stroke_width": 2})

    def _apply_skill_suggestions_from_plan(self, job: JobState, plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        clean = sanitize_skill_param_suggestions(plan.get("skill_param_suggestions"))
        if not clean:
            return {}

        target_pages = self._target_pages_for_plan(job, plan)

        for pi in target_pages:
            page = job.pages[pi]
            for skill_name, params in clean.items():
                self._merge_page_skill_params(page, skill_name, params)
        return clean

    def _apply_strategy_suggestions_from_plan(self, job: JobState, plan: Dict[str, Any]) -> Dict[str, Any]:
        clean = sanitize_strategy_suggestions(plan.get("strategy_suggestions"))
        if not clean:
            return {}

        if clean.get("revision_scope") in {"page_only", "global"}:
            plan["scope"] = clean["revision_scope"]

        for pi in self._target_pages_for_plan(job, plan):
            page = job.pages[pi]
            merged = dict(page.strategy_suggestions or {})
            merged.update(clean)
            page.strategy_suggestions = merged
            profile = clean.get("page_profile")
            if isinstance(profile, str) and profile:
                self._apply_page_profile_presets(page, profile)
        return clean

    def _apply_page_strategy_suggestions(
        self,
        page: PageState,
        *,
        strategy_suggestions: Dict[str, Any],
        skill_param_suggestions: Dict[str, Dict[str, Any]],
        summary: Optional[str],
        confidence: Optional[float],
        stage_label: str,
    ) -> None:
        had_actionable_update = bool(strategy_suggestions or skill_param_suggestions)
        if strategy_suggestions:
            merged_strategy = dict(page.strategy_suggestions or {})
            merged_strategy.update(strategy_suggestions)
            page.strategy_suggestions = merged_strategy
            profile = strategy_suggestions.get("page_profile")
            if isinstance(profile, str) and profile:
                self._apply_page_profile_presets(page, profile)

        for skill_name, params in (skill_param_suggestions or {}).items():
            if isinstance(params, dict) and params:
                self._merge_page_skill_params(page, skill_name, params)

        if summary is not None:
            clean_summary = summary.strip()
            if clean_summary:
                page.page_brain_summary = clean_summary
                if had_actionable_update and clean_summary.lower() not in {"default page strategy", "no extra page strategy"}:
                    page.page_brain_last_actionable_summary = clean_summary
        page.page_brain_confidence = confidence
        page.page_brain_stage_last = stage_label

    def _page_range_from_indices(self, page_indices: List[int]) -> Dict[str, Any]:
        ordered = sorted(int(x) for x in page_indices)
        ranges = []
        start = ordered[0]
        prev = ordered[0]
        for value in ordered[1:]:
            if value == prev + 1:
                prev = value
                continue
            ranges.append((start, prev))
            start = value
            prev = value
        ranges.append((start, prev))
        label = ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)
        return {
            "start_page_index": ordered[0],
            "end_page_index": ordered[-1],
            "page_count": len(ordered),
            "page_indices": ordered,
            "label": label,
        }

    def _apply_job_strategy_suggestions(
        self,
        job: JobState,
        *,
        strategy_suggestions: Dict[str, Any],
        summary: Optional[str],
        confidence: Optional[float],
        stage_label: str,
    ) -> None:
        if strategy_suggestions:
            merged = dict(job.job_strategy_suggestions or {})
            merged.update(strategy_suggestions)
            job.job_strategy_suggestions = merged
            if "vlm_batch_size" in strategy_suggestions:
                try:
                    job.vlm_batch_size = max(1, min(10, int(strategy_suggestions["vlm_batch_size"])))
                except Exception:
                    pass

        if summary is not None and str(summary).strip():
            job.job_brain_summary = str(summary).strip()
        job.job_brain_confidence = confidence
        job.job_brain_stage_last = stage_label

    def _build_job_summary_for_brain(self, job: JobState) -> Dict[str, Any]:
        pages_sorted = [job.pages[pi] for pi in sorted(job.pages.keys())]
        stage_counts: Dict[str, int] = {}
        for page in pages_sorted:
            stage_counts[page.stage.value] = stage_counts.get(page.stage.value, 0) + 1

        active_stages = {
            PageStage.DETECTED,
            PageStage.OCR_DONE,
            PageStage.TRANSLATED,
            PageStage.REVISING,
            PageStage.ERASING,
            PageStage.ERASED,
            PageStage.TYPESETTING,
            PageStage.TRANSLATING,
            PageStage.DETECTING,
            PageStage.OCRING,
        }
        active_page_indices = [page.page_index for page in pages_sorted if page.stage in active_stages]
        next_not_started_page_indices = [page.page_index for page in pages_sorted if page.stage == PageStage.NOT_STARTED]

        return {
            "job_id": job.job_id,
            "page_count": len(job.pages),
            "status": job.status.value,
            "preferred_vlm_batch_size": max(1, min(10, int(getattr(job, "vlm_batch_size", 10) or 10))),
            "stage_counts": stage_counts,
            "active_page_count": len(active_page_indices),
            "active_page_indices": active_page_indices[:20],
            "next_not_started_page_indices": next_not_started_page_indices[:10],
            "completed_vlm_batches": len(getattr(job, "vlm_batches", []) or []),
            "completed_revision_batches": len(getattr(job, "revision_batches", []) or []),
            "current_job_strategy": dict(job.job_strategy_suggestions or {}),
        }

    def _refresh_job_brain_strategy(self, job: JobState, *, trigger: str) -> None:
        if not hasattr(self.brain, "suggest_job_strategy"):
            return
        try:
            suggested = self.brain.suggest_job_strategy(
                job_summary=self._build_job_summary_for_brain(job),
                pipeline_mode=str(job.user_preferences.get("pipeline_mode") or "classic").lower(),
                target_language=job.target_language,
                skill_context=build_skill_context_for_brain(),
            )
        except Exception as e:
            job.logs.append(f"Job strategy brain failed: {e}")
            return

        clean_strategy = sanitize_job_strategy_suggestions((suggested or {}).get("strategy_suggestions"))
        summary = suggested.get("summary") if isinstance(suggested, dict) else None
        if not isinstance(summary, str):
            summary = None
        confidence = None
        if isinstance(suggested, dict):
            try:
                confidence = float(suggested.get("confidence")) if suggested.get("confidence") is not None else None
            except Exception:
                confidence = None

        self._apply_job_strategy_suggestions(
            job,
            strategy_suggestions=clean_strategy,
            summary=summary,
            confidence=confidence,
            stage_label=trigger,
        )

    def _build_page_summary_for_brain(self, page: PageState) -> Dict[str, Any]:
        regions = list(page.regions.values())
        total_regions = len(regions)
        boxed_regions = sum(1 for r in regions if r.has_text_box is True)
        embedded_regions = sum(1 for r in regions if r.has_text_box is False)
        empty_tgt_regions = sum(1 for r in regions if not str(r.translated_text or "").strip())
        overflow_regions = sum(1 for r in regions if bool(r.render_overflow))
        bad_render_regions = sum(1 for r in regions if (not r.render_ok) or bool(r.render_overflow))
        embedded_ratio = (embedded_regions / float(total_regions)) if total_regions else 0.0

        return {
            "page_index": page.page_index,
            "page_type": page.page_type,
            "stage": page.stage.value,
            "regions_total": total_regions,
            "boxed_regions": boxed_regions,
            "embedded_regions": embedded_regions,
            "embedded_ratio": embedded_ratio,
            "empty_tgt_regions": empty_tgt_regions,
            "overflow_regions": overflow_regions,
            "bad_render_regions": bad_render_regions,
            "vlm_quality_ok": page.vlm_quality_ok,
            "vlm_quality_reason": page.vlm_quality_reason,
            "vlm_judge_decision": page.vlm_judge_decision,
            "vlm_judge_reason_code": page.vlm_judge_reason_code,
            "retry_count_vlm": page.retry_count_vlm,
            "auto_revision_done": page.auto_revision_done,
            "auto_revision_changed": page.auto_revision_changed,
            "erase_ok": page.erase_ok,
            "erase_backend": page.erase_backend,
            "erase_metrics": page.erase_metrics or {},
            "erase_mask_quality": ((page.erase_metrics or {}).get("mask_quality")),
            "erase_mask_quality_reason": ((page.erase_metrics or {}).get("mask_quality_reason")),
            "erase_mask_strategy": ((page.erase_metrics or {}).get("mask_strategy")),
            "erase_mask_coverage": ((page.erase_metrics or {}).get("mask_coverage")),
            "erase_embedded_regions": ((page.erase_metrics or {}).get("embedded_regions")),
            "erase_mask_retry_hint": ((page.erase_metrics or {}).get("mask_retry_hint") or {}),
            "lettering_ok": page.lettering_ok,
            "typeset_metrics": page.typeset_metrics or {},
            "retry_count_erase": page.retry_count_erase,
            "has_erased_image": bool(page.erased_image_ref),
            "has_rendered_image": bool(page.rendered_image_ref),
            "current_strategy_suggestions": dict(page.strategy_suggestions or {}),
            "current_skill_param_suggestions": dict(page.skill_param_suggestions or {}),
        }

    def _refresh_page_brain_strategy(self, job: JobState, page: PageState, *, trigger: str) -> None:
        if not hasattr(self.brain, "suggest_page_strategy"):
            return
        try:
            suggested = self.brain.suggest_page_strategy(
                page_summary=self._build_page_summary_for_brain(page),
                pipeline_mode=str(job.user_preferences.get("pipeline_mode") or "classic").lower(),
                target_language=job.target_language,
                skill_context=build_skill_context_for_brain(),
            )
        except Exception as e:
            job.logs.append(f"Page strategy brain failed on page {page.page_index}: {e}")
            return

        clean_strategy = sanitize_strategy_suggestions((suggested or {}).get("strategy_suggestions"))
        clean_skills = sanitize_skill_param_suggestions((suggested or {}).get("skill_param_suggestions"))
        summary = suggested.get("summary") if isinstance(suggested, dict) else None
        if not isinstance(summary, str):
            summary = None
        confidence = None
        if isinstance(suggested, dict):
            try:
                confidence = float(suggested.get("confidence")) if suggested.get("confidence") is not None else None
            except Exception:
                confidence = None

        self._apply_page_strategy_suggestions(
            page,
            strategy_suggestions=clean_strategy,
            skill_param_suggestions=clean_skills,
            summary=summary,
            confidence=confidence,
            stage_label=trigger,
        )

    # -----------------------------------------------------
    # 对外接口 1：创建任务
    # -----------------------------------------------------
    def create_job(
        self,
        user_id: str,
        source_file_name: str,
        target_language: str,
        pages: List[Tuple[int, str]],
    ) -> str:
        """
        pages: [(page_index, image_ref), ...]
        image_ref 可以是：
        - 图片路径
        - 对象存储 key
        - 其它资源引用
        """
        job_id = str(uuid.uuid4())

        job = JobState(
            job_id=job_id,
            user_id=user_id,
            source_file_name=source_file_name,
            target_language=target_language,
            status=JobStatus.CREATED,
            vlm_batch_size=max(1, min(10, int(os.getenv("MANGA_AGENT_VLM_BATCH_SIZE", "10")))),
        )

        # Initialize glossary bucket for in-job corrections (RAG / translation memory).
        job.glossary = {
            "version": "0.1.0",
            "entries": [],
            "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }

        for page_index, image_ref in pages:
            job.pages[page_index] = PageState(
                page_index=page_index,
                image_ref=image_ref,
                stage=PageStage.NOT_STARTED,
            )

        job.logs.append(f"Job created with {len(pages)} pages.")
        self.memory.initialize_job(job)
        self.store.create_job(job)
        return job_id

    # -----------------------------------------------------
    # 对外接口 2：处理用户后续消息/反馈
    # -----------------------------------------------------
    def handle_user_message(self, job_id: str, user_message: str) -> None:
        """
        例如：
        - “翻译成中文”
        - “第7页没做好，重跑第7页”
        - “拟声词更夸张”
        """
        job = self.store.get_job(job_id)

        # 更新轻量偏好
        self.memory.update_preferences(job, user_message)
        self.memory.initialize_job(job)

        # 如果用户指定某页要重跑，则标记
        wants_rerun = any(k in user_message for k in ("重跑", "重试", "重新", "rerun", "retry"))
        if wants_rerun:
            page_num = self._extract_page_index_from_message(user_message)
            if page_num is not None and page_num in job.pages:
                job.pages[page_num].needs_manual_rerun = True
                job.logs.append(f"User requested rerun for page {page_num}.")

        # Route A: use the decision brain to turn the message into a structured plan.
        try:
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            brain_context = self.memory.build_brain_context(job, user_message=user_message)
            plan = self.brain.plan_user_message(
                user_message=user_message,
                page_count=len(job.pages),
                pipeline_mode=pipeline_mode,
                target_language=job.target_language,
                brain_context=brain_context,
                skill_context=build_skill_context_for_brain(),
            )
            if isinstance(plan, dict) and plan:
                plan.setdefault("user_message", user_message)
                clean_strategy = self._apply_strategy_suggestions_from_plan(job, plan)
                clean_suggestions = self._apply_skill_suggestions_from_plan(job, plan)
                plan["skill_param_suggestions"] = clean_suggestions
                plan["strategy_suggestions"] = clean_strategy
                intent = str(plan.get("intent") or "unknown").strip().lower()
                if intent in {"correction", "rerun_page"} or bool(plan.get("needs_clarification")):
                    job.pending_user_plan = plan
                else:
                    job.pending_user_plan = None
                job.logs.append(f"Brain plan: intent={plan.get('intent')} scope={plan.get('scope')} ref={plan.get('reference_page_number')} conf={plan.get('confidence')}")
                self.memory.record_brain_event(job, user_message=user_message, plan=plan)
        except Exception as e:
            job.logs.append(f"Brain planning skipped/failed: {e}")
            self.memory.record_brain_event(job, user_message=user_message, note=f"planning_failed: {e}")

        job.logs.append(f"Handled user message: {user_message}")

        # If the job was completed/waiting, allow resuming for corrections.
        if job.status in {JobStatus.WAITING_USER, JobStatus.COMPLETED}:
            job.status = JobStatus.RUNNING
            job.waiting_reason = None

        self.store.save_job(job)

    def _extract_page_index_from_message(self, msg: str) -> Optional[int]:
        """
        极简示例：
        从用户消息中提取第一个整数，作为 page_index。
        更严谨的版本后面可以改成正则或 LLM 解析。
        """
        # Common user phrasing is human-facing 1-based: "第7页".
        # Prefer mapping that to 0-based page_index to avoid off-by-one.
        m = re.search(r"第\s*(\d+)\s*页", msg)
        if m:
            try:
                n = int(m.group(1))
                return 0 if n <= 0 else (n - 1)
            except Exception:
                pass

        digits = []
        reading = False

        for ch in msg:
            if ch.isdigit():
                digits.append(ch)
                reading = True
            elif reading:
                break

        if digits:
            n = int("".join(digits))
            # For bare numbers like "重跑 1" / "retry page 1", interpret as human-facing 1-based.
            # Users can explicitly request 0-based by mentioning "page_index" or "0-based".
            lower = (msg or "").lower()
            if ("page_index" in lower) or ("0-based" in lower) or ("0based" in lower):
                return n
            if n <= 0:
                return 0
            return n - 1
        return None

    # -----------------------------------------------------
    # 对外接口 3：推进任务，直到需要用户或任务完成
    # -----------------------------------------------------
    def run_until_waiting_or_done(self, job_id: str, max_steps: Optional[int] = None) -> JobState:
        job = self.store.get_job(job_id)

        if job.status == JobStatus.CREATED:
            job.status = JobStatus.RUNNING
            self.store.save_job(job)

        # Auto-scale max_steps to avoid truncating medium-sized books.
        # Classic pipeline needs multiple actions per page; VLM also needs several actions including NO_OP transitions.
        if max_steps is None or int(max_steps) <= 0:
            page_count = len(job.pages)
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            steps_per_page = 10 if pipeline_mode == "classic" else 6
            max_steps = max(200, page_count * steps_per_page + 50)
            job.logs.append(f"Auto max_steps={max_steps} (pages={page_count}, mode={pipeline_mode}).")

        for _ in range(int(max_steps)):
            job = self.store.get_job(job_id)
            self._refresh_job_brain_strategy(job, trigger="before_decision")
            action = self.decision_engine.decide_next_action(job)
            self._log(job, f"Decision: {action.action_type.value}, reason={action.reason}")

            if action.action_type == ActionType.NO_OP:
                if job.pages and all(page.stage == PageStage.DONE for page in job.pages.values()):
                    job.status = JobStatus.COMPLETED
                    self._log(job, "All pages completed.")
                    self.store.save_job(job)
                    return job

                self.store.save_job(job)
                continue

            if action.action_type == ActionType.FINISH_JOB:
                job.status = JobStatus.COMPLETED
                self._log(job, "Job finished.")
                self.store.save_job(job)
                return job

            if action.action_type == ActionType.FAIL_JOB:
                job.status = JobStatus.FAILED
                job.error_message = action.reason
                self._log(job, f"Job failed: {action.reason}")
                self.store.save_job(job)
                return job

            if action.action_type == ActionType.ASK_USER:
                job.status = JobStatus.WAITING_USER
                job.waiting_reason = action.reason
                self._log(job, f"Waiting for user: {action.reason}")
                self.tools.ask_user(action.reason)
                self.store.save_job(job)
                return job

            self._execute_action(job, action)
            self.store.save_job(job)

        job.status = JobStatus.WAITING_USER
        job.waiting_reason = f"Max steps reached (max_steps={max_steps}); waiting for user or operator."
        self._log(job, "Max steps reached; stopping loop.")
        self.store.save_job(job)
        return job

    # -----------------------------------------------------
    # 内部：执行具体动作
    # -----------------------------------------------------
    def _execute_action(self, job: JobState, action: AgentAction) -> None:
        if action.action_type == ActionType.RUN_VLM_BATCH:
            page_indices = [int(x) for x in (action.payload.get("page_indices") or [])]
            target_language = action.payload.get("target_language", job.target_language)
            self._do_vlm_batch(job, page_indices=page_indices, target_language=target_language)
            return

        if action.page_index is None:
            raise ValueError("Page-level action must include page_index.")

        page = job.pages[action.page_index]
        job.status = JobStatus.RUNNING

        if action.action_type in [ActionType.RUN_DETECTION, ActionType.RETRY_DETECTION]:
            self._do_detection(job, page, retry=(action.action_type == ActionType.RETRY_DETECTION))
            return

        if action.action_type in [ActionType.RUN_VLM_PAGE, ActionType.RETRY_VLM_PAGE]:
            target_language = action.payload.get("target_language", job.target_language)
            self._do_vlm_page(
                job,
                page,
                target_language=target_language,
                retry=(action.action_type == ActionType.RETRY_VLM_PAGE),
            )
            return

        if action.action_type == ActionType.RUN_REVISION:
            self._do_revision(job, action)
            return

        if action.action_type in [ActionType.RUN_OCR, ActionType.RETRY_OCR]:
            self._do_ocr(job, page, retry=(action.action_type == ActionType.RETRY_OCR))
            return

        if action.action_type in [ActionType.RUN_TRANSLATION, ActionType.RETRY_TRANSLATION]:
            target_language = action.payload.get("target_language", job.target_language)
            self._do_translation(
                job,
                page,
                target_language=target_language,
                retry=(action.action_type == ActionType.RETRY_TRANSLATION),
            )
            return

        if action.action_type in [ActionType.RUN_ERASE, ActionType.RETRY_ERASE]:
            self._do_erase(
                job,
                page,
                retry=(action.action_type == ActionType.RETRY_ERASE),
                skill_params=action.payload.get("skill_params") if isinstance(action.payload, dict) else None,
            )
            return

        if action.action_type in [ActionType.RUN_TYPESET, ActionType.RETRY_TYPESET]:
            self._do_typeset(
                job,
                page,
                retry=(action.action_type == ActionType.RETRY_TYPESET),
                skill_params=action.payload.get("skill_params") if isinstance(action.payload, dict) else None,
            )
            return

        raise ValueError(f"Unsupported action type: {action.action_type}")

    def _do_revision(self, job: JobState, action: AgentAction) -> None:
        """Apply text-only translation revisions (no VLM rerun).

        The revision skill returns the same VLM-style schema (page->regions with src/tgt texts).
        We merge updated tgt_text back into RegionState, then re-run lettering for changed pages.
        """

        plan = action.payload or {}
        revision_mode = str(plan.get("revision_mode") or "user_feedback_revision").strip().lower()
        page = job.pages[action.page_index]

        target_page_indices = self._select_revision_target_pages(job, action, plan, revision_mode)
        apply_page_indices = self._select_revision_apply_pages(job, action, plan, revision_mode)
        stage_before_revision = {pi: job.pages[pi].stage for pi in target_page_indices if pi in job.pages}
        for pi in target_page_indices:
            if pi in job.pages:
                job.pages[pi].stage = PageStage.REVISING
        document_range = plan.get("document_range") if isinstance(plan.get("document_range"), dict) else None
        if not isinstance(document_range, dict):
            document_range = self._page_range_from_indices(target_page_indices)
        pages_payload: List[Dict[str, Any]] = []
        for page_index in target_page_indices:
            page_state = job.pages[page_index]
            regions_payload = []
            for region_id in sorted(page_state.regions.keys()):
                r = page_state.regions[region_id]
                regions_payload.append(
                    {
                        "region_id": r.region_id,
                        "bbox_xywh": r.bbox,
                        "has_text_box": r.has_text_box,
                        "color": r.color,
                        "src_text": r.ocr_text or "",
                        "tgt_text": r.translated_text or "",
                    }
                )
            pages_payload.append(
                {
                    "page_index": page_index,
                    "page_type": page_state.page_type,
                    "regions": regions_payload,
                }
            )

        reading_direction = str(job.user_preferences.get("reading_direction") or "ltr")
        retrieval_context = self.memory.build_revision_retrieval(
            job,
            pages=[job.pages[pi] for pi in target_page_indices if pi in job.pages],
            document_range=document_range,
        )
        user_request = str(plan.get("user_message") or plan.get("raw") or "")
        if revision_mode == "auto_rag_revision":
            user_request = user_request or "Apply one automatic RAG-guided consistency pass to this page."

        result = self.tools.run_vlm_translation_revision(
            pages=pages_payload,
            document_range=document_range,
            revision_mode=revision_mode,
            user_request=user_request,
            reference_page_number=plan.get("reference_page_number"),
            scope=str(plan.get("scope") or "page_only"),
            apply_page_indices=apply_page_indices,
            from_side=str(plan.get("from_side") or "auto"),
            from_text=plan.get("from_text"),
            to_text=plan.get("to_text"),
            reading_direction=reading_direction,
            target_language=job.target_language,
            glossary=job.glossary or {},
            retrieval_context=retrieval_context,
        )

        if not result.get("ok"):
            err = result.get("error") or {}
            if revision_mode == "auto_rag_revision":
                usage = (result.get("metrics") or {}).get("usage")
                for pi in target_page_indices:
                    if pi not in job.pages:
                        continue
                    target_page = job.pages[pi]
                    target_page.stage = PageStage.TRANSLATED
                    target_page.auto_revision_done = True
                    target_page.auto_revision_changed = False
                    target_page.revision_mode_last = revision_mode
                    target_page.revision_warnings = [f"{err.get('code') or 'revision_failed'}: {err.get('message') or 'unknown'}"]
                    target_page.revision_usage = usage if isinstance(usage, dict) else None
                    target_page.revision_batch_range = dict(document_range)
                job.logs.append(
                    f"Auto revision skipped on pages {target_page_indices}: {err.get('code')} {err.get('message')}"
                )
                return

            job.pending_user_plan = None
            for pi in target_page_indices:
                if pi in job.pages:
                    job.pages[pi].stage = PageStage.TRANSLATED
            job.logs.append(f"Revision failed: {err.get('code')} {err.get('message')}")
            job.status = JobStatus.WAITING_USER
            job.waiting_reason = f"Revision failed: {err.get('message') or err.get('code') or 'unknown'}"
            return

        data = result.get("data") or {}
        decision = str(data.get("decision") or "apply").strip().lower()
        if decision == "ask_user":
            if revision_mode == "auto_rag_revision":
                usage = (result.get("metrics") or {}).get("usage")
                for pi in target_page_indices:
                    if pi not in job.pages:
                        continue
                    target_page = job.pages[pi]
                    target_page.stage = PageStage.TRANSLATED
                    target_page.auto_revision_done = True
                    target_page.auto_revision_changed = False
                    target_page.revision_mode_last = revision_mode
                    target_page.revision_warnings = [str(data.get("ask_user") or "auto revision skipped")]
                    target_page.revision_usage = usage if isinstance(usage, dict) else None
                    target_page.revision_batch_range = dict(document_range)
                job.logs.append(f"Auto revision skipped on pages {target_page_indices}: {data.get('ask_user')}")
                return

            q = str(data.get("ask_user") or "Please clarify your correction request.").strip()
            for pi in target_page_indices:
                if pi in job.pages:
                    job.pages[pi].stage = PageStage.TRANSLATED
            job.pending_user_plan = plan
            job.pending_user_plan["needs_clarification"] = True
            job.pending_user_plan["clarifying_question"] = q
            return

        returned_document_range = data.get("document_range") if isinstance(data.get("document_range"), dict) else None
        if isinstance(returned_document_range, dict):
            document_range = returned_document_range
        allowed_apply_page_set = {int(x) for x in apply_page_indices}
        pages_out = [
            p for p in (data.get("pages") or [])
            if isinstance(p, dict) and str(p.get("page_index")).strip().isdigit() and int(p.get("page_index")) in allowed_apply_page_set
        ]
        changed_pages = set(
            int(x) for x in (data.get("changed_pages") or [])
            if str(x).strip().isdigit() and int(x) in allowed_apply_page_set
        )
        patches = [
            x for x in (data.get("patches") or [])
            if isinstance(x, dict) and str(x.get("page_index")).strip().isdigit() and int(x.get("page_index")) in allowed_apply_page_set
        ]
        glossary_updates = [x for x in (data.get("glossary_updates") or []) if isinstance(x, dict)]
        warnings = [str(x) for x in (data.get("warnings") or []) if str(x).strip()]
        usage = (result.get("metrics") or {}).get("usage")
        for pi in target_page_indices:
            if pi in job.pages:
                job.pages[pi].revision_batch_range = dict(document_range)

        # Merge tgt_text back into state.
        changed_regions = 0
        regions_before_merge = {
            pi: {rid: region for rid, region in page.regions.items()}
            for pi, page in job.pages.items()
        }
        for p in pages_out:
            try:
                pi = int(p.get("page_index"))
            except Exception:
                continue
            if pi not in job.pages:
                continue

            page = job.pages[pi]
            returned_page_type = str(p.get("page_type") or page.page_type or "content").strip().lower()
            if returned_page_type in {"cover", "info", "content"}:
                page.page_type = returned_page_type
            for r in p.get("regions", []) or []:
                try:
                    rid = int(r.get("region_id"))
                except Exception:
                    continue
                if rid not in page.regions:
                    continue
                new_tgt = r.get("tgt_text")
                if not isinstance(new_tgt, str):
                    continue

                region = page.regions[rid]
                region.color = str(r.get("color") or region.color or "black")
                if (region.translated_text or "") != new_tgt:
                    region.translated_text = new_tgt
                    region.stage = RegionStage.TRANSLATED
                    region.render_ok = False
                    region.render_overflow = False
                    changed_regions += 1
                    changed_pages.add(pi)

        persist_revision_memory = revision_mode != "auto_rag_revision"

        # Update glossary bucket only for explicit/user-confirmed revisions.
        if persist_revision_memory and isinstance(job.glossary, dict):
            entries = job.glossary.get("entries")
            if not isinstance(entries, list):
                entries = []
            if isinstance(glossary_updates, list):
                entries.extend([x for x in glossary_updates if isinstance(x, dict)])
            job.glossary["entries"] = entries
            job.glossary["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        if persist_revision_memory:
            self.memory.record_revision_updates(
                job,
                page_index_to_regions=regions_before_merge,
                patches=patches,
                glossary_updates=glossary_updates,
                glossary_type=(plan.get("glossary_type") if isinstance(plan, dict) else None),
            )

        job.revision_batches.append(
            {
                "revision_mode": revision_mode,
                "page_range": dict(document_range),
                "candidate_pages": len(target_page_indices),
                "changed_pages": sorted(changed_pages),
                "patch_count": len(patches),
                "warnings": warnings,
                "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        )

        # Re-run lettering on changed pages.
        for pi in changed_pages:
            if pi in job.pages:
                job.pages[pi].stage = PageStage.ERASED if job.pages[pi].erased_image_ref else PageStage.TRANSLATED
                job.pages[pi].lettering_ok = False
                job.pages[pi].translation_consistent = True
                job.pages[pi].rendered_image_ref = None

        if revision_mode == "auto_rag_revision":
            for pi in target_page_indices:
                if pi not in job.pages:
                    continue
                target_page = job.pages[pi]
                target_page.auto_revision_done = True
                target_page.auto_revision_changed = bool(pi in changed_pages)
                target_page.revision_mode_last = revision_mode
                target_page.revision_warnings = warnings
                target_page.revision_usage = usage if isinstance(usage, dict) else None
                if target_page.stage == PageStage.REVISING:
                    target_page.stage = PageStage.TRANSLATED
        else:
            for pi in target_page_indices:
                if pi in job.pages:
                    job.pages[pi].auto_revision_done = True
                    job.pages[pi].auto_revision_changed = bool(pi in changed_pages)
                    job.pages[pi].revision_mode_last = revision_mode
                    job.pages[pi].revision_usage = usage if isinstance(usage, dict) else None
                    if job.pages[pi].stage == PageStage.REVISING:
                        if pi in changed_pages:
                            job.pages[pi].stage = PageStage.TRANSLATED
                        else:
                            job.pages[pi].stage = stage_before_revision.get(pi, PageStage.TRANSLATED)
            job.pending_user_plan = None

        job.logs.append(
            f"Revision applied: mode={revision_mode} changed_pages={sorted(changed_pages)} changed_regions={changed_regions} warnings={warnings}"
        )

    def _do_detection(self, job: JobState, page: PageState, retry: bool) -> None:
        page.stage = PageStage.DETECTING
        if retry:
            page.retry_count_detection += 1

        result = self.tools.run_detection(page)
        if not result["ok"]:
            page.stage = PageStage.FAILED
            page.page_error_message = "Detection failed."
            self._log(job, f"Page {page.page_index}: detection failed.")
            return

        page.regions.clear()
        for item in result["regions"]:
            page.regions[item["region_id"]] = RegionState(
                region_id=item["region_id"],
                bbox=item["bbox"],
                region_type=item.get("region_type", "text"),
                has_text_box=item.get("has_text_box"),
                stage=RegionStage.DETECTED,
            )

        page.detection_ok = True
        page.detection_count = len(page.regions)
        page.bbox_error = result.get("bbox_error", False)
        page.stage = PageStage.DETECTED
        page.needs_manual_rerun = False

        self._log(job, f"Page {page.page_index}: detection done, regions={page.detection_count}")

    def _prepare_page_for_vlm_attempt(self, page: PageState, *, retry: bool) -> None:
        page.stage = PageStage.TRANSLATING

        # Reset QA fields per attempt (avoid carrying judge results across retries).
        page.vlm_quality_ok = True
        page.vlm_quality_reason = None
        page.vlm_warnings = []
        page.vlm_judge_mode = (os.getenv("MANGA_AGENT_VLM_JUDGE_MODE", "auto") or "auto").lower().strip()
        page.vlm_judge_decision = None
        page.vlm_judge_reason_code = None
        page.vlm_judge_explanation = None
        page.vlm_judge_confidence = None
        page.vlm_judge_warnings = []
        page.vlm_judge_usage = None

        if retry:
            page.retry_count_vlm += 1

    def _apply_vlm_success_to_page(
        self,
        job: JobState,
        page: PageState,
        *,
        result: Dict[str, Any],
        target_language: str,
        reading_direction: str,
        source_language: str,
        batch_range: Optional[Dict[str, Any]] = None,
        batch_id: Optional[str] = None,
    ) -> None:
        page.regions.clear()
        for item in result.get("regions", []) or []:
            region_id = int(item["region_id"])
            page.regions[region_id] = RegionState(
                region_id=region_id,
                bbox=item.get("bbox"),
                region_type=item.get("region_type", "text"),
                has_text_box=item.get("has_text_box"),
                color=str(item.get("color") or "black"),
                ocr_text=item.get("src_text"),
                translated_text=item.get("tgt_text"),
                stage=RegionStage.TRANSLATED,
            )

        page_type = str(result.get("page_type") or "content").strip().lower()
        if page_type not in {"cover", "info", "content"}:
            page_type = "content"
        page.page_type = page_type
        page.detection_ok = True
        page.detection_count = len(page.regions)
        page.bbox_error = bool(result.get("bbox_error", False))
        page.avg_ocr_confidence = None
        page.translation_consistent = True
        page.auto_revision_done = False
        page.auto_revision_changed = False
        page.revision_mode_last = None
        page.revision_warnings = []
        page.revision_usage = None
        page.erase_ok = False
        page.erase_backend = None
        page.erase_metrics = None
        page.erased_image_ref = None
        page.rendered_image_ref = None
        page.lettering_ok = False
        page.typeset_metrics = None
        page.stage = PageStage.TRANSLATED
        page.needs_manual_rerun = False
        page.vlm_batch_range = dict(batch_range) if isinstance(batch_range, dict) else None
        page.vlm_batch_id = batch_id

        vm = result.get("vlm_metrics") or {}
        page.vlm_orig_size = list(vm.get("orig") or []) or None
        page.vlm_sent_size = list(vm.get("sent") or []) or None
        page.vlm_downscaled = vm.get("downscaled") if vm.get("downscaled") is not None else None
        page.vlm_usage = vm.get("usage") if isinstance(vm.get("usage"), dict) else None

        if page.page_type in {"cover", "info"}:
            page.vlm_quality_ok = True
            page.vlm_quality_reason = None
            page.vlm_warnings = [f"Skipped OCR/translation post-processing for page_type={page.page_type}."]
            page.vlm_judge_decision = "proceed"
            page.vlm_judge_reason_code = "non_content_page"
            page.vlm_judge_explanation = f"Page classified as {page.page_type}; downstream text pipeline skipped."
            page.vlm_judge_confidence = 1.0
            page.vlm_judge_warnings = []
            page.vlm_judge_usage = None
            page.rendered_image_ref = page.image_ref
            page.lettering_ok = True
            page.stage = PageStage.DONE
            self._log(
                job,
                f"Page {page.page_index}: classified as {page.page_type}; skipped revision/erase/typeset.",
            )
            self._refresh_page_brain_strategy(job, page, trigger="after_vlm_page")
            return

        q_ok, q_reason, q_warn = self._evaluate_vlm_page_quality(job, page, target_language=target_language)
        page.vlm_quality_ok = q_ok
        page.vlm_quality_reason = q_reason
        page.vlm_warnings = q_warn

        if page.vlm_quality_ok:
            try:
                regions_sorted = sorted(page.regions.values(), key=lambda r: r.region_id)
                judge_regions = [
                    {"region_id": r.region_id, "src_text": r.ocr_text or "", "tgt_text": r.translated_text or ""}
                    for r in regions_sorted
                ]

                def _has_cjk(s: str) -> bool:
                    for ch in s:
                        code = ord(ch)
                        if 0x4E00 <= code <= 0x9FFF:
                            return True
                    return False

                n = len(judge_regions)
                empty_src = sum(1 for r in judge_regions if not (r.get("src_text") or "").strip())
                empty_tgt = sum(1 for r in judge_regions if not (r.get("tgt_text") or "").strip())
                tgt_nonempty = sum(1 for r in judge_regions if (r.get("tgt_text") or "").strip())
                tgt_cjk = sum(1 for r in judge_regions if _has_cjk((r.get("tgt_text") or "").strip()))

                judge_metrics = {
                    "regions": n,
                    "empty_src": empty_src,
                    "empty_tgt": empty_tgt,
                    "tgt_nonempty": tgt_nonempty,
                    "tgt_cjk": tgt_cjk,
                    "tgt_cjk_ratio": (tgt_cjk / float(tgt_nonempty)) if tgt_nonempty else None,
                    "vlm_downscaled": page.vlm_downscaled,
                    "vlm_orig_size": page.vlm_orig_size,
                    "vlm_sent_size": page.vlm_sent_size,
                }

                if n > 0:
                    jr = self.tools.run_vlm_judge_text_only(
                        page_index=page.page_index,
                        target_language=target_language,
                        reading_direction=reading_direction,
                        regions=judge_regions,
                        metrics=judge_metrics,
                        warnings=list(page.vlm_warnings or []),
                    )

                    if jr.get("ok"):
                        jd = jr.get("data") or {}
                        page.vlm_judge_decision = jd.get("decision")
                        page.vlm_judge_reason_code = jd.get("reason_code")
                        page.vlm_judge_explanation = jd.get("explanation")
                        page.vlm_judge_confidence = jd.get("confidence")
                        page.vlm_judge_warnings = list(jd.get("warnings") or [])

                        ju = (jr.get("metrics") or {}).get("usage")
                        page.vlm_judge_usage = ju if isinstance(ju, dict) else None

                        if page.vlm_judge_decision == "retry":
                            page.vlm_quality_ok = False
                            page.vlm_quality_reason = "judge_retry"
                            if page.vlm_judge_explanation:
                                page.vlm_warnings = list(page.vlm_warnings or []) + [f"Judge: {page.vlm_judge_explanation}"]
                        elif page.vlm_judge_decision == "ask_user":
                            page.vlm_quality_ok = False
                            page.vlm_quality_reason = "judge_ask_user"
                            if page.vlm_judge_explanation:
                                page.vlm_warnings = list(page.vlm_warnings or []) + [f"Judge: {page.vlm_judge_explanation}"]
                    else:
                        err = jr.get("error") or {}
                        page.vlm_judge_decision = "proceed"
                        page.vlm_judge_reason_code = str(err.get("code") or "judge_failed")
                        page.vlm_judge_explanation = str(err.get("message") or "judge failed")
                        page.vlm_judge_warnings = [page.vlm_judge_explanation] if page.vlm_judge_explanation else []
                        if page.vlm_judge_explanation:
                            page.vlm_warnings = list(page.vlm_warnings or []) + [f"Judge failed: {page.vlm_judge_explanation}"]
            except Exception as e:
                page.vlm_judge_decision = "proceed"
                page.vlm_judge_reason_code = "judge_exception"
                page.vlm_judge_explanation = str(e)
                page.vlm_judge_warnings = [str(e)]
                page.vlm_warnings = list(page.vlm_warnings or []) + [f"Judge exception: {e}"]

        usage = vm.get("usage") or {}
        usage_str = ""
        if isinstance(usage, dict) and any(usage.get(k) is not None for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            usage_str = f" usage={usage}"
        batch_note = f" batch={batch_range.get('label')}" if isinstance(batch_range, dict) else ""
        self._log(
            job,
            f"Page {page.page_index}: VLM core done, regions={page.detection_count}, orig={vm.get('orig')} sent={vm.get('sent')} downscaled={vm.get('downscaled')}{usage_str}{batch_note}",
        )

        if page.vlm_judge_decision in {"retry", "ask_user"}:
            self._log(
                job,
                f"Page {page.page_index}: VLM judge decided {page.vlm_judge_decision} reason_code={page.vlm_judge_reason_code} conf={page.vlm_judge_confidence}",
            )

        if not page.vlm_quality_ok:
            self._log(
                job,
                f"Page {page.page_index}: VLM quality not ok: {page.vlm_quality_reason} warnings={page.vlm_warnings}",
            )

        self._refresh_page_brain_strategy(job, page, trigger="after_vlm_page")

    def _do_vlm_page(self, job: JobState, page: PageState, target_language: str, retry: bool) -> None:
        # VLM core does: detect regions + OCR + translate.
        self._prepare_page_for_vlm_attempt(page, retry=retry)

        reading_direction = str(job.user_preferences.get("reading_direction") or "ltr")
        source_language = str(job.user_preferences.get("source_language") or "en")

        result = self.tools.run_vlm_page(
            page,
            reading_direction=reading_direction,
            source_language=source_language,
            target_language=target_language,
            force_fullres=retry,
            retrieval_context=self.memory.build_vlm_retrieval(job, page),
        )

        if not result.get("ok"):
            page.stage = PageStage.FAILED
            page.page_error_message = "VLM core failed."
            page.vlm_quality_ok = False
            err = result.get("error") or {}
            page.vlm_quality_reason = str(err.get("code") or "vlm_failed")
            page.vlm_warnings = [str(err.get("message") or "VLM core failed")]
            self._log(job, f"Page {page.page_index}: VLM core failed.")
            return

        self._apply_vlm_success_to_page(
            job,
            page,
            result=result,
            target_language=target_language,
            reading_direction=reading_direction,
            source_language=source_language,
        )

    def _do_vlm_batch(self, job: JobState, *, page_indices: List[int], target_language: str) -> None:
        if not page_indices:
            return

        ordered_indices = [pi for pi in sorted(dict.fromkeys(page_indices)) if pi in job.pages]
        if not ordered_indices:
            return

        pages = [job.pages[pi] for pi in ordered_indices]
        for page in pages:
            self._prepare_page_for_vlm_attempt(page, retry=False)

        reading_direction = str(job.user_preferences.get("reading_direction") or "ltr")
        source_language = str(job.user_preferences.get("source_language") or "en")
        retrieval_contexts = {page.page_index: self.memory.build_vlm_retrieval(job, page) for page in pages}

        result = self.tools.run_vlm_pages(
            pages,
            reading_direction=reading_direction,
            source_language=source_language,
            target_language=target_language,
            retrieval_contexts=retrieval_contexts,
        )

        batch_range = result.get("page_range") if isinstance(result.get("page_range"), dict) else self._page_range_from_indices(ordered_indices)
        batch_id = f"vlm:{batch_range['label']}:{int(time.time() * 1000)}"

        if not result.get("ok"):
            err = result.get("error") or {}
            for page in pages:
                page.stage = PageStage.FAILED
                page.page_error_message = "VLM batch failed."
                page.vlm_quality_ok = False
                page.vlm_quality_reason = str(err.get("code") or "vlm_batch_failed")
                page.vlm_warnings = [str(err.get("message") or "VLM batch failed")]
                page.vlm_batch_range = dict(batch_range)
                page.vlm_batch_id = batch_id
            self._log(job, f"VLM batch {batch_range.get('label')}: failed code={err.get('code')} msg={err.get('message')}")
            return

        pages_out = result.get("pages") or []
        pages_by_index = {}
        for item in pages_out:
            if not isinstance(item, dict):
                continue
            try:
                pages_by_index[int(item.get("page_index"))] = item
            except Exception:
                continue

        batch_metrics = result.get("batch_metrics") if isinstance(result.get("batch_metrics"), dict) else {}
        job.vlm_batches.append(
            {
                "batch_id": batch_id,
                "page_range": dict(batch_range),
                "page_indices": list(batch_range.get("page_indices") or ordered_indices),
                "metrics": batch_metrics,
                "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
        )

        for page in pages:
            page_result = pages_by_index.get(page.page_index)
            if not isinstance(page_result, dict):
                page.stage = PageStage.FAILED
                page.page_error_message = "VLM batch missing page result."
                page.vlm_quality_ok = False
                page.vlm_quality_reason = "vlm_batch_missing_page_result"
                page.vlm_warnings = ["VLM batch did not return this page."]
                page.vlm_batch_range = dict(batch_range)
                page.vlm_batch_id = batch_id
                self._log(job, f"Page {page.page_index}: missing from VLM batch result.")
                continue

            self._apply_vlm_success_to_page(
                job,
                page,
                result=page_result,
                target_language=target_language,
                reading_direction=reading_direction,
                source_language=source_language,
                batch_range=batch_range,
                batch_id=batch_id,
            )

    def _evaluate_vlm_page_quality(self, job: JobState, page: PageState, *, target_language: str) -> tuple[bool, Optional[str], list[str]]:
        """Heuristic quality gate for VLM results.

        Goals:
        - Catch obvious failures (empty texts, wrong language, 0 regions when detector suggests text exists)
        - Keep false positives low to avoid spending extra tokens
        """

        warnings: list[str] = []
        regions = list(page.regions.values())
        n = len(regions)

        if getattr(page, "page_type", "content") in {"cover", "info"}:
            return True, None, warnings

        def _bbox_iou(a: list[int], b: list[int]) -> float:
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

        def _bbox_quality_signals() -> tuple[list[str], Optional[str]]:
            size = page.vlm_orig_size or []
            if not (isinstance(size, list) and len(size) == 2 and all(isinstance(x, int) and x > 0 for x in size)):
                return [], None

            page_w, page_h = int(size[0]), int(size[1])
            page_area = float(page_w * page_h)
            boxed_infos: list[dict[str, Any]] = []
            suspicious_messages: list[str] = []
            hard_fail = False

            for region in regions:
                bbox = region.bbox
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                try:
                    x, y, w, h = [int(round(float(v))) for v in bbox]
                except Exception:
                    continue
                if w <= 0 or h <= 0:
                    continue
                area = float(w * h)
                area_ratio = area / page_area if page_area > 0 else 0.0
                src = str(region.ocr_text or "").strip()
                tgt = str(region.translated_text or "").strip()
                char_count = max(len(src), len(tgt))
                chars_per_10k = (char_count * 10000.0 / area) if area > 0 else 0.0
                boxed = region.has_text_box is not False

                if boxed:
                    boxed_infos.append(
                        {
                            "region_id": region.region_id,
                            "bbox": [x, y, w, h],
                            "area_ratio": area_ratio,
                            "char_count": char_count,
                            "chars_per_10k": chars_per_10k,
                        }
                    )

                    if area_ratio >= 0.05 and char_count <= 80 and chars_per_10k < 3.0:
                        suspicious_messages.append(
                            f"r{region.region_id} boxed bbox is too large for its text (area={area_ratio:.3f}, chars={char_count}, density={chars_per_10k:.2f}/10k)."
                        )
                        hard_fail = True
                    elif area_ratio >= 0.035 and char_count <= 48 and chars_per_10k < 2.2:
                        suspicious_messages.append(
                            f"r{region.region_id} boxed bbox looks loose (area={area_ratio:.3f}, chars={char_count}, density={chars_per_10k:.2f}/10k)."
                        )

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
                        suspicious_messages.append(
                            f"r{a['region_id']} and r{b['region_id']} have highly-overlapping oversized boxed bboxes (IoU={iou:.2f})."
                        )
                        hard_fail = True

            return suspicious_messages, ("suspicious_bbox_geometry" if hard_fail else None)

        def _has_cjk(s: str) -> bool:
            for ch in s:
                code = ord(ch)
                if 0x4E00 <= code <= 0x9FFF:
                    return True
            return False

        if n == 0:
            # If VLM returns 0 regions, validate with local detector when possible.
            # If detector sees text, treat VLM as suspicious.
            try:
                if page.image_ref:
                    from pathlib import Path

                    p = Path(page.image_ref)
                    if p.exists() and p.is_file():
                        det = self.tools.run_detection(page)
                        if det.get("ok") and (len(det.get("regions") or []) > 0):
                            return False, "vlm_zero_regions_but_detector_found_text", ["VLM returned 0 regions but detector found text regions."]
            except Exception:
                # If detector validation fails, do not fail the page purely for that.
                warnings.append("Detector validation skipped/failed for 0-region VLM output.")

            # Could be a full-art / no-text page.
            return True, None, warnings

        bbox_warnings, bbox_reason = _bbox_quality_signals()
        if bbox_warnings:
            warnings.extend(bbox_warnings)
        if bbox_reason is not None:
            return False, bbox_reason, warnings

        src_nonempty = 0
        tgt_nonempty = 0
        tgt_cjk = 0

        for r in regions:
            src = (r.ocr_text or "").strip()
            tgt = (r.translated_text or "").strip()
            if src:
                src_nonempty += 1
            if tgt:
                tgt_nonempty += 1
            if tgt and _has_cjk(tgt):
                tgt_cjk += 1

        # Extremely permissive thresholds to avoid expensive retries.
        if n >= 3 and (src_nonempty / n) < 0.34:
            return False, "too_many_empty_src_text", [f"Only {src_nonempty}/{n} regions have non-empty src_text."]

        if n >= 3 and (tgt_nonempty / n) < 0.34:
            return False, "too_many_empty_tgt_text", [f"Only {tgt_nonempty}/{n} regions have non-empty tgt_text."]

        if target_language == "zh" and tgt_nonempty >= 3:
            cjk_ratio = tgt_cjk / float(tgt_nonempty)
            if cjk_ratio < 0.34:
                return False, "target_language_mismatch", [f"Low CJK ratio in tgt_text: {tgt_cjk}/{tgt_nonempty}."]

        return True, None, warnings

    def _do_ocr(self, job: JobState, page: PageState, retry: bool) -> None:
        page.stage = PageStage.OCRING
        result = self.tools.run_ocr(page)

        if not result["ok"]:
            page.stage = PageStage.FAILED
            page.page_error_message = "OCR failed."
            self._log(job, f"Page {page.page_index}: OCR failed.")
            return

        confidences = []
        for region_id, ocr_data in result["results"].items():
            region = page.regions[region_id]
            if retry:
                region.retry_count_ocr += 1

            region.ocr_text = ocr_data["text"]
            region.ocr_confidence = ocr_data["confidence"]
            region.stage = RegionStage.OCR_DONE
            confidences.append(region.ocr_confidence)

        page.avg_ocr_confidence = sum(confidences) / len(confidences) if confidences else None
        page.stage = PageStage.OCR_DONE

        self._log(job, f"Page {page.page_index}: OCR done, avg_conf={page.avg_ocr_confidence:.2f}")

    def _do_translation(self, job: JobState, page: PageState, target_language: str, retry: bool) -> None:
        page.stage = PageStage.TRANSLATING
        retrieval = self.memory.build_translation_retrieval(job, page)
        result = self.tools.run_translation(
            page,
            target_language=target_language,
            page_context=str(retrieval.get("page_context") or ""),
            glossary_entries=[x for x in (retrieval.get("glossary_hits") or []) if isinstance(x, dict)]
            + [x for x in (retrieval.get("translation_memory_hits") or []) if isinstance(x, dict)],
        )

        if not result["ok"]:
            page.stage = PageStage.FAILED
            page.page_error_message = "Translation failed."
            self._log(job, f"Page {page.page_index}: translation failed.")
            return

        for region_id, trans_data in result["results"].items():
            region = page.regions[region_id]
            if retry:
                region.retry_count_translation += 1

            region.translated_text = trans_data["translated_text"]
            region.translation_confidence = trans_data["confidence"]
            region.stage = RegionStage.TRANSLATED

        page.translation_consistent = result.get("consistent", True)
        page.auto_revision_done = False
        page.auto_revision_changed = False
        page.revision_mode_last = None
        page.revision_warnings = []
        page.revision_usage = None
        page.erase_ok = False
        page.erase_backend = None
        page.erase_metrics = None
        page.erased_image_ref = None
        page.stage = PageStage.TRANSLATED
        page.rendered_image_ref = None
        page.lettering_ok = False
        page.typeset_metrics = None

        self._log(job, f"Page {page.page_index}: translation done, consistent={page.translation_consistent}")

    def _do_erase(self, job: JobState, page: PageState, retry: bool, skill_params: Optional[Dict[str, Any]] = None) -> None:
        page.stage = PageStage.ERASING
        result = self.tools.run_erase(page, retry=retry, skill_params=skill_params)

        if not result["ok"]:
            page.stage = PageStage.FAILED
            page.page_error_message = "Erase failed."
            page.erase_ok = False
            if retry:
                page.retry_count_erase += 1
            self._log(job, f"Page {page.page_index}: erase failed.")
            return

        if retry:
            page.retry_count_erase += 1

        page.erase_ok = True
        page.erase_backend = result.get("erase_backend")
        page.erase_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else None
        page.stage = PageStage.ERASED
        page.rendered_image_ref = None
        self._log(job, f"Page {page.page_index}: erase done, backend={page.erase_backend}, params={skill_params or {}}")
        self._refresh_page_brain_strategy(job, page, trigger="after_erase")

    def _do_typeset(self, job: JobState, page: PageState, retry: bool, skill_params: Optional[Dict[str, Any]] = None) -> None:
        page.stage = PageStage.TYPESETTING
        result = self.tools.run_typeset(page, skill_params=skill_params)

        if not result["ok"]:
            page.stage = PageStage.FAILED
            page.page_error_message = "Typesetting failed."
            self._log(job, f"Page {page.page_index}: typesetting failed.")
            return

        all_ok = True
        for region_id, render_data in result["results"].items():
            region = page.regions[region_id]
            if retry:
                region.retry_count_render += 1

            region.render_ok = render_data["render_ok"]
            region.render_overflow = render_data["overflow"]

            if region.render_ok and not region.render_overflow:
                region.stage = RegionStage.RENDERED
            else:
                all_ok = False

        page.lettering_ok = all_ok
        page.typeset_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else None
        page.stage = PageStage.TYPESETTING

        self._log(job, f"Page {page.page_index}: typesetting done, all_ok={all_ok}, params={skill_params or {}}")
        self._refresh_page_brain_strategy(job, page, trigger="after_typeset")

    # -----------------------------------------------------
    # 工具：日志
    # -----------------------------------------------------
    def _log(self, job: JobState, msg: str) -> None:
        job.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


if __name__ == "__main__":
    # 一个最小可跑 demo
    from .store import InMemoryStateStore
    from .decision_engine import DecisionEngine
    from .tools import Tools

    store = InMemoryStateStore()
    tools = Tools()
    decision_engine = DecisionEngine()
    agent = ComicAgent(store=store, tools=tools, decision_engine=decision_engine)

    job_id = agent.create_job(
        user_id="user_001",
        source_file_name="comic.zip",
        target_language="zh",
        pages=[
            (0, "page_0000.png"),
            (1, "page_0001.png"),
            (2, "page_0002.png"),
        ],
    )

    job = agent.run_until_waiting_or_done(job_id)
    print("=== First run ===")
    print("job_status:", job.status)
    print("waiting_reason:", job.waiting_reason)
    print("last_logs:")
    for line in job.logs[-12:]:
        print(line)

    if job.status == JobStatus.WAITING_USER:
        agent.handle_user_message(job_id, "第1页没做好，重跑第1页")
        job = store.get_job(job_id)
        job.status = JobStatus.RUNNING
        job.waiting_reason = None
        store.save_job(job)

        job = agent.run_until_waiting_or_done(job_id)
        print("\n=== After user feedback ===")
        print("job_status:", job.status)
        print("waiting_reason:", job.waiting_reason)
        print("last_logs:")
        for line in job.logs[-12:]:
            print(line)
