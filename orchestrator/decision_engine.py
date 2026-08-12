from __future__ import annotations

from .state import (
    JobState,
    PageState,
    PageStage,
    AgentAction,
    ActionType,
    JobStatus,
)


class DecisionEngine:
    """
    规则驱动决策引擎。

    原则：
    1. 优先处理用户要求重跑的页面
    2. 页面按 stage 向前推进
    3. 如果质量不过关，优先自动重试
    4. 超过重试上限，再等待用户介入
    """

    OCR_CONF_THRESHOLD = 0.70
    MAX_RETRY_DETECTION = 2
    MAX_RETRY_OCR = 2
    MAX_RETRY_TRANSLATION = 1
    MAX_RETRY_ERASE = 1
    MAX_RETRY_RENDER = 2
    MAX_RETRY_VLM = 1

    def _strategy_for_plan(self, plan: dict) -> dict:
        strategy = plan.get("strategy_suggestions")
        return strategy if isinstance(strategy, dict) else {}

    def _strategy_for_page(self, page: PageState) -> dict:
        strategy = getattr(page, "strategy_suggestions", {}) or {}
        return strategy if isinstance(strategy, dict) else {}

    def _strategy_for_job(self, job: JobState) -> dict:
        strategy = getattr(job, "job_strategy_suggestions", {}) or {}
        return strategy if isinstance(strategy, dict) else {}

    def _skill_payload(self, page: PageState, skill_name: str) -> dict:
        suggestions = getattr(page, "skill_param_suggestions", {}) or {}
        params = suggestions.get(skill_name) if isinstance(suggestions, dict) else None
        return {"skill_params": dict(params)} if isinstance(params, dict) and params else {}

    def _auto_revision_payload(self, page: PageState) -> dict:
        return {
            "revision_mode": "auto_rag_revision",
            "scope": "page_only",
            "reference_page_number": page.page_index + 1,
            "user_message": "Brain suggested a text-only revision pass before continuing.",
        }

    def _auto_revision_action_for_page(self, job: JobState, page: PageState) -> AgentAction | None:
        if getattr(page, "page_type", "content") in {"cover", "info"}:
            return None
        page_strategy = self._strategy_for_page(page)
        post_translation_action = str(page_strategy.get("preferred_post_translation_action") or "auto_revision_then_erase").strip().lower()
        recommended_page_action = str(page_strategy.get("recommended_page_action") or "").strip().lower()

        needs_revision = (post_translation_action != "erase_direct") or (recommended_page_action == "run_revision")
        if not needs_revision or getattr(page, "auto_revision_done", False):
            return None

        batch_range = getattr(page, "vlm_batch_range", None)
        if isinstance(batch_range, dict):
            batch_indices = batch_range.get("page_indices") or []
            candidates = []
            for item in batch_indices:
                try:
                    pi = int(item)
                except Exception:
                    continue
                target_page = job.pages.get(pi)
                if not target_page or target_page.stage != PageStage.TRANSLATED:
                    continue
                if getattr(target_page, "auto_revision_done", False):
                    continue
                target_strategy = self._strategy_for_page(target_page)
                target_post_action = str(target_strategy.get("preferred_post_translation_action") or "auto_revision_then_erase").strip().lower()
                target_recommended = str(target_strategy.get("recommended_page_action") or "").strip().lower()
                if target_post_action == "erase_direct" and target_recommended != "run_revision":
                    continue
                candidates.append(pi)
            if not candidates:
                return None
            candidates = sorted(dict.fromkeys(candidates))
            if page.page_index != candidates[0]:
                return None
            payload = self._auto_revision_payload(page)
            payload["scope"] = "window"
            payload["target_page_indices"] = candidates
            payload["document_range"] = {
                "start_page_index": candidates[0],
                "end_page_index": candidates[-1],
                "page_count": len(candidates),
                "page_indices": candidates,
                "label": f"{candidates[0]}-{candidates[-1]}",
                "source_vlm_batch_range": dict(batch_range),
            }
            payload["reference_page_number"] = candidates[0] + 1
            payload["user_message"] = f"Apply one automatic RAG-guided consistency pass to VLM batch pages {candidates[0]}-{candidates[-1]}."
            return AgentAction(
                action_type=ActionType.RUN_REVISION,
                page_index=page.page_index,
                payload=payload,
                reason=f"Translation window produced; run one batch-level RAG self-check for pages {candidates[0]}-{candidates[-1]} before erase.",
            )

        payload = self._auto_revision_payload(page)
        return AgentAction(
            action_type=ActionType.RUN_REVISION,
            page_index=page.page_index,
            payload=payload,
            reason="Translation produced; run post-translation RAG self-check before erase.",
        )

    def _guard_brain_job_action(self, job: JobState) -> AgentAction | None:
        strategy = self._strategy_for_job(job)
        recommended = str(strategy.get("recommended_job_action") or "").strip().lower()
        if not recommended or recommended == "continue":
            return None

        if recommended == "ask_user":
            reason = str(strategy.get("ask_user_reason") or "").strip() or "Job strategy requested user review."
            return AgentAction(ActionType.ASK_USER, reason=reason)

        if recommended == "run_vlm_batch":
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            if pipeline_mode != "vlm":
                return None

            active_stages = {
                PageStage.DETECTING,
                PageStage.DETECTED,
                PageStage.OCRING,
                PageStage.OCR_DONE,
                PageStage.TRANSLATING,
                PageStage.TRANSLATED,
                PageStage.REVISING,
                PageStage.ERASING,
                PageStage.ERASED,
                PageStage.TYPESETTING,
            }
            if any(page.stage in active_stages for page in job.pages.values()):
                return None

            page_indices = strategy.get("vlm_batch_page_indices") or []
            if not isinstance(page_indices, list):
                page_indices = []
            clean_indices = []
            for item in page_indices[:10]:
                try:
                    pi = int(item)
                except Exception:
                    continue
                if pi in job.pages and job.pages[pi].stage == PageStage.NOT_STARTED:
                    clean_indices.append(pi)
            if not clean_indices:
                return None

            return AgentAction(
                action_type=ActionType.RUN_VLM_BATCH,
                payload={"page_indices": clean_indices, "target_language": job.target_language},
                reason=f"Job brain scheduled a VLM batch window for pages {clean_indices[0]}-{clean_indices[-1]}.",
            )

        return None

    def _default_vlm_batch_action(self, job: JobState) -> AgentAction | None:
        pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
        if pipeline_mode != "vlm":
            return None

        active_stages = {
            PageStage.DETECTING,
            PageStage.DETECTED,
            PageStage.OCRING,
            PageStage.OCR_DONE,
            PageStage.TRANSLATING,
            PageStage.TRANSLATED,
            PageStage.REVISING,
            PageStage.ERASING,
            PageStage.ERASED,
            PageStage.TYPESETTING,
        }
        if any(page.stage in active_stages for page in job.pages.values()):
            return None

        next_pages = [page.page_index for page in sorted(job.pages.values(), key=lambda p: p.page_index) if page.stage == PageStage.NOT_STARTED]
        if not next_pages:
            return None

        batch_size = max(1, min(10, int(getattr(job, "vlm_batch_size", 10) or 10)))
        chosen = next_pages[:batch_size]
        return AgentAction(
            action_type=ActionType.RUN_VLM_BATCH,
            payload={"page_indices": chosen, "target_language": job.target_language},
            reason=f"Default job-level VLM batching scheduled pages {chosen[0]}-{chosen[-1]}.",
        )

    def _guard_brain_page_action(
        self,
        job: JobState,
        page: PageState,
        *,
        allowed_actions: set[str],
    ) -> AgentAction | None:
        strategy = self._strategy_for_page(page)
        recommended = str(strategy.get("recommended_page_action") or "").strip().lower()
        if not recommended or recommended == "continue" or recommended not in allowed_actions:
            return None

        ask_reason = str(strategy.get("ask_user_reason") or "").strip()

        if recommended == "ask_user":
            return AgentAction(
                action_type=ActionType.ASK_USER,
                page_index=page.page_index,
                reason=ask_reason or "Brain requested user review for this page.",
            )

        if recommended == "retry_typeset":
            can_retry = (
                bool(page.regions)
                and (page.erased_image_ref or page.rendered_image_ref or page.stage in {PageStage.ERASED, PageStage.TYPESETTING, PageStage.DONE})
                and all(region.retry_count_render < self.MAX_RETRY_RENDER for region in page.regions.values())
            )
            if can_retry:
                return AgentAction(
                    action_type=ActionType.RETRY_TYPESET,
                    page_index=page.page_index,
                    payload=self._skill_payload(page, "lettering"),
                    reason="Brain proactively suggested another typeset pass for this page.",
                )
            return None

        if recommended == "retry_erase":
            if page.regions and page.retry_count_erase < self.MAX_RETRY_ERASE:
                return AgentAction(
                    action_type=ActionType.RETRY_ERASE,
                    page_index=page.page_index,
                    payload=self._skill_payload(page, "lama_eraser"),
                    reason="Brain proactively suggested improving erase/mask quality first.",
                )
            return None

        if recommended == "retry_vlm_page":
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            if pipeline_mode == "vlm" and page.retry_count_vlm < self.MAX_RETRY_VLM:
                return AgentAction(
                    action_type=ActionType.RETRY_VLM_PAGE,
                    page_index=page.page_index,
                    payload={"target_language": job.target_language},
                    reason="Brain proactively suggested rerunning the VLM page step.",
                )
            return None

        if recommended == "retry_translation":
            if page.regions and all(region.retry_count_translation < self.MAX_RETRY_TRANSLATION for region in page.regions.values()):
                return AgentAction(
                    action_type=ActionType.RETRY_TRANSLATION,
                    page_index=page.page_index,
                    payload={"target_language": job.target_language},
                    reason="Brain proactively suggested rerunning translation.",
                )
            return None

        if recommended == "retry_ocr":
            if page.regions and all(region.retry_count_ocr < self.MAX_RETRY_OCR for region in page.regions.values()):
                return AgentAction(
                    action_type=ActionType.RETRY_OCR,
                    page_index=page.page_index,
                    reason="Brain proactively suggested rerunning OCR.",
                )
            return None

        if recommended == "retry_detection":
            if page.retry_count_detection < self.MAX_RETRY_DETECTION:
                return AgentAction(
                    action_type=ActionType.RETRY_DETECTION,
                    page_index=page.page_index,
                    reason="Brain proactively suggested rerunning detection.",
                )
            return None

        if recommended == "run_revision" and page.regions and not getattr(page, "auto_revision_done", False):
            return AgentAction(
                action_type=ActionType.RUN_REVISION,
                page_index=page.page_index,
                payload=self._auto_revision_payload(page),
                reason="Brain proactively suggested a text-only revision pass.",
            )

        return None

    def _guard_brain_rerun_action(self, job: JobState, page: PageState, plan: dict) -> AgentAction | None:
        strategy = self._strategy_for_plan(plan)
        preferred = str(strategy.get("preferred_rerun_action") or "").strip().lower()
        if not preferred:
            return None

        if preferred == "retry_typeset":
            if page.erased_image_ref or page.rendered_image_ref or page.stage in {PageStage.ERASED, PageStage.TYPESETTING, PageStage.DONE}:
                return AgentAction(
                    action_type=ActionType.RETRY_TYPESET,
                    page_index=page.page_index,
                    payload=self._skill_payload(page, "lettering"),
                    reason="Brain suggested re-typesetting as the safest rerun path.",
                )
            return None

        if preferred == "retry_erase":
            if page.regions:
                return AgentAction(
                    action_type=ActionType.RETRY_ERASE,
                    page_index=page.page_index,
                    payload=self._skill_payload(page, "lama_eraser"),
                    reason="Brain suggested re-erasing before re-typesetting.",
                )
            return None

        if preferred == "retry_vlm_page":
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            if pipeline_mode == "vlm":
                return AgentAction(
                    action_type=ActionType.RETRY_VLM_PAGE,
                    page_index=page.page_index,
                    payload={"target_language": job.target_language},
                    reason="Brain suggested rerunning the VLM core.",
                )
            return None

        if preferred == "retry_translation" and page.regions:
            return AgentAction(
                action_type=ActionType.RETRY_TRANSLATION,
                page_index=page.page_index,
                payload={"target_language": job.target_language},
                reason="Brain suggested rerunning text translation.",
            )

        if preferred == "retry_ocr" and page.regions:
            return AgentAction(
                action_type=ActionType.RETRY_OCR,
                page_index=page.page_index,
                reason="Brain suggested rerunning OCR.",
            )

        if preferred == "retry_detection":
            return AgentAction(
                action_type=ActionType.RETRY_DETECTION,
                page_index=page.page_index,
                reason="Brain suggested rerunning detection.",
            )

        if preferred == "run_revision" and page.regions:
            return AgentAction(
                action_type=ActionType.RUN_REVISION,
                page_index=page.page_index,
                payload=plan,
                reason="Brain suggested text-only revision instead of a heavier rerun.",
            )

        return None

    def _guard_brain_correction_action(self, job: JobState, page: PageState, plan: dict) -> AgentAction | None:
        strategy = self._strategy_for_plan(plan)
        preferred = str(strategy.get("preferred_correction_action") or "").strip().lower()
        if not preferred:
            return None

        if preferred == "run_revision":
            return AgentAction(
                action_type=ActionType.RUN_REVISION,
                page_index=page.page_index,
                payload=plan,
                reason="Brain suggested text-only revision for this correction.",
            )

        if preferred == "retry_vlm_page":
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            if pipeline_mode == "vlm":
                return AgentAction(
                    action_type=ActionType.RETRY_VLM_PAGE,
                    page_index=page.page_index,
                    payload={"target_language": job.target_language},
                    reason="Brain suggested rerunning the VLM core for this correction.",
                )
            return None

        if preferred == "retry_translation" and page.regions:
            return AgentAction(
                action_type=ActionType.RETRY_TRANSLATION,
                page_index=page.page_index,
                payload={"target_language": job.target_language},
                reason="Brain suggested rerunning translation for this correction.",
            )

        if preferred == "retry_ocr" and page.regions:
            return AgentAction(
                action_type=ActionType.RETRY_OCR,
                page_index=page.page_index,
                reason="Brain suggested rerunning OCR for this correction.",
            )

        if preferred == "retry_detection":
            return AgentAction(
                action_type=ActionType.RETRY_DETECTION,
                page_index=page.page_index,
                reason="Brain suggested rerunning detection for this correction.",
            )

        return None

    def decide_next_action(self, job: JobState) -> AgentAction:
        """
        根据整个 JobState 决定下一步动作。
        """
        # 0) If the brain produced a pending plan from user feedback, honor it first.
        plan = getattr(job, "pending_user_plan", None)
        if isinstance(plan, dict) and plan:
            return self._decide_for_pending_plan(job, plan)

        if job.status == JobStatus.COMPLETED:
            return AgentAction(ActionType.NO_OP, reason="Job already completed.")

        if job.status == JobStatus.FAILED:
            return AgentAction(ActionType.NO_OP, reason="Job already failed.")

        # 1) 优先处理用户指定重跑的页面
        for page_index in sorted(job.pages.keys()):
            page = job.pages[page_index]
            if page.needs_manual_rerun:
                return self._decide_rerun_for_page(page, job)

        job_level_action = self._guard_brain_job_action(job)
        if job_level_action is not None:
            return job_level_action

        default_batch_action = self._default_vlm_batch_action(job)
        if default_batch_action is not None:
            return default_batch_action

        # 2) 正常推进页面
        for page_index in sorted(job.pages.keys()):
            page = job.pages[page_index]
            action = self._decide_for_page(page, job)
            if action.action_type != ActionType.NO_OP:
                return action

        # 3) 所有页面都完成，则结束任务
        if job.pages and all(page.stage == PageStage.DONE for page in job.pages.values()):
            return AgentAction(ActionType.FINISH_JOB, reason="All pages completed.")

        return AgentAction(ActionType.NO_OP, reason="No action available.")

    def _decide_for_pending_plan(self, job: JobState, plan: dict) -> AgentAction:
        intent = str(plan.get("intent") or "unknown").strip().lower()

        needs_clarification = bool(plan.get("needs_clarification"))
        question = plan.get("clarifying_question")
        if needs_clarification and isinstance(question, str) and question.strip():
            return AgentAction(ActionType.ASK_USER, reason=question.strip())

        # Resolve a reference page index (support both 0-based and 1-based user numbers).
        page_index = None
        ref_num = plan.get("reference_page_number")
        try:
            ref_num = int(ref_num) if ref_num is not None else None
        except Exception:
            ref_num = None

        if ref_num is not None:
            # IMPORTANT: reference_page_number is intended to be human-facing (usually 1-based).
            # Prefer resolving as 1-based when possible to avoid off-by-one for inputs like "第1页".
            if ref_num == 0 and 0 in job.pages:
                page_index = 0
            elif (ref_num - 1) in job.pages:
                page_index = ref_num - 1
            elif ref_num in job.pages:
                page_index = ref_num

        if page_index is None:
            page_index = sorted(job.pages.keys())[0] if job.pages else 0

        if intent == "correction":
            guarded = self._guard_brain_correction_action(job, job.pages[page_index], plan) if page_index in job.pages else None
            if guarded is not None:
                return guarded
            return AgentAction(
                action_type=ActionType.RUN_REVISION,
                page_index=page_index,
                payload=plan,
                reason="Brain requested translation revision.",
            )

        if intent == "rerun_page":
            # Reuse existing rerun policy.
            if page_index in job.pages:
                guarded = self._guard_brain_rerun_action(job, job.pages[page_index], plan)
                if guarded is not None:
                    return guarded
                return self._decide_rerun_for_page(job.pages[page_index], job)
            return AgentAction(ActionType.ASK_USER, reason="Requested rerun page not found; please provide a valid page number.")

        # Unknown plan: ask user for clarification.
        return AgentAction(ActionType.ASK_USER, reason="I couldn't understand the request. Please provide a reference page number and what should be changed.")

    def _decide_rerun_for_page(self, page: PageState, job: JobState) -> AgentAction:
        """
        用户反馈该页效果不好时，决定从哪一步开始重跑。
        策略：
        - 如果嵌字有问题，从 typeset 重跑
        - 否则如果擦除已有结果，优先只重跑 typeset
        - 否则如果翻译缺失，从 translation 重跑
        - 否则如果 OCR 置信度低，从 OCR 重跑
        - 否则从 detection 重跑
        """
        pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()

        any_render_bad = any(
            (not region.render_ok) or region.render_overflow
            for region in page.regions.values()
        )
        if any_render_bad:
            return AgentAction(
                action_type=ActionType.RETRY_TYPESET,
                page_index=page.page_index,
                payload=self._skill_payload(page, "lettering"),
                reason="User requested rerun; typesetting issue detected.",
            )

        if page.erased_image_ref:
            return AgentAction(
                action_type=ActionType.RETRY_TYPESET,
                page_index=page.page_index,
                payload=self._skill_payload(page, "lettering"),
                reason="User requested rerun; reuse erased page and rerun typeset only.",
            )

        if pipeline_mode == "vlm":
            return AgentAction(
                action_type=ActionType.RETRY_VLM_PAGE,
                page_index=page.page_index,
                payload={"target_language": job.target_language},
                reason="User requested rerun; VLM pipeline reruns core step.",
            )

        any_translation_missing = any(
            region.translated_text is None for region in page.regions.values()
        )
        if any_translation_missing:
            return AgentAction(
                action_type=ActionType.RETRY_TRANSLATION,
                page_index=page.page_index,
                reason="User requested rerun; translation missing.",
            )

        low_ocr = any(
            region.ocr_confidence is not None and region.ocr_confidence < self.OCR_CONF_THRESHOLD
            for region in page.regions.values()
        )
        if low_ocr:
            return AgentAction(
                action_type=ActionType.RETRY_OCR,
                page_index=page.page_index,
                reason="User requested rerun; low OCR confidence.",
            )

        return AgentAction(
            action_type=ActionType.RETRY_DETECTION,
            page_index=page.page_index,
            reason="User requested rerun; fallback to detection rerun.",
        )

    def _decide_for_page(self, page: PageState, job: JobState) -> AgentAction:
        """
        决定某一页的下一步。
        """
        if page.stage == PageStage.NOT_STARTED:
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            if pipeline_mode == "vlm":
                batch_size = max(1, min(10, int(getattr(job, "vlm_batch_size", 10) or 10)))
                page_indices = []
                for candidate_index in sorted(job.pages.keys()):
                    if candidate_index < page.page_index:
                        continue
                    candidate_page = job.pages[candidate_index]
                    if candidate_page.stage != PageStage.NOT_STARTED:
                        if page_indices:
                            break
                        continue
                    page_indices.append(candidate_index)
                    if len(page_indices) >= batch_size:
                        break
                return AgentAction(
                    action_type=ActionType.RUN_VLM_BATCH,
                    payload={"page_indices": page_indices or [page.page_index], "target_language": job.target_language},
                    reason=f"Page window not started; run VLM batch for pages {(page_indices or [page.page_index])[0]}-{(page_indices or [page.page_index])[-1]}.",
                )

            return AgentAction(
                action_type=ActionType.RUN_DETECTION,
                page_index=page.page_index,
                reason="Page not started; run detection first.",
            )

        if page.stage == PageStage.DETECTED:
            if page.bbox_error:
                if page.retry_count_detection < self.MAX_RETRY_DETECTION:
                    return AgentAction(
                        action_type=ActionType.RETRY_DETECTION,
                        page_index=page.page_index,
                        reason="Detection bbox_error=True; retry detection.",
                    )
                return AgentAction(
                    action_type=ActionType.ASK_USER,
                    page_index=page.page_index,
                    reason="Detection still bad after retries; ask user.",
                )

            return AgentAction(
                action_type=ActionType.RUN_OCR,
                page_index=page.page_index,
                reason="Detection ok; proceed to OCR.",
            )

        if page.stage == PageStage.OCR_DONE:
            proactive = self._guard_brain_page_action(
                job,
                page,
                allowed_actions={"ask_user", "retry_ocr", "retry_detection"},
            )
            if proactive is not None:
                return proactive

            low_conf_regions = [
                region for region in page.regions.values()
                if region.ocr_confidence is not None and region.ocr_confidence < self.OCR_CONF_THRESHOLD
            ]
            if low_conf_regions:
                can_retry = all(
                    region.retry_count_ocr < self.MAX_RETRY_OCR
                    for region in low_conf_regions
                )
                if can_retry:
                    return AgentAction(
                        action_type=ActionType.RETRY_OCR,
                        page_index=page.page_index,
                        reason="Low OCR confidence detected; retry OCR.",
                    )
                return AgentAction(
                    action_type=ActionType.ASK_USER,
                    page_index=page.page_index,
                    reason="OCR still low confidence after retries; ask user.",
                )

            return AgentAction(
                action_type=ActionType.RUN_TRANSLATION,
                page_index=page.page_index,
                payload={"target_language": job.target_language},
                reason="OCR acceptable; proceed to translation.",
            )

        if page.stage == PageStage.TRANSLATED:
            if getattr(page, "page_type", "content") in {"cover", "info"}:
                page.rendered_image_ref = page.rendered_image_ref or page.image_ref
                page.lettering_ok = True
                page.stage = PageStage.DONE
                return AgentAction(
                    action_type=ActionType.NO_OP,
                    page_index=page.page_index,
                    reason=f"Non-content page ({getattr(page, 'page_type', 'content')}) skipped downstream text pipeline.",
                )

            page_strategy = self._strategy_for_page(page)
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            if pipeline_mode == "vlm":
                # 1) Judge can explicitly request user review.
                if getattr(page, "vlm_judge_decision", None) == "ask_user":
                    detail = getattr(page, "vlm_judge_explanation", None) or getattr(page, "vlm_judge_reason_code", None)
                    return AgentAction(
                        action_type=ActionType.ASK_USER,
                        page_index=page.page_index,
                        reason=f"VLM judge asked for review: {detail}",
                    )

                # 2) Judge can request retry even when heuristics passed.
                if getattr(page, "vlm_judge_decision", None) == "retry":
                    if page.retry_count_vlm < self.MAX_RETRY_VLM:
                        return AgentAction(
                            action_type=ActionType.RETRY_VLM_PAGE,
                            page_index=page.page_index,
                            payload={"target_language": job.target_language},
                            reason=f"VLM judge requested retry ({getattr(page, 'vlm_judge_reason_code', None)}).",
                        )
                    return AgentAction(
                        action_type=ActionType.ASK_USER,
                        page_index=page.page_index,
                        reason="VLM judge requested retry but retry budget exhausted; ask user.",
                    )

                # 3) Heuristic quality gate.
                if page.vlm_quality_ok is False:
                    # Retry only when it is likely to help:
                    # - downscaled result (retry at full res)
                    # - structural error in last attempt (request/json)
                    can_retry = page.retry_count_vlm < self.MAX_RETRY_VLM and (
                        page.vlm_downscaled is True
                        or (page.vlm_quality_reason in {"vlm_request_failed", "bad_json"})
                        or (page.vlm_quality_reason or "").startswith("vlm_zero_regions")
                        or (page.vlm_quality_reason or "") == "zero_regions_but_detector_found_text"
                        or (page.vlm_quality_reason or "").startswith("too_many_empty")
                        or (page.vlm_quality_reason == "target_language_mismatch")
                        or (page.vlm_quality_reason == "suspicious_bbox_geometry")
                        or (page.vlm_quality_reason or "") in {"bbox_too_large_for_text", "bbox_looks_loose", "overlapping_oversized_bboxes"}
                        or (page.vlm_quality_reason or "") in {"judge_retry"}
                    )
                    if can_retry:
                        return AgentAction(
                            action_type=ActionType.RETRY_VLM_PAGE,
                            page_index=page.page_index,
                            payload={"target_language": job.target_language},
                            reason=f"VLM quality not ok ({page.vlm_quality_reason}); retry VLM core.",
                        )

                    return AgentAction(
                        action_type=ActionType.ASK_USER,
                        page_index=page.page_index,
                        reason=f"VLM quality not ok ({page.vlm_quality_reason}); ask user.",
                    )

            proactive = self._guard_brain_page_action(
                job,
                page,
                allowed_actions={
                    "ask_user",
                    "retry_vlm_page",
                    "retry_translation",
                    "retry_ocr",
                    "retry_detection",
                },
            )
            if proactive is not None:
                return proactive

            if not page.translation_consistent:
                can_retry = all(
                    region.retry_count_translation < self.MAX_RETRY_TRANSLATION
                    for region in page.regions.values()
                )
                if can_retry:
                    return AgentAction(
                        action_type=ActionType.RETRY_TRANSLATION,
                        page_index=page.page_index,
                        payload={"target_language": job.target_language},
                        reason="Translation inconsistency detected; retry translation.",
                    )
                return AgentAction(
                    action_type=ActionType.ASK_USER,
                    page_index=page.page_index,
                    reason="Translation still inconsistent; ask user.",
                )

            auto_revision_action = self._auto_revision_action_for_page(job, page)
            if auto_revision_action is not None:
                return auto_revision_action

            return AgentAction(
                action_type=ActionType.RUN_ERASE,
                page_index=page.page_index,
                payload=self._skill_payload(page, "lama_eraser"),
                reason="Translation ok; proceed to erase original text.",
            )

        if page.stage == PageStage.ERASED:
            erase_qa = page.qa_reports.get("erase") if isinstance(page.qa_reports, dict) else {}
            erase_qa_status = str((erase_qa or {}).get("status") or "").strip().lower()
            proactive = self._guard_brain_page_action(
                job,
                page,
                allowed_actions={"ask_user", "retry_erase", "run_revision"},
            )
            if proactive is not None:
                return proactive

            if erase_qa_status in {"retryable", "bad"} and page.retry_count_erase < self.MAX_RETRY_ERASE:
                return AgentAction(
                    action_type=ActionType.RETRY_ERASE,
                    page_index=page.page_index,
                    payload=self._skill_payload(page, "lama_eraser"),
                    reason=f"Erase QA marked this page as {erase_qa_status}; retry erase before typesetting.",
                )

            return AgentAction(
                action_type=ActionType.RUN_TYPESET,
                page_index=page.page_index,
                payload=self._skill_payload(page, "lettering"),
                reason="Erase ok; proceed to typesetting.",
            )

        if page.stage == PageStage.TYPESETTING:
            bad_render_regions = [
                region for region in page.regions.values()
                if (not region.render_ok) or region.render_overflow
            ]
            typeset_qa = (page.typeset_metrics or {}).get("qa") if isinstance(page.typeset_metrics, dict) else {}
            typeset_qa_status = str((typeset_qa or {}).get("status") or "").strip().lower()
            erase_qa = page.qa_reports.get("erase") if isinstance(page.qa_reports, dict) else {}
            erase_qa_status = str((erase_qa or {}).get("status") or "").strip().lower()
            if bad_render_regions or typeset_qa_status in {"retryable", "bad"}:
                proactive = self._guard_brain_page_action(
                    job,
                    page,
                    allowed_actions={
                        "ask_user",
                        "retry_erase",
                        "retry_typeset",
                        "retry_translation",
                        "retry_vlm_page",
                        "run_revision",
                    },
                )
                if proactive is not None:
                    return proactive

                if erase_qa_status in {"retryable", "bad"} and page.retry_count_erase < self.MAX_RETRY_ERASE:
                    return AgentAction(
                        action_type=ActionType.RETRY_ERASE,
                        page_index=page.page_index,
                        payload=self._skill_payload(page, "lama_eraser"),
                        reason="Typesetting issue likely rooted in erase QA; retry erase first.",
                    )

                page_strategy = self._strategy_for_page(page)
                preferred_rerun = str(page_strategy.get("preferred_rerun_action") or "").strip().lower()
                if preferred_rerun == "retry_erase" and page.retry_count_erase < self.MAX_RETRY_ERASE:
                    return AgentAction(
                        action_type=ActionType.RETRY_ERASE,
                        page_index=page.page_index,
                        payload=self._skill_payload(page, "lama_eraser"),
                        reason="Page strategy suggests retrying erase before another typeset attempt.",
                    )
                can_retry = all(
                    region.retry_count_render < self.MAX_RETRY_RENDER
                    for region in (bad_render_regions or list(page.regions.values()))
                )
                if can_retry:
                    return AgentAction(
                        action_type=ActionType.RETRY_TYPESET,
                        page_index=page.page_index,
                        payload=self._skill_payload(page, "lettering"),
                        reason="Typesetting QA/layout issue detected; retry typeset.",
                    )
                return AgentAction(
                    action_type=ActionType.ASK_USER,
                    page_index=page.page_index,
                    reason="Typesetting failed after retries; ask user.",
                )

            page.stage = PageStage.DONE
            return AgentAction(
                action_type=ActionType.NO_OP,
                page_index=page.page_index,
                reason="Page marked done.",
            )

        if page.stage == PageStage.ERASING:
            if page.erase_ok and page.erased_image_ref:
                return AgentAction(
                    action_type=ActionType.RUN_TYPESET,
                    page_index=page.page_index,
                    payload=self._skill_payload(page, "lettering"),
                    reason="Erase completed; proceed to typesetting.",
                )

            if page.retry_count_erase < self.MAX_RETRY_ERASE:
                return AgentAction(
                    action_type=ActionType.RETRY_ERASE,
                    page_index=page.page_index,
                    payload=self._skill_payload(page, "lama_eraser"),
                    reason="Erase did not finish cleanly; retry erase.",
                )
            return AgentAction(
                action_type=ActionType.ASK_USER,
                page_index=page.page_index,
                reason="Erase failed after retries; ask user.",
            )

        if page.stage in [PageStage.DONE, PageStage.FAILED]:
            pipeline_mode = str(job.user_preferences.get("pipeline_mode") or "classic").lower()
            if pipeline_mode == "vlm" and page.stage == PageStage.FAILED:
                page_error_stage = str(getattr(page, "page_error_stage", "") or "").strip().lower()
                page_error_code = str(getattr(page, "page_error_code", "") or "").strip().lower()
                erase_metrics = getattr(page, "erase_metrics", None) or {}
                service_error = erase_metrics.get("service_error") if isinstance(erase_metrics, dict) else None
                if page_error_stage == "erase" and (
                    isinstance(service_error, dict)
                    or "service_unreachable" in page_error_code
                    or "service_unavailable" in page_error_code
                    or "timeout" in page_error_code
                ):
                    return AgentAction(
                        action_type=ActionType.ASK_USER,
                        page_index=page.page_index,
                        reason="Windows 远程擦除服务当前无响应，请先重启该服务，再重试本页擦除。",
                    )
                # Allow retrying VLM core if the failure came from the VLM step.
                if (
                    page.page_error_message
                    and page.page_error_message.startswith("VLM")
                    and page.retry_count_vlm < self.MAX_RETRY_VLM
                ):
                    return AgentAction(
                        action_type=ActionType.RETRY_VLM_PAGE,
                        page_index=page.page_index,
                        payload={"target_language": job.target_language},
                        reason="VLM core failed; retry within budget.",
                    )

            return AgentAction(
                action_type=ActionType.NO_OP,
                page_index=page.page_index,
                reason=f"Page already terminal: {page.stage.value}",
            )

        return AgentAction(
            action_type=ActionType.NO_OP,
            page_index=page.page_index,
            reason="No decision matched.",
        )
