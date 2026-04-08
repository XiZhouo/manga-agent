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

    def _skill_payload(self, page: PageState, skill_name: str) -> dict:
        suggestions = getattr(page, "skill_param_suggestions", {}) or {}
        params = suggestions.get(skill_name) if isinstance(suggestions, dict) else None
        return {"skill_params": dict(params)} if isinstance(params, dict) and params else {}

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
                return AgentAction(
                    action_type=ActionType.RUN_VLM_PAGE,
                    page_index=page.page_index,
                    payload={"target_language": job.target_language},
                    reason="Page not started; run VLM core (detect+OCR+translate).",
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
                        or (page.vlm_quality_reason or "").startswith("too_many_empty")
                        or (page.vlm_quality_reason == "target_language_mismatch")
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

            if not getattr(page, "auto_revision_done", False):
                return AgentAction(
                    action_type=ActionType.RUN_REVISION,
                    page_index=page.page_index,
                    payload={
                        "revision_mode": "auto_rag_revision",
                        "scope": "page_only",
                        "reference_page_number": page.page_index + 1,
                        "user_message": "Auto RAG consistency revision before erase.",
                    },
                    reason="Translation produced; run post-translation RAG self-check before erase.",
                )

            return AgentAction(
                action_type=ActionType.RUN_ERASE,
                page_index=page.page_index,
                payload=self._skill_payload(page, "lama_eraser"),
                reason="Translation ok; proceed to erase original text.",
            )

        if page.stage == PageStage.ERASED:
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
            if bad_render_regions:
                can_retry = all(
                    region.retry_count_render < self.MAX_RETRY_RENDER
                    for region in bad_render_regions
                )
                if can_retry:
                    return AgentAction(
                        action_type=ActionType.RETRY_TYPESET,
                        page_index=page.page_index,
                        payload=self._skill_payload(page, "lettering"),
                        reason="Typesetting overflow/bad render; retry typeset.",
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
