from __future__ import annotations

import os
import re
import time
import uuid
from datetime import datetime, timezone
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


class SimpleMemory:
    """
    极简 memory：
    这里只更新用户偏好，不存完整业务状态。
    业务状态由 JobState/PageState/RegionState 存。
    """

    def update_preferences(self, job: JobState, user_message: str) -> None:
        if "中文" in user_message:
            job.user_preferences["target_language"] = "zh"
        if "英文" in user_message:
            job.user_preferences["target_language"] = "en"
        if "更夸张" in user_message:
            job.user_preferences["sfx_style"] = "exaggerated"
        if "术语统一" in user_message:
            job.user_preferences["strict_glossary"] = True


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
            plan = self.brain.plan_user_message(
                user_message=user_message,
                page_count=len(job.pages),
                pipeline_mode=pipeline_mode,
                target_language=job.target_language,
            )
            if isinstance(plan, dict) and plan:
                plan.setdefault("user_message", user_message)
                job.pending_user_plan = plan
                job.logs.append(f"Brain plan: intent={plan.get('intent')} scope={plan.get('scope')} ref={plan.get('reference_page_number')} conf={plan.get('confidence')}")
        except Exception as e:
            job.logs.append(f"Brain planning skipped/failed: {e}")

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
            return int("".join(digits))
        return None

    # -----------------------------------------------------
    # 对外接口 3：推进任务，直到需要用户或任务完成
    # -----------------------------------------------------
    def run_until_waiting_or_done(self, job_id: str, max_steps: int = 50) -> JobState:
        job = self.store.get_job(job_id)

        if job.status == JobStatus.CREATED:
            job.status = JobStatus.RUNNING
            self.store.save_job(job)

        for _ in range(max_steps):
            job = self.store.get_job(job_id)
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
        job.waiting_reason = "Max steps reached; waiting for user or operator."
        self._log(job, "Max steps reached; stopping loop.")
        self.store.save_job(job)
        return job

    # -----------------------------------------------------
    # 内部：执行具体动作
    # -----------------------------------------------------
    def _execute_action(self, job: JobState, action: AgentAction) -> None:
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

        if action.action_type in [ActionType.RUN_LETTERING, ActionType.RETRY_LETTERING]:
            self._do_lettering(job, page, retry=(action.action_type == ActionType.RETRY_LETTERING))
            return

        raise ValueError(f"Unsupported action type: {action.action_type}")

    def _do_revision(self, job: JobState, action: AgentAction) -> None:
        """Apply text-only translation revisions (no VLM rerun).

        The revision skill returns the same VLM-style schema (page->regions with src/tgt texts).
        We merge updated tgt_text back into RegionState, then re-run lettering for changed pages.
        """

        plan = action.payload or {}

        # Prepare full-book payload in VLM-like schema.
        pages_payload: List[Dict[str, Any]] = []
        for page_index in sorted(job.pages.keys()):
            page = job.pages[page_index]
            regions_payload = []
            for region_id in sorted(page.regions.keys()):
                r = page.regions[region_id]
                regions_payload.append(
                    {
                        "region_id": r.region_id,
                        "bbox_xywh": r.bbox,
                        "src_text": r.ocr_text or "",
                        "tgt_text": r.translated_text or "",
                    }
                )
            pages_payload.append({"page_index": page_index, "regions": regions_payload})

        reading_direction = str(job.user_preferences.get("reading_direction") or "ltr")

        result = self.tools.run_vlm_translation_revision(
            pages=pages_payload,
            user_request=str(plan.get("user_message") or plan.get("raw") or ""),
            reference_page_number=plan.get("reference_page_number"),
            scope=str(plan.get("scope") or "page_only"),
            from_side=str(plan.get("from_side") or "auto"),
            from_text=plan.get("from_text"),
            to_text=plan.get("to_text"),
            reading_direction=reading_direction,
            target_language=job.target_language,
            glossary=job.glossary or {},
        )

        if not result.get("ok"):
            err = result.get("error") or {}
            job.pending_user_plan = None
            job.logs.append(f"Revision failed: {err.get('code')} {err.get('message')}")
            job.status = JobStatus.WAITING_USER
            job.waiting_reason = f"Revision failed: {err.get('message') or err.get('code') or 'unknown'}"
            return

        data = result.get("data") or {}
        decision = str(data.get("decision") or "apply").strip().lower()
        if decision == "ask_user":
            q = str(data.get("ask_user") or "Please clarify your correction request.").strip()
            job.pending_user_plan = plan
            job.pending_user_plan["needs_clarification"] = True
            job.pending_user_plan["clarifying_question"] = q
            return

        pages_out = data.get("pages") or []
        changed_pages = set(int(x) for x in (data.get("changed_pages") or []) if str(x).strip().isdigit())

        # Merge tgt_text back into state.
        changed_regions = 0
        for p in pages_out:
            try:
                pi = int(p.get("page_index"))
            except Exception:
                continue
            if pi not in job.pages:
                continue

            page = job.pages[pi]
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
                if (region.translated_text or "") != new_tgt:
                    region.translated_text = new_tgt
                    region.stage = RegionStage.TRANSLATED
                    region.render_ok = False
                    region.render_overflow = False
                    changed_regions += 1
                    changed_pages.add(pi)

        # Update glossary bucket.
        glossary_updates = data.get("glossary_updates") or []
        if isinstance(job.glossary, dict):
            entries = job.glossary.get("entries")
            if not isinstance(entries, list):
                entries = []
            if isinstance(glossary_updates, list):
                entries.extend([x for x in glossary_updates if isinstance(x, dict)])
            job.glossary["entries"] = entries
            job.glossary["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        # Re-run lettering on changed pages.
        for pi in changed_pages:
            if pi in job.pages:
                job.pages[pi].stage = PageStage.TRANSLATED
                job.pages[pi].lettering_ok = False

        job.logs.append(f"Revision applied: changed_pages={sorted(changed_pages)} changed_regions={changed_regions}")
        job.pending_user_plan = None

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
                stage=RegionStage.DETECTED,
            )

        page.detection_ok = True
        page.detection_count = len(page.regions)
        page.bbox_error = result.get("bbox_error", False)
        page.stage = PageStage.DETECTED
        page.needs_manual_rerun = False

        self._log(job, f"Page {page.page_index}: detection done, regions={page.detection_count}")

    def _do_vlm_page(self, job: JobState, page: PageState, target_language: str, retry: bool) -> None:
        # VLM core does: detect regions + OCR + translate.
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

        reading_direction = str(job.user_preferences.get("reading_direction") or "ltr")
        source_language = str(job.user_preferences.get("source_language") or "en")

        result = self.tools.run_vlm_page(
            page,
            reading_direction=reading_direction,
            source_language=source_language,
            target_language=target_language,
            force_fullres=retry,
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

        page.regions.clear()
        for item in result.get("regions", []) or []:
            region_id = int(item["region_id"])
            page.regions[region_id] = RegionState(
                region_id=region_id,
                bbox=item.get("bbox"),
                region_type=item.get("region_type", "text"),
                ocr_text=item.get("src_text"),
                translated_text=item.get("tgt_text"),
                stage=RegionStage.TRANSLATED,
            )

        page.detection_ok = True
        page.detection_count = len(page.regions)
        page.bbox_error = bool(result.get("bbox_error", False))
        page.avg_ocr_confidence = None
        page.translation_consistent = True
        page.stage = PageStage.TRANSLATED
        page.needs_manual_rerun = False

        # Record VLM metrics for later QA/debug.
        vm = result.get("vlm_metrics") or {}
        page.vlm_orig_size = list(vm.get("orig") or []) or None
        page.vlm_sent_size = list(vm.get("sent") or []) or None
        page.vlm_downscaled = vm.get("downscaled") if vm.get("downscaled") is not None else None
        page.vlm_usage = vm.get("usage") if isinstance(vm.get("usage"), dict) else None

        # Evaluate quality. If not ok, DecisionEngine may trigger a retry.
        q_ok, q_reason, q_warn = self._evaluate_vlm_page_quality(job, page, target_language=target_language)
        page.vlm_quality_ok = q_ok
        page.vlm_quality_reason = q_reason
        page.vlm_warnings = q_warn

        # Hybrid policy: obvious failures are handled by heuristics above;
        # non-obvious cases go to a text-only LLM judge (no image upload).
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
        self._log(
            job,
            f"Page {page.page_index}: VLM core done, regions={page.detection_count}, orig={vm.get('orig')} sent={vm.get('sent')} downscaled={vm.get('downscaled')}{usage_str}",
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

    def _evaluate_vlm_page_quality(self, job: JobState, page: PageState, *, target_language: str) -> tuple[bool, Optional[str], list[str]]:
        """Heuristic quality gate for VLM results.

        Goals:
        - Catch obvious failures (empty texts, wrong language, 0 regions when detector suggests text exists)
        - Keep false positives low to avoid spending extra tokens
        """

        warnings: list[str] = []
        regions = list(page.regions.values())
        n = len(regions)

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
        result = self.tools.run_translation(page, target_language=target_language)

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
        page.stage = PageStage.TRANSLATED

        self._log(job, f"Page {page.page_index}: translation done, consistent={page.translation_consistent}")

    def _do_lettering(self, job: JobState, page: PageState, retry: bool) -> None:
        page.stage = PageStage.LETTERING
        result = self.tools.run_lettering(page)

        if not result["ok"]:
            page.stage = PageStage.FAILED
            page.page_error_message = "Lettering failed."
            self._log(job, f"Page {page.page_index}: lettering failed.")
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
        page.stage = PageStage.LETTERING

        self._log(job, f"Page {page.page_index}: lettering done, all_ok={all_ok}")

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