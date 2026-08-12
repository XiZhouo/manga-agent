from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from skills.input_normalization.normalize import normalize_file

from .agent import ComicAgent
from .brain_contracts import BrainIntent
from .file_resolver import extract_path_candidates, find_by_filename, resolve_input_path
from .job_io import build_job_id, resolve_job_dir_from_path, skill_dir
from .state import JobState, JobStatus


class ChatController:
    def __init__(self, agent: ComicAgent, *, cwd: Path) -> None:
        self.agent = agent
        self.cwd = cwd.resolve()
        self.jobs_root = self.cwd / "data" / "chat_jobs"
        self.output_root = self.cwd / "data" / "output"
        self._recent_chat_turns: List[Dict[str, str]] = []
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def handle_message(self, user_message: str, active_job_id: Optional[str] = None) -> Dict[str, Any]:
        active_job = self._safe_get_job(active_job_id) if active_job_id else None
        brain_context = self._build_chat_brain_context(active_job, user_message=user_message)
        plan = self.agent.brain.plan_chat_message(
            user_message=user_message,
            active_job_id=active_job_id,
            has_active_job=active_job is not None,
            target_language=(active_job.target_language if active_job else "zh"),
            pipeline_mode=str((active_job.user_preferences or {}).get("pipeline_mode") or "vlm") if active_job else "vlm",
            brain_context=brain_context,
            skill_context={},
        )

        intent = str(plan.get("intent") or BrainIntent.UNKNOWN.value)
        if plan.get("needs_clarification"):
            return {
                "ok": True,
                "action": "clarify",
                "job_id": active_job_id,
                "message": str(plan.get("clarifying_question") or "请再补充一点信息。"),
                "plan": plan,
            }

        if intent == BrainIntent.CREATE_TRANSLATION_JOB.value:
            return self._handle_create_job(user_message=user_message, plan=plan)
        if intent == BrainIntent.QUERY_PROGRESS.value:
            target_job = active_job or self._latest_job()
            if target_job is None:
                return {
                    "ok": True,
                    "action": "status",
                    "job_id": None,
                    "message": "当前还没有任务。你可以直接说：帮我翻译 `./mangatest/xxx.cbz`。",
                    "plan": plan,
                }
            return {
                "ok": True,
                "action": "status",
                "job_id": target_job.job_id,
                "message": self._format_job_status(target_job),
                "plan": plan,
            }
        if intent == BrainIntent.RESUME_JOB.value:
            target_job = self._resolve_resume_target(plan=plan, active_job=active_job)
            if target_job is None:
                return {
                    "ok": True,
                    "action": "clarify",
                    "job_id": None,
                    "message": "我还没找到要恢复的任务。你可以提供 job_id，或者先创建一个新任务。",
                    "plan": plan,
                }
            return {
                "ok": True,
                "action": "resumed_job",
                "job_id": target_job.job_id,
                "message": f"已切换到任务 `{target_job.job_id}`。\n\n{self._format_job_status(target_job)}",
                "plan": plan,
            }
        if intent == BrainIntent.EXPORT_RESULT.value:
            target_job = active_job or self._latest_job()
            if target_job is None:
                return {
                    "ok": True,
                    "action": "error",
                    "job_id": None,
                    "message": "当前没有可导出的任务。",
                    "plan": plan,
                }
            zip_path = self._export_job(target_job)
            return {
                "ok": True,
                "action": "exported",
                "job_id": target_job.job_id,
                "message": f"已导出结果到 `{zip_path}`。",
                "plan": plan,
            }
        if intent == BrainIntent.IMPORT_RAG_BUNDLE.value:
            target_job = active_job or self._latest_job()
            if target_job is None:
                return {
                    "ok": True,
                    "action": "error",
                    "job_id": None,
                    "message": "当前没有活动任务，无法导入 RAG bundle。请先创建或恢复任务。",
                    "plan": plan,
                }
            return self._handle_import_rag(target_job=target_job, user_message=user_message, plan=plan)
        if intent in {BrainIntent.REVISE_TRANSLATION.value, BrainIntent.RERUN_PAGE.value}:
            target_job = active_job or self._latest_job()
            if target_job is None:
                return {
                    "ok": True,
                    "action": "clarify",
                    "job_id": None,
                    "message": "当前没有活动任务。请先提供漫画文件路径创建任务，或者恢复一个已有任务。",
                    "plan": plan,
                }
            self.agent.handle_user_message(target_job.job_id, user_message)
            updated = self.agent.run_until_waiting_or_done(target_job.job_id)
            return {
                "ok": True,
                "action": "updated_job",
                "job_id": updated.job_id,
                "message": self._format_job_status(updated),
                "plan": plan,
            }

        chat_reply = self.agent.brain.reply_chat_message(user_message=user_message, brain_context=brain_context)
        self._append_recent_chat_turn(user_message=user_message, assistant_message=chat_reply)
        if active_job is not None:
            self.agent.memory.store.append_brain_event(
                active_job,
                event_type="chat",
                message=user_message,
                summary=chat_reply,
                extra={"source": "chat_controller", "scope": "chat"},
            )
        return {
            "ok": True,
            "action": "chat",
            "job_id": active_job_id,
            "message": chat_reply,
            "plan": plan,
        }

    def _handle_create_job(self, *, user_message: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
        raw_path = str(payload.get("input_path") or "").strip()
        path = self._resolve_source_path(raw_path or user_message)
        if path is None:
            return {
                "ok": True,
                "action": "clarify",
                "job_id": None,
                "message": "请把漫画文件路径发我，例如 `./mangatest/xxx.cbz` 或 `/Users/jay/Downloads/xxx.cbz`。",
                "plan": plan,
            }

        pages = normalize_file(path.read_bytes(), path.suffix.lower().lstrip("."))
        job_id = build_job_id(path.name)
        job_dir = self.jobs_root / job_id
        pages_dir = skill_dir(job_dir, "normalization") / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)

        page_refs: List[tuple[int, str]] = []
        for item in pages:
            out_name = f"page_{int(item.page_index):04d}{Path(item.filename).suffix.lower() or '.png'}"
            out_path = pages_dir / out_name
            out_path.write_bytes(item.image_bytes)
            page_refs.append((int(item.page_index), str(out_path)))

        target_language = str(payload.get("target_language") or "zh")
        job_id = self.agent.create_job(
            user_id="chat-user",
            source_file_name=path.name,
            target_language=target_language,
            pages=page_refs,
            job_id=job_id,
            job_dir=str(job_dir),
        )
        job = self.agent.store.get_job(job_id)
        job.user_preferences["pipeline_mode"] = str(payload.get("pipeline_mode") or "vlm")
        job.user_preferences["reading_direction"] = str(payload.get("reading_direction") or "ltr")
        job.user_preferences["source_language"] = str(payload.get("source_language") or "en")
        job.logs.append(f"ChatController created job from {path}")
        self.agent.store.save_job(job)

        final_job = self.agent.run_until_waiting_or_done(job_id)
        return {
            "ok": True,
            "action": "created_job",
            "job_id": final_job.job_id,
            "message": f"已创建并启动任务 `{final_job.job_id}`。\n\n{self._format_job_status(final_job)}",
            "plan": plan,
        }

    def _handle_import_rag(self, *, target_job: JobState, user_message: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
        raw_path = str(payload.get("rag_bundle_path") or "").strip()
        path = self._resolve_source_path(raw_path or user_message, allowed_suffixes={".json"})
        if path is None:
            return {
                "ok": True,
                "action": "clarify",
                "job_id": target_job.job_id,
                "message": "请提供要导入的 RAG bundle 路径，例如 `./memory/batman_bundle.json`。",
                "plan": plan,
            }

        bundle = json.loads(path.read_text(encoding="utf-8"))
        imported = self.agent.memory.store.import_rag_bundle(target_job, bundle)
        buckets = bundle.get("buckets") if isinstance(bundle, dict) else {}
        if isinstance(buckets, dict):
            for bucket in ("glossary", "translation_memory", "style_memory", "project_memory"):
                raw = buckets.get(bucket) if isinstance(buckets.get(bucket), dict) else None
                entries = raw.get("entries") if isinstance(raw, dict) else None
                if isinstance(entries, list):
                    try:
                        self.agent.memory.shared_store.upsert_entries(
                            target_job,
                            bucket,
                            [x for x in entries if isinstance(x, dict)],
                        )
                    except Exception:
                        pass
        self.agent.store.save_job(target_job)
        imported_text = ", ".join(f"{k}={v}" for k, v in imported.items())
        return {
            "ok": True,
            "action": "updated_job",
            "job_id": target_job.job_id,
            "message": f"已导入 RAG bundle `{path}`，新增：{imported_text or '0'}。",
            "plan": plan,
        }

    def _resolve_source_path(self, raw: str, *, allowed_suffixes: Optional[set[str]] = None) -> Optional[Path]:
        candidates = extract_path_candidates(raw)
        for item in candidates:
            resolved = resolve_input_path(item, self.cwd, allowed_suffixes=allowed_suffixes)
            if resolved is not None:
                return resolved

        for item in candidates:
            resolved = find_by_filename(Path(item).name, self.cwd, allowed_suffixes=allowed_suffixes)
            if resolved is not None:
                return resolved

        direct = resolve_input_path(str(raw).strip(), self.cwd, allowed_suffixes=allowed_suffixes)
        if direct is not None:
            return direct
        return None

    def _safe_get_job(self, job_id: Optional[str]) -> Optional[JobState]:
        if not job_id:
            return None
        try:
            return self.agent.store.get_job(job_id)
        except Exception:
            return None

    def _latest_job(self) -> Optional[JobState]:
        jobs = getattr(self.agent.store, "_jobs", None)
        if isinstance(jobs, dict) and jobs:
            return max(jobs.values(), key=lambda item: float(getattr(item, "updated_at", 0.0) or 0.0))
        return None

    def _resolve_resume_target(self, *, plan: Dict[str, Any], active_job: Optional[JobState]) -> Optional[JobState]:
        payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
        job_id = str(payload.get("job_id") or "").strip()
        if job_id:
            return self._safe_get_job(job_id)
        return active_job or self._latest_job()

    def _build_chat_brain_context(self, active_job: Optional[JobState], *, user_message: str = "") -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "active_job_id": getattr(active_job, "job_id", None),
            "active_job_status": getattr(active_job, "status", None).value if getattr(active_job, "status", None) else None,
            "page_count": len(getattr(active_job, "pages", {}) or {}) if active_job else 0,
            "recent_chat_history": list(self._recent_chat_turns[-6:]),
        }
        latest = self._latest_job()
        if latest is not None:
            context["latest_job_id"] = latest.job_id
            context["latest_job_status"] = latest.status.value
            context["latest_source_file_name"] = latest.source_file_name
        if active_job is not None:
            try:
                chat_memory = self.agent.memory.build_chat_context(active_job, user_message=user_message)
                progress_memory = self.agent.memory.build_progress_context(active_job, user_message=user_message)
                context["chat_memory"] = chat_memory
                context["progress_memory"] = progress_memory
                snapshot = self.agent.memory.store.snapshot(active_job)
                brain_memory = snapshot.get("brain_memory") if isinstance(snapshot, dict) else {}
                entries = (brain_memory or {}).get("entries") if isinstance(brain_memory, dict) else []
                if isinstance(entries, list):
                    recent_job_brain = []
                    for item in entries[-6:]:
                        if not isinstance(item, dict):
                            continue
                        recent_job_brain.append(
                            {
                                "type": item.get("type"),
                                "user": item.get("src_text"),
                                "assistant": item.get("tgt_text"),
                                "updated_at": item.get("updated_at"),
                            }
                        )
                    context["recent_job_brain_memory"] = recent_job_brain
            except Exception:
                pass
        return context

    def _append_recent_chat_turn(self, *, user_message: str, assistant_message: str, keep_last: int = 12) -> None:
        self._recent_chat_turns.append(
            {
                "user": str(user_message or "").strip(),
                "assistant": str(assistant_message or "").strip(),
            }
        )
        if keep_last > 0:
            self._recent_chat_turns = self._recent_chat_turns[-keep_last:]

    def _format_job_status(self, job: JobState) -> str:
        page_count = len(job.pages)
        done_pages = sum(1 for page in job.pages.values() if page.stage.value == "done")
        active_pages = [page for page in job.pages.values() if page.stage.value not in {"done", "not_started"}]
        stage_counts: Dict[str, int] = {}
        for page in job.pages.values():
            stage_counts[page.stage.value] = stage_counts.get(page.stage.value, 0) + 1

        lines = [
            f"任务 `{job.job_id}` | 状态：{job.status.value}",
            f"源文件：{job.source_file_name}",
            f"页面进度：{done_pages}/{page_count} 已完成",
        ]
        if active_pages:
            ordered = sorted(active_pages, key=lambda page: page.page_index)
            lines.append(
                "当前活跃页："
                + ", ".join(f"p{page.page_index}:{page.stage.value}" for page in ordered[:6])
            )
        lines.append("阶段统计：" + ", ".join(f"{stage}={count}" for stage, count in sorted(stage_counts.items())))
        if job.waiting_reason:
            lines.append(f"等待原因：{job.waiting_reason}")
        if job.logs:
            lines.append("最近日志：")
            for item in job.logs[-4:]:
                lines.append(f"- {item}")
        return "\n".join(lines)

    def _export_job(self, job: JobState) -> Path:
        from main import _build_run_manifest, _package_zip, _write_artifacts

        first_page = next(iter(sorted(job.pages.values(), key=lambda page: page.page_index)), None)
        if first_page and first_page.image_ref:
            job_dir = resolve_job_dir_from_path(Path(first_page.image_ref).resolve())
        else:
            job_dir = self.jobs_root / job.job_id
            job_dir.mkdir(parents=True, exist_ok=True)

        run_manifest = _build_run_manifest(job)
        glossary = job.glossary if isinstance(job.glossary, dict) else {"version": "0.1.0", "entries": []}
        rag_bundle = self.agent.memory.store.export_rag_bundle(job)
        _write_artifacts(job_dir, run_manifest=run_manifest, glossary=glossary, rag_bundle=rag_bundle)

        output_zip = self.output_root / f"{job.job_id}.zip"
        _package_zip(output_zip, job_dir=job_dir, run_manifest=run_manifest, glossary=glossary, rag_bundle=rag_bundle)
        return output_zip
