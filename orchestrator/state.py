from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import time


class JobStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


class PageStage(str, Enum):
    NOT_STARTED = "not_started"
    DETECTING = "detecting"
    DETECTED = "detected"
    OCRING = "ocring"
    OCR_DONE = "ocr_done"
    TRANSLATING = "translating"
    TRANSLATED = "translated"
    LETTERING = "lettering"
    DONE = "done"
    FAILED = "failed"


class RegionStage(str, Enum):
    DETECTED = "detected"
    OCR_DONE = "ocr_done"
    TRANSLATED = "translated"
    RENDERED = "rendered"
    FAILED = "failed"


class ActionType(str, Enum):
    RUN_DETECTION = "run_detection"
    RUN_OCR = "run_ocr"
    RUN_TRANSLATION = "run_translation"
    RUN_VLM_PAGE = "run_vlm_page"
    RUN_REVISION = "run_revision"
    RUN_LETTERING = "run_lettering"

    RETRY_DETECTION = "retry_detection"
    RETRY_OCR = "retry_ocr"
    RETRY_TRANSLATION = "retry_translation"
    RETRY_VLM_PAGE = "retry_vlm_page"
    RETRY_LETTERING = "retry_lettering"

    ASK_USER = "ask_user"
    FINISH_JOB = "finish_job"
    FAIL_JOB = "fail_job"
    NO_OP = "no_op"


@dataclass
class RegionState:
    """
    单个文本区域的状态。
    region_id 通常在单页内唯一。
    """
    region_id: int
    bbox: Optional[List[int]] = None  # [x, y, w, h]
    region_type: str = "text"

    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None

    translated_text: Optional[str] = None
    translation_confidence: Optional[float] = None

    render_ok: bool = False
    render_overflow: bool = False

    retry_count_ocr: int = 0
    retry_count_translation: int = 0
    retry_count_render: int = 0

    error_message: Optional[str] = None
    stage: RegionStage = RegionStage.DETECTED


@dataclass
class PageState:
    """
    单页状态。Agent 的大部分调度决策都基于 PageState。
    """
    page_index: int
    source_filename: Optional[str] = None
    image_ref: Optional[str] = None  # 文件路径、对象存储 key、数据库引用等

    stage: PageStage = PageStage.NOT_STARTED
    regions: Dict[int, RegionState] = field(default_factory=dict)

    # 页面级质量指标
    detection_ok: bool = False
    detection_count: int = 0
    bbox_error: bool = False

    # VLM pipeline metrics / quality (optional)
    retry_count_vlm: int = 0
    vlm_quality_ok: bool = True
    vlm_quality_reason: Optional[str] = None
    vlm_warnings: List[str] = field(default_factory=list)
    vlm_downscaled: Optional[bool] = None
    vlm_orig_size: Optional[List[int]] = None  # [w,h]
    vlm_sent_size: Optional[List[int]] = None  # [w,h]
    vlm_usage: Optional[Dict[str, Any]] = None

    # Optional LLM judge (text-only) for non-obvious quality decisions
    vlm_judge_mode: Optional[str] = None  # off|auto|on
    vlm_judge_decision: Optional[str] = None  # proceed|retry|ask_user
    vlm_judge_reason_code: Optional[str] = None
    vlm_judge_explanation: Optional[str] = None
    vlm_judge_confidence: Optional[float] = None
    vlm_judge_warnings: List[str] = field(default_factory=list)
    vlm_judge_usage: Optional[Dict[str, Any]] = None

    avg_ocr_confidence: Optional[float] = None
    translation_consistent: bool = True
    lettering_ok: bool = False

    retry_count_detection: int = 0
    page_error_message: Optional[str] = None

    # 用户是否手动要求重跑这一页
    needs_manual_rerun: bool = False


@dataclass
class JobState:
    """
    整个漫画任务的状态。
    """
    job_id: str
    user_id: str
    source_file_name: str
    target_language: str

    status: JobStatus = JobStatus.CREATED
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # 轻量偏好/会话信息，不是完整对话 history
    user_preferences: Dict[str, Any] = field(default_factory=dict)

    # Job-level glossary bucket (RAG / translation memory seed).
    # Stored as a JSON-like dict to match glossary.json artifact.
    glossary: Dict[str, Any] = field(default_factory=dict)

    # Pending brain plan produced from a user message (Route A).
    # Example: {intent: "correction", ...}
    pending_user_plan: Optional[Dict[str, Any]] = None

    # page_index -> PageState
    pages: Dict[int, PageState] = field(default_factory=dict)

    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    waiting_reason: Optional[str] = None


@dataclass
class AgentAction:
    """
    决策器输出的“下一步动作”。
    """
    action_type: ActionType
    page_index: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""