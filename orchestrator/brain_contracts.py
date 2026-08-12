from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class BrainIntent(str, Enum):
    CHAT = "chat"
    CREATE_TRANSLATION_JOB = "create_translation_job"
    RESUME_JOB = "resume_job"
    QUERY_PROGRESS = "query_progress"
    REVISE_TRANSLATION = "revise_translation"
    RERUN_PAGE = "rerun_page"
    EXPORT_RESULT = "export_result"
    IMPORT_RAG_BUNDLE = "import_rag_bundle"
    UNKNOWN = "unknown"


class ResponseStyle(str, Enum):
    CHAT = "chat"
    TASK = "task"
    STATUS = "status"


@dataclass
class BrainPlan:
    intent: BrainIntent = BrainIntent.UNKNOWN
    confidence: Optional[float] = None
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    job_id: Optional[str] = None
    response_style: ResponseStyle = ResponseStyle.CHAT
    payload: Dict[str, Any] = field(default_factory=dict)
    strategy_suggestions: Dict[str, Any] = field(default_factory=dict)
    skill_param_suggestions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    raw_user_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["intent"] = self.intent.value
        data["response_style"] = self.response_style.value
        return data
