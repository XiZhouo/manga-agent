from __future__ import annotations

from typing import Dict

from .state import JobState


class StateStore:
    """
    抽象状态存储接口。

    现在先给内存版，后续你可以扩展：
    - SQLiteStateStore
    - PostgresStateStore
    - RedisStateStore
    """

    def create_job(self, job: JobState) -> None:
        raise NotImplementedError

    def get_job(self, job_id: str) -> JobState:
        raise NotImplementedError

    def save_job(self, job: JobState) -> None:
        raise NotImplementedError


class InMemoryStateStore(StateStore):
    """
    最简单的内存实现。
    适合本地调试和单进程 demo。
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, JobState] = {}

    def create_job(self, job: JobState) -> None:
        self._jobs[job.job_id] = job

    def get_job(self, job_id: str) -> JobState:
        if job_id not in self._jobs:
            raise KeyError(f"Job not found: {job_id}")
        return self._jobs[job_id]

    def save_job(self, job: JobState) -> None:
        self._jobs[job.job_id] = job