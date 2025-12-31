from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


class JobStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"


@dataclass
class Task:
    id: int
    job_id: int
    attempts: int
    max_attempts: int


@dataclass
class JobType:
    id: int
    name: str
    model_name: str
    encoding: str = "unicode"
    line_mode: str = "line"
    k_factor: float = 2.5
    bbox_tolerance: float = 4.0
    merge_lines: bool = False
    use_tps: bool = False
    artifact_granularity: str = "standard"
    description: str | None = None


@dataclass
class Volume:
    id: int
    bdrc_w_id: str
    bdrc_i_id: str

    @property
    def volume_id(self) -> str:
        return f"{self.bdrc_w_id}-{self.bdrc_i_id}"


@dataclass
class Job:
    id: int
    job_key: str
    type_id: int
    status: JobStatus = JobStatus.CREATED
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    total_tasks: int = 0
    done_tasks: int = 0
    failed_tasks: int = 0
    last_progress_at: datetime | None = None
