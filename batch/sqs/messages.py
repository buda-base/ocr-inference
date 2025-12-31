from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class TaskMessage:
    job_id: int
    job_key: str
    task_id: int
    volume_id: int
    bdrc_w_id: str
    bdrc_i_id: str
    attempt: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> TaskMessage:
        return cls(**json.loads(data))
