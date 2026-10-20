from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now


class TraceContext(SchemaModel):
    episode_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    plan_revision: int = Field(ge=0)
    skill_run_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    frame_timestamp: datetime
    pose_timestamp: datetime
    deadline: datetime
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def for_skill(
        self,
        *,
        skill_run_id: str,
        plan_revision: int | None = None,
        observation_id: str | None = None,
        frame_timestamp: datetime | None = None,
        pose_timestamp: datetime | None = None,
        deadline: datetime | None = None,
    ) -> "TraceContext":
        return self.model_copy(
            update={
                "skill_run_id": skill_run_id,
                "plan_revision": self.plan_revision if plan_revision is None else plan_revision,
                "observation_id": self.observation_id if observation_id is None else observation_id,
                "frame_timestamp": self.frame_timestamp if frame_timestamp is None else frame_timestamp,
                "pose_timestamp": self.pose_timestamp if pose_timestamp is None else pose_timestamp,
                "deadline": self.deadline if deadline is None else deadline,
                "created_at": utc_now(),
            }
        )

    def same_async_source(self, other: "TraceContext") -> bool:
        return (
            self.episode_id == other.episode_id
            and self.task_id == other.task_id
            and self.plan_revision == other.plan_revision
            and self.skill_run_id == other.skill_run_id
            and self.observation_id == other.observation_id
        )

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or utc_now()) > self.deadline

