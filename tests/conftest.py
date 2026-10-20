from __future__ import annotations

from datetime import timedelta

import pytest

from agentos.schemas.base import utc_now
from agentos.schemas.trace import TraceContext


def _make_trace(
    *,
    episode_id: str = "ep-test",
    task_id: str = "task-test",
    plan_revision: int = 0,
    skill_run_id: str = "task-root",
    observation_id: str = "obs-test",
    deadline_offset_s: float = 30.0,
) -> TraceContext:
    now = utc_now()
    return TraceContext(
        episode_id=episode_id,
        task_id=task_id,
        plan_revision=plan_revision,
        skill_run_id=skill_run_id,
        observation_id=observation_id,
        frame_timestamp=now,
        pose_timestamp=now,
        deadline=now + timedelta(seconds=deadline_offset_s),
    )


@pytest.fixture
def make_trace():
    return _make_trace
