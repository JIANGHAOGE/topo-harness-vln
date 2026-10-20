#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import timedelta
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentos.core.runtime import AgentRuntime
from agentos.planning.static import StaticPlanner
from agentos.schemas.base import utc_now
from agentos.schemas.enums import EventType
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.task import TaskRequest
from agentos.schemas.trace import TraceContext
from agentos.skills.mock import MockNavigationSkill
from agentos.skills.registry import SkillRegistry
from agentos.verification.simple import SimpleFinishVerifier, SimpleSkillVerifier


def build_trace(*, task_id: str = "mock-task", observation_id: str = "obs-000") -> TraceContext:
    now = utc_now()
    return TraceContext(
        episode_id="mock-episode",
        task_id=task_id,
        plan_revision=0,
        skill_run_id="task-root",
        observation_id=observation_id,
        frame_timestamp=now,
        pose_timestamp=now,
        deadline=now + timedelta(seconds=30),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mock CEE-AgentOS task lifecycle.")
    parser.add_argument("--task", default="Go straight for 5 m and stop.")
    args = parser.parse_args()

    task_trace = build_trace()
    obs_trace = task_trace.for_skill(skill_run_id="observation-root", observation_id="obs-001")
    task = TaskRequest(trace=task_trace, instruction=args.task)
    observation = Observation(
        trace=obs_trace,
        rgb_refs={"front": "mock://front.jpg"},
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0, source="mock", frame="mock_odom_xy"),
        text="Mock observation for end-to-end runtime validation.",
    )

    registry = SkillRegistry()
    registry.register(MockNavigationSkill())
    runtime = AgentRuntime(
        planner=StaticPlanner(),
        skills=registry,
        skill_verifier=SimpleSkillVerifier(),
        finish_verifier=SimpleFinishVerifier(),
    )
    result = await runtime.run_task(task, observation)
    print(
        json.dumps(
            {
                "state": result.state.value,
                "plan_revision": result.plan.plan_revision if result.plan else None,
                "skill_results": [item.model_dump(mode="json") for item in result.skill_results],
                "finish_verification": None
                if result.finish_verification is None
                else result.finish_verification.model_dump(mode="json"),
                "events": [event.event_type.value for event in result.events if event.event_type != EventType.STATE_TRANSITION],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
