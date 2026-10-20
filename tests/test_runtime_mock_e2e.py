from __future__ import annotations

import asyncio

from agentos.core.runtime import AgentRuntime
from agentos.planning.static import StaticPlanner
from agentos.schemas.enums import EventType, RuntimeState, VerificationStatus
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.task import TaskRequest
from agentos.skills.mock import MockNavigationSkill
from agentos.skills.registry import SkillRegistry
from agentos.verification.simple import SimpleFinishVerifier, SimpleSkillVerifier


def build_runtime() -> AgentRuntime:
    registry = SkillRegistry()
    registry.register(MockNavigationSkill())
    return AgentRuntime(
        planner=StaticPlanner(),
        skills=registry,
        skill_verifier=SimpleSkillVerifier(),
        finish_verifier=SimpleFinishVerifier(),
    )


def build_task_and_observation(make_trace, *, force_finish_failure: bool = False):
    task_trace = make_trace(skill_run_id="task-root")
    observation_trace = task_trace.for_skill(skill_run_id="observation-root", observation_id="obs-001")
    task = TaskRequest(
        trace=task_trace,
        instruction="Go straight for 5 m and stop.",
        metadata={"force_finish_failure": force_finish_failure},
    )
    observation = Observation(
        trace=observation_trace,
        rgb_refs={"front": "mock://front.jpg"},
        pose=Pose2D(x=0.0, y=0.0, yaw=0.0, source="mock", frame="mock_odom_xy"),
    )
    return task, observation


def test_mock_runtime_completes_full_lifecycle(make_trace):
    async def run():
        runtime = build_runtime()
        task, observation = build_task_and_observation(make_trace)
        return await runtime.run_task(task, observation)

    result = asyncio.run(run())
    assert result.state == RuntimeState.DONE
    assert len(result.skill_results) == 1
    assert result.skill_results[0].outputs["omninav_stop"] is True
    assert result.finish_verification is not None
    assert result.finish_verification.status == VerificationStatus.SUCCESS

    event_types = [event.event_type for event in result.events]
    assert EventType.SKILL_RESULT in event_types
    assert EventType.FINISH_VERIFIED in event_types
    assert EventType.TASK_DONE in event_types
    assert event_types.index(EventType.FINISH_VERIFIED) > event_types.index(EventType.SKILL_RESULT)


def test_omninav_stop_does_not_bypass_finish_verifier(make_trace):
    async def run():
        runtime = build_runtime()
        task, observation = build_task_and_observation(make_trace, force_finish_failure=True)
        return await runtime.run_task(task, observation)

    result = asyncio.run(run())
    assert result.skill_results[0].outputs["omninav_stop"] is True
    assert result.state == RuntimeState.FAILED
    assert result.finish_verification is not None
    assert result.finish_verification.status == VerificationStatus.FAILURE
    event_types = [event.event_type for event in result.events]
    assert EventType.FINISH_VERIFIED in event_types
    assert EventType.TASK_FAILED in event_types

