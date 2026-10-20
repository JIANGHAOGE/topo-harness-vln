from __future__ import annotations

import asyncio

from agentos.core.events import InMemoryEventBus
from agentos.core.runtime import AgentRuntime
from agentos.harness.skill_runner import IsolatedSkillRunner
from agentos.planning.static import StaticPlanner
from agentos.schemas.enums import EventType, SkillStatus, VerificationLevel, VerificationStatus
from agentos.schemas.observation import Observation
from agentos.schemas.skill import SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult
from agentos.skills.base import CancellationToken
from agentos.skills.registry import SkillRegistry
from agentos.verification.simple import SimpleRuntimeVerifier, SimpleSkillVerifier


class _VerboseSkill:
    name = "VerboseSkill"

    async def run(self, request, observation, context):
        del observation
        await context.emit_progress(request, "local observation acquired", frame_count=12)
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary="raw local execution result",
            compressed_history="Reached the delegated doorway subgoal.",
            outputs={"raw_frames": [f"frame-{index}" for index in range(12)]},
        )


class _HandoffSkill(_VerboseSkill):
    name = "HandoffSkill"

    @staticmethod
    def main_agent_handoff(result):
        return {
            "topology_goal_search": result.outputs["topology_goal_search"],
            "post_observation_id": result.outputs["post_observation_id"],
        }

    async def run(self, request, observation, context):
        del observation, context
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary="raw local execution result",
            compressed_history="Region search attempt completed.",
            outputs={
                "topology_goal_search": {
                    "topology_node_id": "backbone-001",
                    "status": "no_target_observed",
                },
                "post_observation_id": "obs-002",
                "raw_frames": ["frame-001", "frame-002"],
            },
        )


def test_skill_runner_isolates_local_history_from_main_agent(make_trace):
    registry = SkillRegistry()
    registry.register(_VerboseSkill())
    runner = IsolatedSkillRunner(registry)
    event_bus = InMemoryEventBus()
    trace = make_trace(skill_run_id="isolated-skill", observation_id="obs-001")
    request = SkillRequest(
        trace=trace,
        skill_name="VerboseSkill",
        command="Navigate to the doorway.",
        success_conditions=["doorway reached"],
    )
    observation = Observation(trace=trace, rgb_refs={"front": "mock://front.jpg"})

    outcome = asyncio.run(
        runner.run(
            request,
            observation,
            event_bus=event_bus,
            cancellation=CancellationToken(),
            attempt=0,
        )
    )

    assert outcome.raw_result.outputs["raw_frames"][-1] == "frame-11"
    assert outcome.main_agent_result.outputs == {}
    assert outcome.main_agent_result.summary == "Reached the delegated doorway subgoal."
    assert outcome.session.progress_summaries == ["local observation acquired"]
    event_types = [event.event_type for event in event_bus.snapshot()]
    assert EventType.SKILL_CONTEXT_OPENED in event_types
    assert EventType.SKILL_PROGRESS in event_types
    assert EventType.SKILL_CONTEXT_CLOSED in event_types


def test_skill_runner_exposes_only_declared_main_agent_handoff(make_trace):
    registry = SkillRegistry()
    registry.register(_HandoffSkill())
    runner = IsolatedSkillRunner(registry)
    trace = make_trace(skill_run_id="handoff-skill", observation_id="obs-001")
    request = SkillRequest(
        trace=trace,
        skill_name="HandoffSkill",
        command="Search the active region.",
        success_conditions=["region search attempt completed"],
    )
    observation = Observation(trace=trace, rgb_refs={"front": "mock://front.jpg"})

    outcome = asyncio.run(
        runner.run(
            request,
            observation,
            event_bus=InMemoryEventBus(),
            cancellation=CancellationToken(),
            attempt=0,
        )
    )

    assert outcome.raw_result.outputs["raw_frames"] == ["frame-001", "frame-002"]
    assert outcome.main_agent_result.outputs == {
        "topology_goal_search": {
            "topology_node_id": "backbone-001",
            "status": "no_target_observed",
        },
        "post_observation_id": "obs-002",
    }


class _CapturingFinishVerifier:
    def __init__(self) -> None:
        self.skill_results = []

    async def verify_finish(self, task, plan, skill_results, observation):
        del plan, observation
        self.skill_results = skill_results
        return VerificationResult(
            trace=task.trace,
            level=VerificationLevel.FINISH,
            status=VerificationStatus.SUCCESS,
            summary="Global goal verified from compressed skill evidence.",
        )


def test_runtime_keeps_raw_audit_result_out_of_main_agent_boundary(make_trace):
    registry = SkillRegistry()
    registry.register(_VerboseSkill())
    finish_verifier = _CapturingFinishVerifier()
    runtime = AgentRuntime(
        planner=StaticPlanner(skill_name="VerboseSkill"),
        skills=registry,
        runtime_verifier=SimpleRuntimeVerifier(),
        skill_verifier=SimpleSkillVerifier(),
        finish_verifier=finish_verifier,
    )
    trace = make_trace(skill_run_id="task-root", observation_id="obs-001")
    task = TaskRequest(trace=trace, instruction="Navigate to the doorway.")
    observation = Observation(trace=trace, rgb_refs={"front": "mock://front.jpg"})

    result = asyncio.run(runtime.run_task(task, observation))

    assert result.skill_results[0].outputs["raw_frames"][-1] == "frame-11"
    assert result.main_agent_results[0].outputs == {}
    assert finish_verifier.skill_results[0].outputs == {}
    assert finish_verifier.skill_results[0].summary == "Reached the delegated doorway subgoal."
