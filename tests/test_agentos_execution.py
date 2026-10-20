from __future__ import annotations

import asyncio

from agentos.core.events import InMemoryEventBus
from agentos.core.runtime import AgentRuntime
from agentos.planning.static import StaticPlanner
from agentos.providers.vision import (
    audit_observation_views,
    audited_observation_content,
    observation_content,
)
from agentos.schemas.enums import RuntimeState, SkillStatus, VerificationLevel, VerificationStatus
from agentos.schemas.llm import LLMResponse
from agentos.schemas.observation import Observation, Pose2D
from agentos.schemas.planning import Plan, PlanStep
from agentos.schemas.skill import SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult
from agentos.skills.base import CancellationToken, SkillContext
from agentos.skills.mock import MockNavigationSkill
from agentos.skills.navigation.persistent import OmniNavGuidance, PersistentOmniNavSkill
from agentos.skills.registry import SkillRegistry


def _observation(make_trace, observation_id: str, *, x: float = 0.0) -> Observation:
    trace = make_trace(observation_id=observation_id).for_skill(
        skill_run_id=f"observation-{observation_id}",
        observation_id=observation_id,
    )
    return Observation(
        trace=trace,
        rgb_refs={"front": "mock://front.jpg"},
        pose=Pose2D(x=x, y=0.0, yaw=0.0, source="test", frame="odom"),
    )


class _RecordingRuntimeVerifier:
    def __init__(self) -> None:
        self.observation_ids: list[str] = []

    async def verify_runtime(self, request, result, observation):
        del request
        self.observation_ids.append(observation.trace.observation_id)
        return VerificationResult(
            trace=result.trace,
            level=VerificationLevel.RUNTIME,
            status=VerificationStatus.SUCCESS,
            summary="fresh runtime observation",
        )


class _RecordingSkillVerifier:
    def __init__(self) -> None:
        self.observation_ids: list[str] = []

    async def verify_skill(self, request, result, observation):
        del request
        self.observation_ids.append(observation.trace.observation_id)
        return VerificationResult(
            trace=result.trace,
            level=VerificationLevel.SKILL,
            status=VerificationStatus.SUCCESS,
            summary="fresh skill observation",
        )


class _RecordingFinishVerifier:
    def __init__(self) -> None:
        self.observation_ids: list[str] = []

    async def verify_finish(self, task, plan, skill_results, observation):
        del plan, skill_results
        self.observation_ids.append(observation.trace.observation_id)
        return VerificationResult(
            trace=task.trace,
            level=VerificationLevel.FINISH,
            status=VerificationStatus.SUCCESS,
            summary="fresh finish observation",
        )


def test_runtime_verifiers_receive_post_skill_observation(make_trace):
    registry = SkillRegistry()
    registry.register(MockNavigationSkill())
    runtime_verifier = _RecordingRuntimeVerifier()
    skill_verifier = _RecordingSkillVerifier()
    finish_verifier = _RecordingFinishVerifier()

    async def observe_after_skill(trace):
        return _observation(make_trace, "obs-post", x=1.0).model_copy(
            update={"trace": trace.model_copy(update={"observation_id": "obs-post"})}
        )

    runtime = AgentRuntime(
        planner=StaticPlanner(),
        skills=registry,
        runtime_verifier=runtime_verifier,
        skill_verifier=skill_verifier,
        finish_verifier=finish_verifier,
        observation_provider=observe_after_skill,
    )
    task = TaskRequest(trace=make_trace(), instruction="Go to the doorway.")
    result = asyncio.run(runtime.run_task(task, _observation(make_trace, "obs-initial")))

    assert result.state == RuntimeState.DONE
    assert runtime_verifier.observation_ids == ["obs-post"]
    assert skill_verifier.observation_ids == ["obs-post"]
    assert finish_verifier.observation_ids == ["obs-post"]


class _OrderedDelegationPlanner:
    def __init__(self) -> None:
        self.replan_calls = 0

    async def plan(self, task, observation):
        del observation
        return self._plan(task, revision=0, delegated_index=0)

    async def replan(
        self,
        task,
        observation,
        previous_plan,
        skill_results,
        *,
        reason,
        next_revision,
    ):
        del observation, previous_plan, skill_results, reason
        self.replan_calls += 1
        return self._plan(task, revision=next_revision, delegated_index=1)

    @staticmethod
    def _plan(task, *, revision: int, delegated_index: int) -> Plan:
        subtasks = [
            {
                "skill_name": "NavigationSkill",
                "task_type": "point_goal",
                "instruction": "Reach the bedroom region.",
            },
            {
                "skill_name": "NavigationSkill",
                "task_type": "object_goal",
                "instruction": "Find the bed in the bedroom.",
            },
        ]
        delegated = subtasks[delegated_index]
        return Plan(
            task_id=task.trace.task_id,
            plan_revision=revision,
            rationale_summary="Execute the accepted ordered Skill plan.",
            steps=[
                PlanStep(
                    step_id=f"delegation-{delegated_index}",
                    skill_name=delegated["skill_name"],
                    command=delegated["instruction"],
                )
            ],
            metadata={
                "planned_skill_subtasks": subtasks,
                "delegated_subtask_index": delegated_index,
            },
        )


def test_runtime_advances_pending_agentos_subtask_before_finish(make_trace):
    registry = SkillRegistry()
    registry.register(MockNavigationSkill())
    planner = _OrderedDelegationPlanner()
    finish_verifier = _RecordingFinishVerifier()
    runtime = AgentRuntime(
        planner=planner,
        replanner=planner,
        skills=registry,
        runtime_verifier=_RecordingRuntimeVerifier(),
        skill_verifier=_RecordingSkillVerifier(),
        finish_verifier=finish_verifier,
        max_plan_revisions=0,
    )
    task = TaskRequest(trace=make_trace(), instruction="Go to the bedroom and find a bed.")

    result = asyncio.run(runtime.run_task(task, _observation(make_trace, "obs-initial")))

    assert result.state == RuntimeState.DONE
    assert len(result.skill_results) == 2
    assert [item.trace.plan_revision for item in result.skill_results] == [0, 1]
    assert planner.replan_calls == 1
    assert len(finish_verifier.observation_ids) == 1
    assert result.plan is not None
    assert result.plan.metadata["delegated_subtask_index"] == 1


class _SuccessfulNamedSkill:
    def __init__(self, name: str) -> None:
        self.name = name

    async def run(self, request, observation, context):
        del observation, context
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary=f"{self.name} completed.",
        )


class _MappingThenNavigationPlanner:
    def __init__(self) -> None:
        self.replan_calls = 0

    async def plan(self, task, observation):
        del observation
        return self._plan(task, revision=0, skill_name="MappingSkill")

    async def replan(
        self,
        task,
        observation,
        previous_plan,
        skill_results,
        *,
        reason,
        next_revision,
    ):
        del observation, previous_plan, skill_results
        self.replan_calls += 1
        assert "MappingSkill succeeded" in reason
        return self._plan(task, revision=next_revision, skill_name="NavigationSkill")

    @staticmethod
    def _plan(task, *, revision: int, skill_name: str) -> Plan:
        command = (
            "Acquire current map evidence."
            if skill_name == "MappingSkill"
            else "Navigate to the model-selected region."
        )
        return Plan(
            task_id=task.trace.task_id,
            plan_revision=revision,
            rationale_summary="Delegate one atomic AgentOS Skill.",
            steps=[
                PlanStep(
                    step_id=f"delegation-{revision}",
                    skill_name=skill_name,
                    command=command,
                )
            ],
            metadata={
                "delegated_skill": skill_name,
                "planned_skill_subtasks": [
                    {
                        "skill_name": skill_name,
                        "task_type": "object_goal",
                        "instruction": command,
                    }
                ],
                "delegated_subtask_index": 0,
            },
        )


def test_runtime_returns_mapping_result_to_main_before_finish(make_trace):
    registry = SkillRegistry()
    registry.register(_SuccessfulNamedSkill("MappingSkill"))
    registry.register(_SuccessfulNamedSkill("NavigationSkill"))
    planner = _MappingThenNavigationPlanner()
    finish_verifier = _RecordingFinishVerifier()
    runtime = AgentRuntime(
        planner=planner,
        replanner=planner,
        skills=registry,
        runtime_verifier=_RecordingRuntimeVerifier(),
        skill_verifier=_RecordingSkillVerifier(),
        finish_verifier=finish_verifier,
        max_plan_revisions=0,
    )
    task = TaskRequest(trace=make_trace(), instruction="Go to the office and find a fridge.")

    result = asyncio.run(runtime.run_task(task, _observation(make_trace, "obs-initial")))

    assert result.state == RuntimeState.DONE
    assert [item.skill_name for item in result.skill_results] == [
        "MappingSkill",
        "NavigationSkill",
    ]
    assert planner.replan_calls == 1
    assert len(finish_verifier.observation_ids) == 1


class _FakeOneStepOmniNav:
    def __init__(self, *, stop_after: int | None, distance_m: float) -> None:
        self.stop_after = stop_after
        self.distance_m = distance_m
        self.calls = 0
        self.instructions: list[str] = []

    async def run(self, request, observation, context):
        del observation, context
        self.calls += 1
        self.instructions.append(str(request.inputs.get("instruction") or request.command))
        stop = self.stop_after is not None and self.calls >= self.stop_after
        return SkillResult(
            trace=request.trace,
            skill_name="OmniNavSkill",
            status=SkillStatus.SUCCESS,
            summary="one bounded OmniNav action",
            outputs={
                "omninav_stop": stop,
                "execution": {
                    "action": "hold" if stop else "forward",
                    "distance_m": 0.0 if stop else self.distance_m,
                    "collided": False,
                },
            },
        )


def _persistent_request(make_trace) -> SkillRequest:
    return SkillRequest(
        trace=make_trace(skill_run_id="persistent-run"),
        skill_name="NavigationSkill",
        command="Navigate towards the doorway.",
        timeout_s=30.0,
    )


def _skill_context() -> SkillContext:
    return SkillContext(event_bus=InMemoryEventBus(), cancellation=CancellationToken())


def test_persistent_omninav_stop_is_local_evidence(make_trace):
    inner = _FakeOneStepOmniNav(stop_after=3, distance_m=0.4)
    observations = iter(
        [_observation(make_trace, "obs-1", x=0.4), _observation(make_trace, "obs-2", x=0.8),
         _observation(make_trace, "obs-3", x=0.8)]
    )

    async def observe(trace):
        del trace
        return next(observations)

    skill = PersistentOmniNavSkill(inner=inner, observe=observe, max_steps=10)
    result = asyncio.run(
        skill.run(_persistent_request(make_trace), _observation(make_trace, "obs-0"), _skill_context())
    )

    assert result.status == SkillStatus.SUCCESS
    assert result.outputs["omninav_stop"] is True
    assert result.outputs["steps"] == 3
    assert result.outputs["distance_m"] == 0.8
    assert result.stop_reason == "omninav_local_stop"
    assert "Main Agent verification" in result.summary


def test_persistent_omninav_stagnation_is_bounded_failure(make_trace):
    inner = _FakeOneStepOmniNav(stop_after=None, distance_m=0.0)
    counter = 0

    async def observe(trace):
        nonlocal counter
        del trace
        counter += 1
        return _observation(make_trace, f"obs-{counter}")

    skill = PersistentOmniNavSkill(
        inner=inner,
        observe=observe,
        max_steps=20,
        no_progress_window=3,
        minimum_window_progress_m=0.5,
    )
    result = asyncio.run(
        skill.run(_persistent_request(make_trace), _observation(make_trace, "obs-0"), _skill_context())
    )

    assert result.status == SkillStatus.RETRYABLE_FAILURE
    assert result.outputs["steps"] == 3
    assert result.stop_reason == "stagnation_guard"


def test_persistent_omninav_consumes_cached_guidance_while_slow_refreshes(
    make_trace,
):
    inner = _FakeOneStepOmniNav(stop_after=None, distance_m=0.2)
    observations = iter(
        [
            _observation(make_trace, "obs-1", x=0.2),
            _observation(make_trace, "obs-2", x=0.4),
        ]
    )

    async def observe(trace):
        del trace
        return next(observations)

    initial_observation = _observation(make_trace, "obs-0")
    initial = OmniNavGuidance(
        instruction="Navigate toward the visible doorway.",
        source_observation=initial_observation,
    )

    async def refresh(source_observation):
        await asyncio.sleep(0.01)
        return OmniNavGuidance(
            instruction="Hold for the visible bed target.",
            source_observation=source_observation,
            terminal_reason="target_ready",
        )

    skill = PersistentOmniNavSkill(inner=inner, observe=observe, max_steps=10)
    outcome = asyncio.run(
        skill.run_guided(
            _persistent_request(make_trace),
            initial_observation,
            _skill_context(),
            initial_guidance=initial,
            refresh_guidance=refresh,
            slow_update_interval_steps=1,
            max_cached_steps=2,
        )
    )

    assert inner.instructions == [
        "Navigate toward the visible doorway.",
        "Navigate toward the visible doorway.",
    ]
    assert outcome.guidance.terminal_reason == "target_ready"
    assert outcome.result.stop_reason == "target_ready"
    assert outcome.result.outputs["slow_fast_scheduler"]["mode"] == "asynchronous_cached_guidance"


def test_multiview_observation_order_is_left_front_right(tmp_path, make_trace):
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"{view}.jpg"
        path.write_bytes(view.encode("ascii"))
        refs[view] = str(path)
    observation = _observation(make_trace, "obs-views").model_copy(update={"rgb_refs": refs})

    content = observation_content("Inspect the scene.", observation)
    labels = [item["text"] for item in content if item["type"] == "text" and item["text"].startswith("ROBOT_")]

    assert labels == ["ROBOT_LEFT_CAMERA", "ROBOT_FRONT_CAMERA", "ROBOT_RIGHT_CAMERA"]


def test_tri_view_audit_is_single_frame_bound_and_cached(tmp_path, make_trace):
    refs = {}
    for view in ("left", "front", "right"):
        path = tmp_path / f"audit-{view}.jpg"
        path.write_bytes(view.encode("ascii"))
        refs[view] = str(path)
    observation = _observation(make_trace, "obs-audit").model_copy(update={"rgb_refs": refs})

    class Provider:
        def __init__(self):
            self.calls = []

        async def complete(self, messages, *, response_schema=None):
            self.calls.append(messages)
            system = str(messages[0].content)
            view = next(
                item for item in ("left", "front", "right") if f"robot-{item.upper()}" in system
            )
            return LLMResponse(
                model="fake-qwen",
                content="{}",
                raw={},
                parsed_json={
                    "view": view,
                    "visible_objects": [f"{view} object"],
                    "navigable_semantic_anchors": [f"{view} doorway"],
                    "target_relevant_evidence": [],
                    "scene_summary": f"The {view} camera scene.",
                },
            )

    provider = Provider()

    first = asyncio.run(audit_observation_views(provider, observation))
    second = asyncio.run(audit_observation_views(provider, observation))
    content = audited_observation_content("Inspect.", observation, second)

    assert first == second
    assert len(provider.calls) == 3
    assert [first.left.view, first.front.view, first.right.view] == ["left", "front", "right"]
    assert all(item["type"] != "image_url" for item in content)
    assert "physical robot camera" in content[0]["text"]
