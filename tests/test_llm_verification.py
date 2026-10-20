from __future__ import annotations

import asyncio

from agentos.schemas.enums import SkillStatus, VerificationAction, VerificationStatus
from agentos.schemas.llm import LLMResponse
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan, PlanStep
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.verification.llm import LLMFinishVerifier, LLMSkillVerifier


class _ContradictoryFinishProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, response_schema=None):
        del messages, response_schema
        self.calls += 1
        decision = {
            "disposition": "finish",
            "summary": "The bed is visible and the original task is satisfied.",
            "reasons": [],
            "observed_evidence": ["A bed is clearly visible in the bedroom."],
        }
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


def test_finish_verifier_uses_one_independent_model_judgment(make_trace):
    trace = make_trace(observation_id="bedroom-observation")
    provider = _ContradictoryFinishProvider()
    verifier = LLMFinishVerifier(provider)
    task = TaskRequest(trace=trace, instruction="go to bedroom to find a bed")
    plan = Plan(
        task_id=trace.task_id,
        plan_revision=0,
        steps=[
            PlanStep(
                step_id="bedroom",
                skill_name="NavigationSkill",
                command="Navigate to the bedroom.",
            )
        ],
    )
    result = SkillResult(
        trace=trace,
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="Reached the bedroom node.",
    )
    observation = Observation(trace=trace)

    verification = asyncio.run(
        verifier.verify_finish(task, plan, [result], observation)
    )

    assert provider.calls == 1
    assert verification.status == VerificationStatus.SUCCESS
    assert verification.action == VerificationAction.FINISH


def test_finish_verifier_trace_uses_current_evidence_observation_and_plan(make_trace):
    root_trace = make_trace(observation_id="initial-observation", plan_revision=0)
    evidence_trace = make_trace(observation_id="current-observation", plan_revision=1)
    provider = _ContradictoryFinishProvider()
    verifier = LLMFinishVerifier(provider)
    task = TaskRequest(trace=root_trace, instruction="find a bed")
    plan = Plan(
        task_id=root_trace.task_id,
        plan_revision=2,
        steps=[
            PlanStep(
                step_id="find-bed",
                skill_name="NavigationSkill",
                command="Find a bed.",
            )
        ],
    )
    result = SkillResult(
        trace=root_trace,
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="A bed was reached.",
    )

    verification = asyncio.run(
        verifier.verify_finish(task, plan, [result], Observation(trace=evidence_trace))
    )

    assert verification.trace.task_id == root_trace.task_id
    assert verification.trace.observation_id == "current-observation"
    assert verification.trace.plan_revision == 2
    assert verification.trace.frame_timestamp == evidence_trace.frame_timestamp


class _LocalFailureProvider:
    async def complete(self, messages, *, response_schema=None):
        del messages, response_schema
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json={
                "outcome": "not_achieved",
                "summary": "The current local route is blocked.",
                "reasons": ["No traversable local continuation is visible."],
                "observed_evidence": ["The current passage is blocked."],
            },
        )


def test_skill_failure_returns_control_to_main_agent(make_trace):
    trace = make_trace(observation_id="blocked-observation")
    verifier = LLMSkillVerifier(_LocalFailureProvider())
    request = SkillRequest(
        trace=trace,
        skill_name="NavigationSkill",
        command="Navigate through the passage.",
        timeout_s=30.0,
    )
    result = SkillResult(
        trace=trace,
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="The local rollout stopped at a blocked passage.",
    )

    verification = asyncio.run(
        verifier.verify_skill(request, result, Observation(trace=trace))
    )

    assert verification.status == VerificationStatus.RETRY
    assert verification.action == VerificationAction.REPLAN


class _PrematureObjectSuccessProvider:
    async def complete(self, messages, *, response_schema=None):
        del messages, response_schema
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json={
                "outcome": "achieved",
                "summary": "The requested object is visible.",
                "reasons": [],
                "observed_evidence": ["The object appears in the current view."],
            },
        )


def test_object_goal_affordance_cannot_bypass_target_terminal_contract(make_trace):
    trace = make_trace(observation_id="object-observation")
    verifier = LLMSkillVerifier(_PrematureObjectSuccessProvider())
    request = SkillRequest(
        trace=trace,
        skill_name="NavigationSkill",
        command="Find and approach the bed.",
        inputs={
            "delegation": {
                "task_specification": {"task_type": "object_goal"},
            }
        },
    )
    result = SkillResult(
        trace=trace,
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="Reached an intermediate doorway affordance.",
        outputs={
            "anchor_role": "affordance",
            "semantic_completion": {
                "task_type": "object_goal",
                "required_terminal_role": "target",
                "target_grounded": False,
                "target_executed": False,
                "target_reached": False,
                "satisfied": False,
            },
        },
    )

    verification = asyncio.run(
        verifier.verify_skill(request, result, Observation(trace=trace))
    )

    assert verification.status == VerificationStatus.RETRY
    assert verification.action == VerificationAction.REPLAN
    assert "Target" in verification.summary


class _TargetArtifactProvider:
    def __init__(self) -> None:
        self.messages = None

    async def complete(self, messages, *, response_schema=None):
        del response_schema
        self.messages = messages
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json={
                "outcome": "achieved",
                "summary": "The grounded target was reached.",
                "reasons": ["The source artifact and execution facts agree."],
                "observed_evidence": ["The target artifact shows the requested object."],
            },
        )


class _TargetConflictReviewProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.messages = []

    async def complete(self, messages, *, response_schema=None):
        del response_schema
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            decision = {
                "outcome": "not_achieved",
                "summary": "The target is no longer visible in the final camera frame.",
                "reasons": ["The final view is too close to recognize the object."],
                "observed_evidence": ["The post-skill view does not show the full object."],
            }
        else:
            decision = {
                "outcome": "achieved",
                "summary": "The source-bound target and reach evidence confirm close approach.",
                "reasons": ["The target artifact is correct and target-role execution reached it."],
                "observed_evidence": ["The source artifact clearly shows the requested object."],
            }
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json=decision,
        )


def test_skill_verifier_reviews_target_evidence_conflict_before_replanning(
    make_trace,
    tmp_path,
):
    trace = make_trace(observation_id="post-close-approach")
    artifact = tmp_path / "fridge-target.jpg"
    artifact.write_bytes(b"fridge-target")
    provider = _TargetConflictReviewProvider()
    verifier = LLMSkillVerifier(provider)
    request = SkillRequest(
        trace=trace,
        skill_name="NavigationSkill",
        command="Find and approach the white fridge.",
    )
    result = SkillResult(
        trace=trace,
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="The white-fridge Target pixel was reached.",
        evidence=[
            Evidence(
                kind="slow_brain_pixel_goal",
                summary="Current-frame white-fridge Target grounding.",
                ref=str(artifact),
            )
        ],
        outputs={
            "anchor_role": "target",
            "semantic_completion": {
                "task_type": "object_goal",
                "target_label": "white fridge",
                "target_view": "left",
                "target_confidence": 0.95,
                "target_confidence_threshold": 0.90,
                "target_grounded": True,
                "target_executed": True,
                "target_reached": True,
                "satisfied": True,
            },
        },
    )

    verification = asyncio.run(
        verifier.verify_skill(request, result, Observation(trace=trace))
    )

    assert provider.calls == 2
    assert verification.status == VerificationStatus.SUCCESS
    assert verification.metadata["conflict_review"] is True
    review_text = provider.messages[1][1].content[0]["text"]
    assert "Conflict review" in review_text


def test_skill_verifier_receives_target_artifact_and_recent_execution_keyframes(
    make_trace,
    tmp_path,
):
    trace = make_trace(observation_id="post-target-observation")
    artifact = tmp_path / "target.jpg"
    artifact.write_bytes(b"target-image")
    keyframe_1 = tmp_path / "keyframe-1.jpg"
    keyframe_1.write_bytes(b"keyframe-1")
    keyframe_2 = tmp_path / "keyframe-2.jpg"
    keyframe_2.write_bytes(b"keyframe-2")
    provider = _TargetArtifactProvider()
    verifier = LLMSkillVerifier(provider)
    request = SkillRequest(
        trace=trace,
        skill_name="NavigationSkill",
        command="Find and approach the refrigerator.",
    )
    result = SkillResult(
        trace=trace,
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="Target pixel reached.",
        evidence=[
            Evidence(
                kind="slow_brain_pixel_goal",
                summary="Current-frame target grounding.",
                ref=str(artifact),
            ),
            Evidence(
                kind="trajectory_keyframe",
                summary="Recent execution keyframe 1.",
                ref=str(keyframe_1),
            ),
            Evidence(
                kind="trajectory_keyframe",
                summary="Recent execution keyframe 2.",
                ref=str(keyframe_2),
            ),
        ],
        outputs={
            "anchor_role": "target",
            "semantic_completion": {
                "task_type": "object_goal",
                "target_grounded": True,
                "target_executed": True,
                "target_reached": True,
                "satisfied": True,
            },
        },
    )

    verification = asyncio.run(
        verifier.verify_skill(request, result, Observation(trace=trace))
    )

    assert verification.status == VerificationStatus.SUCCESS
    assert provider.messages is not None
    content = provider.messages[-1].content
    assert isinstance(content, list)
    assert any(
        item.get("type") == "text"
        and item.get("text") == "TARGET_GROUNDING_SOURCE_ARTIFACT"
        for item in content
    )
    assert any(
        item.get("type") == "text"
        and item.get("text") == "RECENT_EXECUTION_KEYFRAME_2"
        for item in content
    )
    assert sum(item.get("type") == "image_url" for item in content) == 3


class _FinishTargetArtifactProvider:
    def __init__(self) -> None:
        self.messages = None

    async def complete(self, messages, *, response_schema=None):
        del response_schema
        self.messages = messages
        return LLMResponse(
            model="fake-qwen",
            content="{}",
            raw={},
            parsed_json={
                "disposition": "finish",
                "summary": "The terminal target artifact proves the fridge was found.",
                "reasons": ["The grounded fridge Target was executed and reached."],
                "observed_evidence": ["The source artifact clearly shows the fridge."],
            },
        )


def test_finish_verifier_receives_only_bound_satisfied_target_artifact(
    make_trace,
    tmp_path,
):
    trace = make_trace(observation_id="post-close-approach")
    artifact = tmp_path / "fridge-target.jpg"
    artifact.write_bytes(b"fridge-target")
    provider = _FinishTargetArtifactProvider()
    verifier = LLMFinishVerifier(provider)
    task = TaskRequest(trace=trace, instruction="go to office to find a fridge")
    plan = Plan(
        task_id=trace.task_id,
        plan_revision=1,
        steps=[
            PlanStep(
                step_id="find-fridge",
                skill_name="NavigationSkill",
                command="Find the fridge in the office.",
            )
        ],
    )
    result = SkillResult(
        trace=trace,
        skill_name="NavigationSkill",
        status=SkillStatus.SUCCESS,
        summary="The fridge Target pixel was reached.",
        evidence=[
            Evidence(
                kind="slow_brain_pixel_goal",
                summary="Current-frame fridge Target grounding.",
                ref=str(artifact),
            )
        ],
        outputs={
            "semantic_completion": {
                "task_type": "object_goal",
                "target_label": "white refrigerator",
                "target_view": "front",
                "target_confidence": 0.95,
                "target_confidence_threshold": 0.90,
                "target_grounded": True,
                "target_executed": True,
                "target_reached": True,
                "source_observation_id": "source-fridge-observation",
                "post_observation_id": "post-close-approach",
                "satisfied": True,
            }
        },
    )

    verification = asyncio.run(
        verifier.verify_finish(task, plan, [result], Observation(trace=trace))
    )

    assert verification.status == VerificationStatus.SUCCESS
    assert provider.messages is not None
    content = provider.messages[-1].content
    assert isinstance(content, list)
    assert any(
        item.get("type") == "text"
        and item.get("text") == "TERMINAL_TARGET_GROUNDING_SOURCE_ARTIFACT"
        for item in content
    )
    assert sum(item.get("type") == "image_url" for item in content) == 1
    prompt = next(item["text"] for item in content if item.get("type") == "text")
    assert "target_reached" in prompt
