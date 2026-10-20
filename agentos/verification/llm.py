from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationError, model_validator

from agentos.providers.base import LLMProvider
from agentos.providers.vision import (
    audit_observation_views,
    audited_observation_content,
    image_ref_content,
)
from agentos.schemas.base import SchemaModel
from agentos.schemas.enums import (
    SkillStatus,
    VerificationAction,
    VerificationLevel,
    VerificationStatus,
)
from agentos.schemas.llm import ChatMessage
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult


class _SkillDecision(SchemaModel):
    outcome: Literal["achieved", "not_achieved"]
    summary: str = Field(min_length=1)
    reasons: list[str] = Field(default_factory=list)
    observed_evidence: list[str] = Field(default_factory=list)

class _FinishDecision(SchemaModel):
    disposition: Literal["finish", "replan", "fail"]
    summary: str = Field(min_length=1)
    reasons: list[str] = Field(default_factory=list)
    observed_evidence: list[str] = Field(min_length=1)


class ProgressRuntimeVerifier:
    """Deterministic first-stage verification for stale, stuck, and failed runs."""

    async def verify_runtime(
        self,
        request: SkillRequest,
        result: SkillResult,
        observation: Observation,
    ) -> VerificationResult:
        del request
        execution = (result.outputs or {}).get("result") or {}
        telemetry = (result.outputs or {}).get("execution_telemetry") or {}
        termination_reason = str(
            telemetry.get("termination_reason")
            or execution.get("termination_reason")
            or ""
        )
        blocked_reasons = {
            "blocked",
            "cancelled",
            "no_affordance_pixel",
            "no_visible_target_pixel",
            "path_invalid",
            "repeated_zero_progress",
            "stalled",
            "timeout",
        }
        mechanically_blocked = bool(telemetry.get("stalled")) or (
            telemetry.get("state") in {"blocked", "stalled", "cancelled", "failed"}
        ) or bool(execution.get("stalled")) or (
            termination_reason in blocked_reasons
        )
        moved_distance = float(telemetry.get("progress_m") or 0.0)
        motion_attempted = bool(telemetry.get("motion_attempted"))
        remaining_distance = float(telemetry.get("remaining_distance_m") or 0.0)
        no_progress = (
            motion_attempted
            and moved_distance < 0.05
            and remaining_distance > 0.3
        )
        if result.status == SkillStatus.SUCCESS and not mechanically_blocked and not no_progress:
            return self._result(
                result,
                VerificationStatus.SUCCESS,
                VerificationAction.CONTINUE,
                "Runtime execution remained healthy and produced a fresh post-skill observation.",
                observation,
            )
        if result.status in {SkillStatus.SUCCESS, SkillStatus.RETRYABLE_FAILURE}:
            summary = result.summary
            if mechanically_blocked:
                summary = f"Runtime execution is blocked: {termination_reason or 'stalled'}."
            elif no_progress:
                summary = "Runtime execution attempted motion without measurable progress."
            return self._result(
                result,
                VerificationStatus.RETRY,
                VerificationAction.REPLAN,
                summary,
                observation,
            )
        return self._result(
            result,
            VerificationStatus.FAILURE,
            VerificationAction.FAIL,
            result.error or result.summary,
            observation,
        )

    @staticmethod
    def _result(
        result: SkillResult,
        status: VerificationStatus,
        action: VerificationAction,
        summary: str,
        observation: Observation,
    ) -> VerificationResult:
        return VerificationResult(
            trace=result.trace,
            level=VerificationLevel.RUNTIME,
            status=status,
            summary=summary,
            reasons=[] if status == VerificationStatus.SUCCESS else [summary],
            evidence=result.evidence,
            recommended_action=action,
            metadata={"post_observation_id": observation.trace.observation_id},
        )


class LLMSkillVerifier:
    """Verify semantic subgoal completion from result evidence and a fresh scene."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def verify_skill(
        self,
        request: SkillRequest,
        result: SkillResult,
        observation: Observation,
    ) -> VerificationResult:
        verifier_evidence = _compact_skill_evidence(result)
        execution_facts = verifier_evidence["deterministic_execution_facts"]
        completion = (result.outputs or {}).get("semantic_completion")
        verifier_images: list[tuple[str, str]] = []
        if isinstance(completion, dict) and completion.get("task_type") == "object_goal":
            if bool(completion.get("satisfied")):
                for item in result.evidence:
                    if item.kind == "slow_brain_pixel_goal" and item.ref:
                        verifier_images = [("TARGET_GROUNDING_SOURCE_ARTIFACT", item.ref)]
            keyframes = [
                item.ref
                for item in result.evidence
                if item.kind == "trajectory_keyframe" and item.ref
            ][-3:]
            verifier_images.extend(
                (f"RECENT_EXECUTION_KEYFRAME_{index}", ref)
                for index, ref in enumerate(keyframes, start=1)
                if ref not in {item[1] for item in verifier_images}
            )
        prompt = (
            f"Skill command: {request.command}\n"
            f"Required semantic success conditions: {request.success_conditions}\n"
            f"Skill status: {result.status.value}\n"
            f"Skill summary: {result.summary}\n"
            f"Compressed execution history: {result.compressed_history or 'none'}\n"
            f"Authoritative deterministic execution facts: {execution_facts}\n"
            f"Bounded execution evidence: {verifier_evidence}\n"
            "Deterministic execution facts are produced by the Runtime and embodiment adapter. Treat them as "
            "authoritative for whether a named topology PointGoal was reached, whether motion stalled, and which "
            "controller terminated. Never claim that a reached_topology_node_id was not reached. These facts do "
            "not by themselves prove that a semantic object is visible or that the global task is complete. "
            "An affordance_execution_chunk_reached result is only an intermediate visual-semantic movement chunk; "
            "judge it achieved only when the delegated success condition itself is supported by the fresh scene. "
            "For an object_goal, semantic_completion is a mandatory protocol boundary: only a current-frame, "
            "high-confidence target grounding followed by target-role execution may complete the Skill. Seeing the "
            "object in the verifier image is useful corrective evidence, but cannot silently replace the missing "
            "Target artifact; return not_achieved so the Navigation Skill can ground and execute that Target. "
            "When a TARGET_GROUNDING_SOURCE_ARTIFACT is attached, it is the synchronized source observation used "
            "for the executed Target pixel, not historical plan text. Jointly verify that artifact, its confidence "
            "and role in semantic_completion, the deterministic target execution facts, and the fresh post-skill "
            "scene. A target leaving the camera after a successful close approach does not by itself invalidate the "
            "executed Target artifact; reject it only when the source artifact is visually wrong, confidence or "
            "execution is insufficient, or the post-skill evidence contradicts the claimed physical result. "
            "RECENT_EXECUTION_KEYFRAME images are bounded chronological evidence from this Skill run only. Use them "
            "to determine what the robot actually approached; they are not persistent scene truth and must not be "
            "confused with the latest synchronized camera observation. "
            "Judge whether the semantic subgoal is supported by the source-bound Target artifact, its execution "
            "trajectory, and the fresh post-skill observation together. Do not require the approached object to "
            "remain fully visible in the final camera frame when the same Skill run already provides a valid "
            "Target artifact and authoritative target-role reach evidence. A depth-limited Target projection is "
            "only an intermediate proxy waypoint: target_proxy_reached does not mean target_reached and cannot "
            "complete the ObjectGoal. "
            "A normal RPC return or OmniNav local STOP is evidence only. If evidence is insufficient or the "
            "scene contradicts the required subgoal, return outcome=not_achieved. Return outcome=achieved exactly "
            "when the delegated semantic subgoal is supported, otherwise return "
            "outcome=not_achieved. Do not choose a next action or global task disposition. Skill-level verification "
            "never authorizes global task failure; only the Main Agent and Finish Verifier may decide what happens "
            "next. LEFT, FRONT, "
            "and RIGHT are equally valid synchronized visual evidence, with no privileged primary view. Seeing the "
            "target in any one of them can satisfy a visibility condition. In summary and "
            "reasons, state whether the subgoal was achieved, why it failed if it failed, what scene information "
            "was discovered, what recovery was attempted, and how the Main Agent should use the outcome."
        )
        decision = await _complete_decision(
            self.provider,
            "You are the ABot-AgentOS skill-level semantic verifier. Return strict JSON.",
            prompt,
            observation,
            decision_model=_SkillDecision,
            evidence_images=verifier_images,
        )
        verification = _verification_from_decision(
            decision,
            trace=_trace_for_observation(result.trace, observation),
            level=VerificationLevel.SKILL,
            observation=observation,
            base_evidence=result.evidence,
        )
        has_target_artifact = any(
            label == "TARGET_GROUNDING_SOURCE_ARTIFACT"
            for label, _ref in verifier_images
        )
        if (
            isinstance(completion, dict)
            and bool(completion.get("satisfied"))
            and has_target_artifact
            and verification.status != VerificationStatus.SUCCESS
        ):
            review_prompt = (
                f"{prompt}\n"
                "Conflict review: the first Verifier verdict rejected this Skill even though the structured "
                "semantic_completion says the current Skill run grounded a high-confidence Target, executed that "
                "Target role, reached an un-clipped final projection, and attached a bound "
                "TARGET_GROUNDING_SOURCE_ARTIFACT. Reinspect the "
                "source artifact and execution keyframes. The final camera may crop or lose the object after close "
                "approach. Return not_achieved only if the source artifact depicts the wrong object, the confidence "
                "or target-role execution is insufficient, or fresh evidence physically contradicts reaching it. "
                f"First verdict summary: {verification.summary}"
            )
            reviewed_decision = await _complete_decision(
                self.provider,
                "You are the ABot-AgentOS skill-level semantic verifier performing an evidence-conflict review. Return strict JSON.",
                review_prompt,
                observation,
                decision_model=_SkillDecision,
                evidence_images=verifier_images,
            )
            reviewed = _verification_from_decision(
                reviewed_decision,
                trace=_trace_for_observation(result.trace, observation),
                level=VerificationLevel.SKILL,
                observation=observation,
                base_evidence=result.evidence,
            )
            verification = reviewed.model_copy(
                update={
                    "metadata": {
                        **reviewed.metadata,
                        "conflict_review": True,
                        "first_verdict": verification.model_dump(mode="json"),
                    }
                }
            )
        if (
            isinstance(completion, dict)
            and completion.get("task_type") == "object_goal"
            and not bool(completion.get("satisfied"))
        ):
            summary = (
                "ObjectGoal remains active: Affordance progress or visual recognition cannot complete "
                "the Skill until a high-confidence Target is grounded and reached."
            )
            return verification.model_copy(
                update={
                    "status": VerificationStatus.RETRY,
                    "summary": summary,
                    "reasons": [*verification.reasons, summary],
                    "recommended_action": VerificationAction.REPLAN,
                    "metadata": {
                        **verification.metadata,
                        "semantic_completion": completion,
                    },
                }
            )
        return verification


def _compact_skill_evidence(result: SkillResult) -> dict:
    """Keep verification grounded without serializing raw trajectories into the prompt."""
    outputs = result.outputs or {}
    execution = outputs.get("result") or {}
    decision = outputs.get("decision") or {}
    slow_output = outputs.get("slow_brain_output") or {}
    contract = outputs.get("fast_system_contract") or {}
    execution_mode = outputs.get("execution_mode")
    reached = execution.get("reached")
    reached_topology_node_id = None
    if execution_mode == "topology_node_point_goal" and reached is True:
        reached_topology_node_id = contract.get("topology_node_id")
    return {
        "status": outputs.get("status"),
            "deterministic_execution_facts": {
                "execution_mode": execution_mode,
                "anchor_role": outputs.get("anchor_role"),
                "return_control_reason": outputs.get("return_control_reason"),
                "reached": reached,
            "reached_topology_node_id": reached_topology_node_id,
            "termination_reason": execution.get("termination_reason"),
            "stalled": execution.get("stalled"),
            "execution_telemetry": outputs.get("execution_telemetry"),
        },
        "slow_system": {
            key: slow_output.get(key)
            for key in ("cot", "affordance_pixel", "target_pixel")
            if slow_output.get(key) is not None
        },
        "semantic_completion": outputs.get("semantic_completion"),
        "legacy_action": {
            key: decision.get(key)
            for key in ("action_type", "target", "view", "topology_node_id", "location_id")
            if decision.get(key) is not None
        },
        "execution": {
            key: execution.get(key)
            for key in (
                "controller",
                "reached",
                "distance_m",
                "remaining_distance_m",
                "collisions",
                "stalled",
                "termination_reason",
            )
            if execution.get(key) is not None
        },
        "next_observation_id": outputs.get("next_observation_id"),
        "evidence": [item.model_dump(mode="json") for item in result.evidence[:6]],
    }


class LLMFinishVerifier:
    """Authorize global completion only against the original user instruction."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def verify_finish(
        self,
        task: TaskRequest,
        plan: Plan,
        skill_results: list[SkillResult],
        observation: Observation,
    ) -> VerificationResult:
        terminal_target = _latest_terminal_target_evidence(skill_results)
        verifier_images = []
        if terminal_target is not None:
            verifier_images.append(
                ("TERMINAL_TARGET_GROUNDING_SOURCE_ARTIFACT", terminal_target["artifact_ref"])
            )
        result_summary = "\n".join(
            f"- {item.skill_name}: {item.status.value}; {item.compressed_history or item.summary}"
            for item in skill_results
        )
        prompt = (
            f"Original user instruction: {task.instruction}\n"
            f"Explicit task success conditions: "
            f"{task.success_conditions or plan.metadata.get('task_success_conditions', [])}\n"
            f"Current plan revision: {plan.plan_revision}\n"
            f"Plan intent, which is not execution evidence: "
            f"{[step.model_dump(mode='json') for step in plan.steps]}\n"
            f"Actual executed Skill summaries, which override conflicting plan intent:\n"
            f"{result_summary or '- none'}\n"
            f"Bound terminal ObjectGoal evidence: {terminal_target or 'none'}\n"
            "Inspect the attached latest synchronized LEFT/FRONT/RIGHT observation as current visual evidence. "
            "Use the explicit ROBOT_LEFT_CAMERA, ROBOT_FRONT_CAMERA, and ROBOT_RIGHT_CAMERA labels attached "
            "immediately before each image. The first, second, and third attached images are LEFT, FRONT, and "
            "RIGHT respectively; do not relabel an image from visual continuity or image-local left/right. "
            "Do not infer that a planned node was reached "
            "when the actual Skill summary names a different intermediate node. For find/locate/search tasks, "
            "the exact requested object must be visually proven either in a latest current view or in an attached "
            "TERMINAL_TARGET_GROUNDING_SOURCE_ARTIFACT from a satisfied ObjectGoal. The terminal artifact is valid "
            "only when the bound structured evidence says the Target was grounded above threshold, executed in "
            "the target role, and reached without a depth-limited proxy projection. Inspect the artifact yourself "
            "and reject it when it shows the wrong "
            "object. A target becoming cropped, occluded, or too close to recognize in the post-execution view "
            "does not erase a correct source artifact from that same terminal Skill execution. A room label, "
            "nearby related object, plan text, or unbound historical image is not visual evidence. Decide whether "
            "the original task is completely satisfied by the bounded terminal evidence together with the latest "
            "observation. Use finish only when those source-bound visual and execution facts support every required "
            "condition. "
            "OmniNav STOP is never sufficient by itself. Use replan if meaningful work remains. "
            "Use disposition=finish exactly when every requirement is supported; otherwise use replan when "
            "meaningful work remains or fail only for a supported terminal failure. Put missing requirements and "
            "corrective feedback for the Main Agent in reasons."
        )
        decision = await _complete_decision(
            self.provider,
            "You are the ABot-AgentOS finish verifier and sole authority for global task completion. Return strict JSON.",
            prompt,
            observation,
            decision_model=_FinishDecision,
            evidence_images=verifier_images,
        )
        return _verification_from_decision(
            decision,
            trace=_trace_for_observation(
                task.trace,
                observation,
                plan_revision=plan.plan_revision,
            ),
            level=VerificationLevel.FINISH,
            observation=observation,
            base_evidence=[],
        )


def _latest_terminal_target_evidence(
    skill_results: list[SkillResult],
) -> dict[str, object] | None:
    """Return only a source artifact bound to a satisfied terminal ObjectGoal."""
    for result in reversed(skill_results):
        completion = (result.outputs or {}).get("semantic_completion")
        if not isinstance(completion, dict):
            continue
        if completion.get("task_type") != "object_goal" or not bool(
            completion.get("satisfied")
        ):
            continue
        if not all(
            bool(completion.get(key))
            for key in ("target_grounded", "target_executed", "target_reached")
        ):
            continue
        if bool(completion.get("target_projection_limited")):
            continue
        artifact = next(
            (
                item
                for item in result.evidence
                if item.kind == "slow_brain_pixel_goal" and item.ref
            ),
            None,
        )
        if artifact is None:
            continue
        return {
            "target_label": completion.get("target_label"),
            "target_view": completion.get("target_view"),
            "target_confidence": completion.get("target_confidence"),
            "target_confidence_threshold": completion.get(
                "target_confidence_threshold"
            ),
            "source_observation_id": completion.get("source_observation_id"),
            "post_observation_id": completion.get("post_observation_id"),
            "target_grounded": True,
            "target_executed": True,
            "target_reached": True,
            "artifact_ref": artifact.ref,
        }
    return None


def _trace_for_observation(base_trace, observation: Observation, *, plan_revision=None):
    """Keep Agent task identity while binding semantic evidence to its actual frame."""
    current = observation.trace
    return base_trace.model_copy(
        update={
            "plan_revision": (
                base_trace.plan_revision if plan_revision is None else plan_revision
            ),
            "observation_id": current.observation_id,
            "frame_timestamp": current.frame_timestamp,
            "pose_timestamp": current.pose_timestamp,
        }
    )

async def _complete_decision(
    provider: LLMProvider,
    system: str,
    prompt: str,
    observation: Observation,
    *,
    decision_model: type[SchemaModel] = _SkillDecision,
    evidence_images: list[tuple[str, str]] | None = None,
) -> SchemaModel:
    view_audit = await audit_observation_views(provider, observation)
    content = audited_observation_content(prompt, observation, view_audit)
    for label, ref in (evidence_images or [])[-4:]:
        content.extend(image_ref_content(label, ref))
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(
            role="user",
            content=content,
        ),
    ]
    last_error = "unknown verifier protocol error"
    for attempt in range(3):
        response = await provider.complete(
            messages,
            response_schema=decision_model.model_json_schema(),
        )
        if isinstance(response.parsed_json, dict):
            try:
                return decision_model.model_validate(response.parsed_json)
            except ValidationError as exc:
                last_error = str(exc)
        else:
            last_error = "response did not contain a complete JSON object"
        if attempt < 2:
            messages.extend(
                [
                    ChatMessage(role="assistant", content=response.content[-1000:] or "{}"),
                    ChatMessage(
                        role="user",
                            content=(
                                "The Verifier response violated its JSON protocol. Return only one complete "
                                "JSON object matching the requested verifier schema. "
                            f"Previous error: {last_error}"
                        ),
                    ),
                ]
            )
    raise ValueError(f"Verifier protocol failed after 3 attempts: {last_error}")


def _verification_from_decision(
    decision: _SkillDecision | _FinishDecision,
    *,
    trace,
    level: VerificationLevel,
    observation: Observation,
    base_evidence: list[Evidence],
) -> VerificationResult:
    if isinstance(decision, _FinishDecision):
        status = {
            "finish": VerificationStatus.SUCCESS,
            "replan": VerificationStatus.RETRY,
            "fail": VerificationStatus.FAILURE,
        }[decision.disposition]
        action = {
            "finish": VerificationAction.FINISH,
            "replan": VerificationAction.REPLAN,
            "fail": VerificationAction.FAIL,
        }[decision.disposition]
    else:
        status = {
            "achieved": VerificationStatus.SUCCESS,
            "not_achieved": VerificationStatus.RETRY,
        }[decision.outcome]
        action = {
            "achieved": VerificationAction.CONTINUE,
            "not_achieved": VerificationAction.REPLAN,
        }[decision.outcome]
    evidence = list(base_evidence)
    evidence.extend(
        Evidence(
            kind="semantic_verification",
            summary=summary,
            ref=f"observation:{observation.trace.observation_id}",
        )
        for summary in decision.observed_evidence
    )
    return VerificationResult(
        trace=trace,
        level=level,
        status=status,
        summary=decision.summary,
        reasons=decision.reasons,
        evidence=evidence,
        recommended_action=action,
        metadata={"post_observation_id": observation.trace.observation_id},
    )
