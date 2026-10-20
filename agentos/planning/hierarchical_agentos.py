from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import Field, ValidationError

from agentos.providers.base import LLMProvider
from agentos.providers.vision import (
    audit_observation_views,
    audited_observation_content,
    image_ref_content,
)
from agentos.schemas.capability import RiskLevel, SkillDescriptor
from agentos.schemas.enums import SkillStatus
from agentos.schemas.llm import ChatMessage, LLMResponse
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan, PlanStep
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.task_reasoning import AdvanceNavigationInput, SubtaskReflection
from agentos.schemas.base import SchemaModel
from agentos.skills.base import SkillContext


class HierarchicalToolService(Protocol):
    def begin_navigation(self, payload: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def advance_navigation(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class HierarchicalAgentSession:
    """Bounded handoff shared by one Main Agent and its Skill Runner."""

    global_goal: str
    handoff: dict[str, Any] | None = None
    macro_index: int = 0
    recent_macro_decisions: list[dict[str, Any]] = field(default_factory=list)


class _ModelSubtaskReflection(SchemaModel):
    subtask_id: str = Field(pattern=r"^subtask-[0-9]{3,}$")
    disposition: Literal["completed", "continue_active_subtask", "replan"]
    feedback: str = Field(min_length=1, max_length=240)

    def to_legacy(self) -> SubtaskReflection:
        return SubtaskReflection(
            subtask_id=self.subtask_id,
            completed=self.disposition == "completed",
            outcome={
                "completed": "completed",
                "continue_active_subtask": "partial",
                "replan": "failed",
            }[self.disposition],
            feedback=self.feedback,
            should_replan=self.disposition == "replan",
        )


class _ReflectedAdvanceNavigationInput(AdvanceNavigationInput):
    """Slow decision contract after a macro has produced execution evidence."""

    self_reflection: _ModelSubtaskReflection


class HierarchicalMacroPlanner:
    """Generate one ABot-style macro action per AgentOS plan revision."""

    skill_name = "HierarchicalNavigationSkill"

    def __init__(
        self,
        provider: LLMProvider,
        service: HierarchicalToolService,
        session: HierarchicalAgentSession,
    ) -> None:
        self.provider = provider
        self.service = service
        self.session = session

    async def plan(self, task: TaskRequest, observation: Observation) -> Plan:
        if self.session.handoff is None:
            packet = await asyncio.to_thread(
                self.service.begin_navigation,
                {"global_goal": task.instruction},
            )
            self.session.handoff = self.compress_handoff(packet)
        return await self._plan_macro(task, observation, task.trace.plan_revision)

    async def replan(
        self,
        task: TaskRequest,
        observation: Observation,
        previous_plan: Plan,
        skill_results: list[SkillResult],
        *,
        reason: str,
        next_revision: int,
    ) -> Plan:
        del previous_plan, skill_results
        return await self._plan_macro(task, observation, next_revision, reason=reason)

    async def _plan_macro(
        self,
        task: TaskRequest,
        observation: Observation,
        revision: int,
        *,
        reason: str = "",
    ) -> Plan:
        handoff = self.session.handoff or {}
        packet_observation = self._packet_observation(handoff) or observation
        context = self._decision_context(handoff, reason)
        view_audit = await audit_observation_views(self.provider, packet_observation)
        content = audited_observation_content(
            (
                f"Original global task: {task.instruction}\n"
                f"AgentOS plan revision: {revision}\n"
                f"Structured Slow context and previous macro result:\n{context}\n"
                "Produce exactly one AdvanceNavigationInput. Select the most useful next macro action. "
                "Use NODE_NAV for a deliberate connected regional transition, VISUAL_ANCHOR_NAV for a "
                "currently visible task-relevant anchor, VIEWPOINT_ADJUST when a 90-degree observation change "
                "is more informative, or L_REVISIT only when historical evidence is stronger. The first call "
                "must include planner_output. Later calls must follow next_call_contract and include a concise "
                "Self-Reflector assessment of the completed macro. Do not invent topology or location IDs. "
                "Protocol invariants: visible_anchor must include view=left|front|right and no location IDs; "
                "its target must name a visible object, doorway, passage, or traversable region and must never "
                "be a backbone-* or location-* ID; "
                "known_location must include exactly one topology_node_id or location_id and no view; "
                "viewpoint_adjustment must include viewpoint_action. Planner subtask stage/task_type pairs are "
                "approaching/point_goal, reaching/object_goal|poi_goal, and "
                "interaction/instruction_following|person_following. When regional_guidance.mode is "
                "route_to_region, select known_location for its exact next_topology_node_id. Do not use a "
                "visible affordance anchor or viewpoint adjustment instead; the only exception is a directly "
                "visible final task object selected as anchor_role=target. When regional_guidance.mode is "
                "search_within_region, keep the active object-search subtask and select a visible target or "
                "traversable affordance anchor, a 90-degree viewpoint adjustment, or an evidence-backed "
                "historical revisit. Do not navigate to the current topology node again and do not replace the "
                "regional plan merely because one view does not contain the object."
            ),
            packet_observation,
            view_audit,
        )
        for index, ref in enumerate(self._previous_keyframe_refs(handoff), start=1):
            content.extend(image_ref_content(f"PREVIOUS_MACRO_FRONT_KEYFRAME_{index}", ref))
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are the server-side ABot-AgentOS Main Agent. Return strict JSON only. "
                    "Semantic understanding, causal task progress, subtask reflection, and action selection "
                    "are your responsibility. Runtime geometry and control are deterministic Skills. "
                    "Use the ABot-N0 typed Planner/Actor/Self-Reflector fields and ABot-N1 task-specific CoT."
                ),
            ),
            ChatMessage(role="user", content=content),
        ]
        decision, response = await self._complete_valid_decision(
            messages,
            observation_id=packet_observation.trace.observation_id,
            global_goal=task.instruction,
        )
        self.session.macro_index += 1
        action = decision.action
        self.session.recent_macro_decisions.append(
            {
                "revision": revision,
                "action_type": action.action_type,
                "target": action.target,
                "topology_node_id": action.topology_node_id,
                "location_id": action.location_id,
                "viewpoint_action": action.viewpoint_action,
                "verifier_feedback": reason or None,
            }
        )
        self.session.recent_macro_decisions = self.session.recent_macro_decisions[-4:]
        rationale = (
            f"{action.action_type}: {action.target}; intent={action.intent}; "
            f"expected={action.expected_evidence}"
        )
        return Plan(
            task_id=task.trace.task_id,
            plan_revision=revision,
            rationale_summary=rationale,
            steps=[
                PlanStep(
                    step_id=f"macro-{self.session.macro_index:03d}",
                    skill_name=self.skill_name,
                    command=f"Execute {action.action_type}: {action.target}",
                    success_conditions=[
                        "The selected macro action executes without a runtime failure.",
                        "A fresh synchronized three-view observation is produced for verification.",
                    ],
                    timeout_s=300.0,
                    max_retries=0,
                    inputs=decision.model_dump(mode="json"),
                )
            ],
            metadata={
                "task_type": decision.navigation_task_type,
                "task_success_conditions": task.success_conditions,
                "macro_action": action.model_dump(mode="json"),
                "planner_model": response.model,
            },
        )

    async def _complete_valid_decision(
        self,
        messages: list[ChatMessage],
        *,
        observation_id: str,
        global_goal: str,
    ) -> tuple[AdvanceNavigationInput, LLMResponse]:
        last_error: Exception | None = None
        decision_model = self._decision_model()
        for attempt in range(3):
            response = await self.provider.complete(
                messages,
                response_schema=decision_model.model_json_schema(),
            )
            if not isinstance(response.parsed_json, dict):
                raise ValueError("Hierarchical Main Agent did not return a JSON object")
            payload = dict(response.parsed_json)
            payload["observation_id"] = observation_id
            payload["global_goal"] = global_goal
            payload = self._normalize_structural_action_fields(payload)
            errors: list[dict[str, str]] = []
            try:
                model_decision = decision_model.model_validate(payload)
                if isinstance(model_decision.self_reflection, _ModelSubtaskReflection):
                    converted = model_decision.model_dump(mode="json")
                    converted["self_reflection"] = (
                        model_decision.self_reflection.to_legacy().model_dump(mode="json")
                    )
                    decision = AdvanceNavigationInput.model_validate(converted)
                else:
                    decision = model_decision
            except ValidationError as exc:
                last_error = exc
                errors = [
                    {
                        "location": ".".join(str(item) for item in error["loc"]),
                        "message": error["msg"],
                    }
                    for error in exc.errors(include_url=False, include_input=False)
                ]
            else:
                errors.extend(self._semantic_action_errors(decision))
                if self.session.macro_index == 0 and decision.planner_output is None:
                    errors.append(
                        {
                            "location": "planner_output",
                            "message": "planner_output is required for the first macro decision",
                        }
                    )
                if not errors:
                    return decision, response
                last_error = ValueError(f"AgentOS decision protocol errors: {errors}")
            if attempt == 2:
                break
            if errors:
                messages.extend(
                    [
                        ChatMessage(
                            role="assistant",
                            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        ),
                        ChatMessage(
                            role="user",
                            content=(
                                "Your JSON failed the execution protocol. Correct every listed error, add any "
                                "required missing decision object, and return the complete JSON object again. "
                                "For visible_anchor, "
                                "view is mandatory and must identify the actual source camera. For planner "
                                "subtasks, use only the allowed stage/task_type pairs stated above. "
                                f"Errors: {errors}"
                            ),
                        ),
                    ]
                )
        assert last_error is not None
        raise last_error

    def _decision_model(self) -> type[AdvanceNavigationInput]:
        contract = (self.session.handoff or {}).get("next_call_contract") or {}
        if str(contract.get("self_reflection") or "").startswith("required"):
            return _ReflectedAdvanceNavigationInput
        return AdvanceNavigationInput

    def _semantic_action_errors(
        self,
        decision: AdvanceNavigationInput,
    ) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = []
        contract = (self.session.handoff or {}).get("next_call_contract") or {}
        reflection_contract = str(contract.get("self_reflection") or "")
        expected_subtask_id = contract.get("self_reflection_subtask_id")
        if reflection_contract.startswith("required"):
            if decision.self_reflection is None:
                errors.append(
                    {
                        "location": "self_reflection",
                        "message": (
                            "Self-Reflector output is required after the previous macro. Assess its actual "
                            "result and decide completed/partial/failed before selecting the next action."
                        ),
                    }
                )
            elif (
                expected_subtask_id
                and decision.self_reflection.subtask_id != expected_subtask_id
            ):
                errors.append(
                    {
                        "location": "self_reflection.subtask_id",
                        "message": f"Self-Reflector must assess active subtask {expected_subtask_id}.",
                    }
                )
        route_plan = (self.session.handoff or {}).get("semantic_route_plan") or {}
        guidance = route_plan.get("regional_guidance") or {}
        next_node_id = guidance.get("next_topology_node_id")
        if guidance.get("mode") == "search_within_region":
            execution = (self.session.handoff or {}).get("execution_summary") or {}
            runtime_blocked = bool(execution.get("stalled")) or execution.get(
                "termination_reason"
            ) in {"blocked", "path_invalid", "timeout", "cancelled"}
            if decision.self_reflection is not None:
                if decision.self_reflection.should_replan and not runtime_blocked:
                    errors.append(
                        {
                            "location": "self_reflection.disposition",
                            "message": (
                                "The target region is reached and runtime is healthy. Use disposition="
                                "continue_active_subtask while performing local visual search; absence from "
                                "one view is not evidence that the regional plan failed."
                            ),
                        }
                    )
            if (
                decision.action.action_type == "known_location"
                and decision.action.topology_node_id
                in {
                    guidance.get("current_topology_node_id"),
                    guidance.get("target_topology_node_id"),
                }
            ):
                errors.append(
                    {
                        "location": "action",
                        "message": (
                            "The robot is already inside the target region. Select a visible target/affordance "
                            "anchor, a viewpoint adjustment, or an evidence-backed location revisit instead "
                            "of navigating to the same topology node."
                        ),
                    }
                )
            recent_actions = self.session.recent_macro_decisions[-2:]
            if (
                decision.action.action_type == "viewpoint_adjustment"
                and len(recent_actions) == 2
                and all(
                    item.get("action_type") == "viewpoint_adjustment"
                    for item in recent_actions
                )
            ):
                errors.append(
                    {
                        "location": "action",
                        "message": (
                            "Two consecutive 90-degree viewpoint adjustments already sampled new sectors. "
                            "Select a currently visible traversable affordance anchor to make spatial "
                            "progress, or use an evidence-backed historical revisit; do not rotate again."
                        ),
                    }
                )
        if guidance.get("mode") != "route_to_region" or not next_node_id:
            return errors
        action = decision.action
        if action.action_type == "visible_anchor" and action.anchor_role == "target":
            return errors
        if action.action_type == "known_location" and action.topology_node_id == next_node_id:
            return errors
        errors.append(
            {
                "location": "action",
                "message": (
                    "Topo-Memory requires a regional transition. Select action_type=known_location "
                    f"with topology_node_id={next_node_id}. A visible_anchor is valid here only for a "
                    "directly visible final task object with anchor_role=target."
                ),
            }
        )
        return errors

    @staticmethod
    def _normalize_structural_action_fields(payload: dict[str, Any]) -> dict[str, Any]:
        """Drop fields that cannot affect the model-selected action type."""
        action = payload.get("action")
        if not isinstance(action, dict):
            return payload
        normalized = dict(payload)
        normalized_action = dict(action)
        action_type = normalized_action.get("action_type")
        if action_type != "visible_anchor":
            normalized_action["anchor_role"] = "affordance"
        if action_type == "viewpoint_adjustment":
            for field in ("view", "topology_node_id", "location_id"):
                normalized_action[field] = None
        normalized["action"] = normalized_action
        return normalized

    @staticmethod
    def _packet_observation(packet: dict[str, Any]) -> Observation | None:
        raw = packet.get("post_observation") or packet.get("observation")
        return Observation.model_validate(raw) if isinstance(raw, dict) else None

    def _decision_context(self, handoff: dict[str, Any], reason: str) -> str:
        context = {
            "slow_context": handoff.get("slow_context"),
            "semantic_route_plan": handoff.get("semantic_route_plan"),
            "next_call_contract": handoff.get("next_call_contract"),
            "previous_decision": handoff.get("previous_decision"),
            "previous_execution": handoff.get("execution_summary"),
            "active_target_evidence": handoff.get("active_target_evidence"),
            "runtime_replan_reason": reason or None,
            "recent_macro_decisions": self.session.recent_macro_decisions[-4:],
        }
        return json.dumps(context, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _previous_keyframe_refs(handoff: dict[str, Any]) -> list[str]:
        return [str(ref) for ref in handoff.get("selected_keyframe_refs") or []][:4]

    @staticmethod
    def compress_handoff(packet: dict[str, Any]) -> dict[str, Any]:
        """Create the only execution state visible to the Main Agent."""
        result = packet.get("result") or {}
        frames = result.get("trajectory_keyframes") or []
        refs: list[str] = []
        for frame in frames[-4:]:
            ref = (frame.get("rgb_refs") or {}).get("front")
            if ref and ref not in refs:
                refs.append(str(ref))
        observation = packet.get("post_observation") or packet.get("observation")
        return {
            "ok": bool(packet.get("ok", False)),
            "status": packet.get("status"),
            "observation": observation,
            "slow_context": packet.get("slow_context"),
            "semantic_route_plan": packet.get("semantic_route_plan"),
            "next_call_contract": packet.get("next_call_contract"),
            "previous_decision": packet.get("decision"),
            "execution_summary": {
                "controller": result.get("controller"),
                "reached": result.get("reached"),
                "control_steps": result.get("control_steps"),
                "distance_m": result.get("distance_m"),
                "remaining_distance_m": result.get("remaining_distance_m"),
                "collisions": result.get("collisions"),
                "stalled": result.get("stalled"),
                "termination_reason": result.get("termination_reason"),
                "next_observation_id": packet.get("next_observation_id"),
            },
            "active_target_evidence": packet.get("active_target_evidence"),
            "selected_keyframe_refs": refs,
        }


class HierarchicalNavigationSkill:
    """Execute one model-selected macro action through the stable hierarchical tools."""

    name = HierarchicalMacroPlanner.skill_name
    descriptor = SkillDescriptor(
        name=name,
        description=(
            "Execute one isolated macro navigation action using semantic topology, visible-anchor grounding, "
            "historical-location revisit, viewpoint adjustment, depth projection, A*, and bounded control."
        ),
        input_schema=AdvanceNavigationInput.model_json_schema(),
        output_schema={"type": "object"},
        required_modalities=["rgb", "depth", "pose", "map"],
        supported_embodiments=["habitat-gs", "ros2"],
        risk_level=RiskLevel.MEDIUM,
        side_effects=["moves_robot"],
        default_timeout_s=300.0,
    )

    def __init__(
        self,
        service: HierarchicalToolService,
        session: HierarchicalAgentSession,
    ) -> None:
        self.service = service
        self.session = session

    async def run(
        self,
        request: SkillRequest,
        observation: Observation,
        context: SkillContext,
    ) -> SkillResult:
        del observation
        context.cancellation.raise_if_cancelled()
        decision = AdvanceNavigationInput.model_validate(request.inputs)
        decision = self._adapt_legacy_reflection(decision)
        await context.emit_progress(
            request,
            f"Executing {decision.action.action_type}: {decision.action.target}",
            action=decision.action.model_dump(mode="json"),
        )
        packet = await asyncio.to_thread(
            self.service.advance_navigation,
            decision.model_dump(mode="json"),
        )
        self.session.handoff = HierarchicalMacroPlanner.compress_handoff(packet)
        result = packet.get("result") or {}
        finish_check_required = packet.get("status") == "awaiting_finish_verification"
        action = packet.get("decision") or decision.action.model_dump(mode="json")
        ok = bool(packet.get("ok", False)) and packet.get("status") != "failed"
        status = SkillStatus.SUCCESS if ok else SkillStatus.RETRYABLE_FAILURE
        if finish_check_required:
            summary = (
                "FINISH_CHECK_REQUIRED; legacy plan has no active subtask; "
                f"next_observation={packet.get('next_observation_id')}"
            )
        else:
            mode = self._action_mode(action)
            executed_target = self._action_target(action)
            requested_target = self._action_target(
                decision.action.model_dump(mode="json")
            )
            if requested_target != executed_target:
                summary = (
                    f"INTERMEDIATE_{mode} requested={requested_target}; executed={executed_target}; "
                    f"requested_reached=False; executed_reached={result.get('reached')}; "
                    f"steps={result.get('control_steps', 0)}; "
                    f"distance={float(result.get('distance_m') or 0.0):.2f}m; "
                    f"next_observation={packet.get('next_observation_id')}"
                )
            else:
                summary = (
                    f"{mode} target={executed_target}; reached={result.get('reached')}; "
                    f"steps={result.get('control_steps', 0)}; "
                    f"distance={float(result.get('distance_m') or 0.0):.2f}m; "
                    f"next_observation={packet.get('next_observation_id')}"
                )
        evidence = self._evidence(packet)
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=status,
            summary=summary,
            evidence=evidence,
            outputs=packet,
            compressed_history=summary,
            stop_reason=(str(result.get("termination_reason")) if result.get("termination_reason") else None),
            error=None if ok else str(packet.get("error") or "hierarchical macro action failed"),
        )

    def _adapt_legacy_reflection(
        self,
        decision: AdvanceNavigationInput,
    ) -> AdvanceNavigationInput:
        """Translate the AgentOS assessment into the legacy tool protocol."""
        if decision.self_reflection is not None:
            return decision
        route_plan = (self.session.handoff or {}).get("semantic_route_plan") or {}
        agentic_plan = route_plan.get("agentic_plan") or {}
        subtasks = agentic_plan.get("subtasks") or []
        active_index = int(agentic_plan.get("active_subtask_index") or 0)
        if active_index >= len(subtasks):
            return decision
        subtask_id = subtasks[active_index].get("subtask_id")
        if not subtask_id:
            return decision
        assessment = decision.previous_assessment
        outcome = {
            "completed": "completed",
            "partial": "partial",
            "not_started": "partial",
            "failed": "failed",
        }[assessment.execution_status]
        reflection = SubtaskReflection(
            subtask_id=str(subtask_id),
            completed=assessment.execution_status == "completed",
            outcome=outcome,
            feedback=assessment.evidence,
            should_replan=(
                assessment.execution_status == "failed"
                or assessment.task_effect in {"hypothesis_rejected", "blocked"}
            ),
        )
        return decision.model_copy(update={"self_reflection": reflection})

    @staticmethod
    def _action_mode(action: dict[str, Any]) -> str:
        if action.get("action_type") == "known_location":
            return "NODE_NAV" if action.get("topology_node_id") else "L_REVISIT"
        if action.get("action_type") == "visible_anchor":
            return "VISUAL_ANCHOR_NAV"
        if action.get("action_type") == "viewpoint_adjustment":
            return "VIEWPOINT_ADJUST"
        return "UNKNOWN"

    @staticmethod
    def _action_target(action: dict[str, Any]) -> str | None:
        return action.get("topology_node_id") or action.get("location_id") or action.get("target")

    @staticmethod
    def _evidence(packet: dict[str, Any]) -> list[Evidence]:
        evidence: list[Evidence] = []
        generated = packet.get("generated_point") or {}
        if generated.get("annotated_ref"):
            evidence.append(
                Evidence(
                    kind="anchor_grounding",
                    summary=str(generated.get("evidence_summary") or "Grounded visible anchor."),
                    ref=str(generated["annotated_ref"]),
                )
            )
        next_id = packet.get("next_observation_id")
        if next_id:
            evidence.append(
                Evidence(
                    kind="post_observation",
                    summary="Fresh synchronized observation after macro execution.",
                    ref=f"observation:{next_id}",
                )
            )
        frames = (packet.get("result") or {}).get("trajectory_keyframes") or []
        for index, frame in enumerate(frames[-4:], start=1):
            ref = (frame.get("rgb_refs") or {}).get("front")
            if ref:
                evidence.append(
                    Evidence(
                        kind="trajectory_keyframe",
                        summary=f"Selected front-view trajectory keyframe {index}.",
                        ref=str(ref),
                        data={
                            "pose": frame.get("pose"),
                            "yaw": frame.get("yaw"),
                            "location_id": frame.get("location_id"),
                        },
                    )
                )
        return evidence
