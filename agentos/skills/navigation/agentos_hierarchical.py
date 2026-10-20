from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from pydantic import ValidationError

from agentos.harness.navigation_session import N1NavigationSkillSession
from agentos.schemas.base import utc_now
from agentos.planning.agentos_hierarchical import compress_navigation_handoff
from agentos.providers.base import LLMProvider
from agentos.providers.vision import audit_observation_views, image_ref_content
from agentos.schemas.capability import RiskLevel, SkillDescriptor
from agentos.schemas.enums import SkillStatus
from agentos.schemas.llm import ChatMessage, LLMResponse
from agentos.schemas.navigation import EmbodimentExecutionTelemetry
from agentos.schemas.observation import Observation
from agentos.schemas.perception import (
    VisualDetection,
    VisualGroundingBatchOutput,
    VisualGroundingConstraints,
    VisualGroundingQuery,
)
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.schemas.task_reasoning import (
    AgentOSSkillDelegation,
    GroundedNavigationSlowBrainOutput,
    InstructionFollowingSlowBrainSemanticOutput,
    InstructionFollowingCoT,
    NavigationSlowBrainSemanticOutput,
    NavigationFastSystemInput,
    NavigationSlowBrainInput,
    NavigationSlowBrainOutput,
    ObjectGoalCoT,
    ObjectGoalSlowBrainSemanticOutput,
    PixelOnlySlowBrainSemanticOutput,
    SlowBrainReferenceMemoryFrame,
    SlowBrainReferenceObservation,
    SlowBrainSemanticTopologyNode,
    SlowBrainTopologyContext,
    SlowBrainTopologyEdge,
    SlowBrainTriViewPixelGoal,
    SlowBrainViewBinding,
    SlowBrainViewBindingVerification,
    SlowBrainVisualGrounding,
)
from agentos.skills.base import SkillContext
from agentos.skills.navigation.persistent import OmniNavGuidance, PersistentOmniNavSkill
from agentos.skills.perception import VisualGroundingSkill


class AgentOSHierarchicalService(Protocol):
    def localization_topology_snapshot(
        self, observation: Observation
    ) -> dict[str, Any]: ...

    def prepare_agentic_plan(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def navigate_to_anchor(
        self,
        payload: dict[str, Any],
        *,
        allow_stale_slow_reference: bool = False,
    ) -> dict[str, Any]: ...

    def navigate_to_known_location(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def capture_skill_observation(self) -> Observation: ...

    def record_topology_search(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class SlowPixelGoalAssembler:
    """Convert audited Slow-System bboxes into bottom-center pixel goals."""

    @classmethod
    def assemble(
        cls, grounded: GroundedNavigationSlowBrainOutput
    ) -> NavigationSlowBrainOutput:
        return NavigationSlowBrainOutput(
            cot=grounded.cot,
            affordance_pixel=cls._pixel_slots(grounded.affordance_grounding),
            target_pixel=cls._pixel_slots(grounded.target_grounding),
        )

    @staticmethod
    def _pixel_slots(grounding: SlowBrainVisualGrounding) -> SlowBrainTriViewPixelGoal:
        slots: dict[str, list[int]] = {"left": [], "front": [], "right": []}
        pixel = grounding.bottom_center()
        if grounding.view is not None and pixel is not None:
            slots[grounding.view] = pixel
        return SlowBrainTriViewPixelGoal.model_validate(slots)


class AgentOSMappingSkill:
    """Read-only Skill that gives the Main Agent bounded map and spatial evidence."""

    name = "MappingSkill"
    descriptor = SkillDescriptor(
        name=name,
        description=(
            "Read current localization, semantic landmarks, and bounded topology connectivity without moving the robot."
        ),
        input_schema={
            "type": "object",
            "required": ["source_observation_id"],
            "properties": {"source_observation_id": {"type": "string"}},
            "additionalProperties": True,
        },
        output_schema={"type": "object"},
        required_modalities=["pose", "map"],
        supported_embodiments=["habitat-gs", "ros2"],
        risk_level=RiskLevel.READ_ONLY,
        side_effects=[],
        default_timeout_s=10.0,
    )

    def __init__(self, service: AgentOSHierarchicalService) -> None:
        self.service = service

    async def run(
        self,
        request: SkillRequest,
        observation: Observation,
        context: SkillContext,
    ) -> SkillResult:
        context.cancellation.raise_if_cancelled()
        source_id = str(request.inputs.get("source_observation_id") or "")
        if source_id != observation.trace.observation_id:
            raise ValueError("Mapping delegation source observation is stale")
        snapshot = await asyncio.to_thread(
            self.service.localization_topology_snapshot,
            observation,
        )
        pose = snapshot.get("pose") or {}
        nearby = snapshot.get("nearby_nodes") or []
        nearby_ids = [
            str(item.get("topology_node_id"))
            for item in nearby
            if isinstance(item, dict) and item.get("topology_node_id")
        ]
        topo_memory = snapshot.get("topo_memory") or {}
        connected_ids = {
            str(snapshot.get("current_topology_node_id") or ""),
            *nearby_ids,
        }
        roads = [
            item
            for item in (topo_memory.get("road_layer") or [])
            if isinstance(item, dict)
            and (
                str(item.get("source")) in connected_ids
                or str(item.get("target")) in connected_ids
            )
        ][:12]
        compact = {
            "map_id": snapshot.get("map_id"),
            "pose": {
                key: pose.get(key) for key in ("x", "y", "yaw", "frame", "source")
            },
            "current_topology_node_id": snapshot.get("current_topology_node_id"),
            "nearby_nodes": nearby[:4],
            "connected_edges": roads,
            "topo_memory": topo_memory,
        }
        summary = "Mapping evidence: " + json.dumps(
            compact,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        evidence = [
            Evidence(
                kind="localization_topology",
                summary=summary,
                ref=snapshot.get("topdown_ref"),
                data={
                    "map_id": snapshot.get("map_id"),
                    "nearby_node_ids": nearby_ids,
                },
            )
        ]
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=SkillStatus.SUCCESS,
            summary=summary,
            evidence=evidence,
            outputs=snapshot,
            compressed_history=summary,
            stop_reason="spatial_evidence_observed",
        )


# Compatibility alias for imports outside the AgentOS replica runtime.
AgentOSLocalizationMapSkill = AgentOSMappingSkill

@dataclass(frozen=True)
class _SlowDecisionBundle:
    source_observation: Observation
    slow_input: NavigationSlowBrainInput
    semantic_decision: NavigationSlowBrainSemanticOutput
    view_binding_verification: SlowBrainViewBindingVerification
    slow_output: NavigationSlowBrainOutput
    response: LLMResponse
    grounding: GroundedNavigationSlowBrainOutput
    visual_grounding_result: SkillResult | None
    slow_latency_s: float
    view_binding_latency_s: float
    grounding_latency_s: float
    latency_s: float


class QwenNavigationSlowBrain:
    """ABot-N1 low-frequency reasoner scoped to one navigation Skill run."""

    def __init__(self, provider: LLMProvider, session: N1NavigationSkillSession) -> None:
        self.provider = provider
        self.session = session

    async def decide(
        self,
        request: SkillRequest,
        observation: Observation,
        delegation: AgentOSSkillDelegation,
    ) -> tuple[
        NavigationSlowBrainInput,
        NavigationSlowBrainSemanticOutput,
        LLMResponse,
        SlowBrainViewBindingVerification,
        float,
    ]:
        slow_input = self._build_input(observation, delegation)
        current_view_audit = await audit_observation_views(self.provider, observation)
        if current_view_audit is None:
            raise ValueError("ABot-N1 Slow System requires synchronized left/front/right RGB views")
        slow_input = slow_input.model_copy(
            update={"current_view_audit": current_view_audit}
        )
        previous_observation_id = self.session.previous_decision_observation_id or "unknown"
        prompt = (
            "ABot-N1 Slow-System input (four official categories plus optional N0 topology extension): "
            f"{json.dumps(slow_input.model_dump(mode='json'), ensure_ascii=False, separators=(',', ':'))}\n"
            f"Stable AgentOS global task: {self.session.global_goal}\n"
            f"Current delegated Navigation Skill subtask: {request.command}\n"
            f"Delegated subtask success conditions: {request.success_conditions}\n"
            f"Bounded AgentOS retrieved context: {request.context_summary or 'none'}\n"
            "Retrieved long-term facts are source-grounded semantic hypotheses. They may guide which visible route "
            "anchor is useful, but their historical camera coordinates are invalid and current images remain "
            "authoritative. "
            f"The previous_decision was produced from observation {previous_observation_id}; the authoritative "
            f"current observation is {observation.trace.observation_id}. The authoritative current_view_audit was "
            "produced by three independent single-camera calls, so LEFT, FRONT, and RIGHT identify physical robot "
            "cameras rather than horizontal regions inside one image. previous_decision is a temporal "
            "hypothesis, not current visual evidence. Its view names and pixels belong to the older camera "
            "orientation. Re-evaluate every previous claim against all three current_view_audit entries and regenerate the "
            "task-specific CoT fields plus coordinate-free grounding queries. Current visual evidence overrides "
            "previous_decision. Copy an earlier conclusion only when the current audits independently confirm its "
            "semantic and spatial relation. Historical images may support semantic continuity, but they have "
            "no valid coordinates in the current observation. "
            "A target seen only in previous_decision or reference memory is a historical hypothesis, not a currently "
            "visible target. State that distinction explicitly. Never claim that an object is currently visible or "
            "assign it a current camera view unless that named single-frame audit supports the claim. When the "
            "current audits contradict or no longer show the historical target, update the CoT and route/search intent "
            "instead of preserving the old visibility statement. "
            "Recompute the CoT from current evidence rather than treating previous wording as an answer template. "
            "Before task inference, read the three independently produced current camera audits in the explicit "
            "LEFT, FRONT, RIGHT order. Never reinterpret image-local left, center, and right as camera identities. "
            "For object_goal, begin "
            "target_localization_and_key_context with the compact form "
            "'CURRENT_VIEW_AUDIT: LEFT=<visible evidence>; FRONT=<visible evidence>; RIGHT=<visible evidence>.' "
            "Each clause must faithfully summarize only the corresponding named audit. Use these facts, not image-local "
            "direction words, to assign preferred_view and to reason about target visibility. Keep each CoT field "
            "under 420 characters: retain the task-relevant evidence, topology hypothesis, progress, and next action, "
            "but do not repeat complete visible-object lists or the same target description. "
            "Update semantic state when visibility, route progress, topology region, blockage, or target evidence "
            "changes. If the new observation provides no meaningful semantic change, preserving the same progress "
            "ledger and ongoing intent is valid; never invent progress or paraphrase merely to appear different. "
            "When previous_decision is null, no earlier sub-instruction may be claimed complete solely from the "
            "current scene. "
            "Do not copy incomplete or truncated prior text. "
            "Return one Slow-System semantic decision. Never output a bbox, pixel, depth, world coordinate, or "
            "controller action; an independent current-frame Grounding Skill owns all localization. "
            "affordance_query is required and identifies the next visually groundable, safe, traversable semantic "
            "waypoint described by the CoT. Its label must name a physical, visually localizable and approachable "
            "object or transition; a camera name, view name, or whole image is never an affordance. Selecting this "
            "affordance is a Slow-System reasoning responsibility, "
            "not a Grounding-model route choice: jointly use the stable global task, delegated subtask, current "
            "three-camera audit, previous decision and execution progress, retrieved memory, and the bounded "
            "Topo-Memory node semantics and connectivity. Infer the currently visible traversable region or "
            "transition most likely to advance the unresolved task. When the final target is not directly visible "
            "after reaching a mapped region, explicitly compare that region and its connected nodes' place_type, "
            "function, objects, landmarks, and task_affordances. Use those semantics to choose a currently visible "
            "transition or landmark that exposes the most likely target-bearing space; do not ignore an explicit "
            "mapped object or landmark in favor of undirected corridor motion. Prefer a distinctive directional anchor that is "
            "actually visible, such "
            "as a doorway frame, corridor entrance, room opening, turn corner, end of a counter, gap between named "
            "equipment, or a landmark immediately beside the intended passage. The label must identify that concrete "
            "anchor or transition rather than defaulting to generic phrases such as 'clear floor', 'open area', or "
            "'path ahead'. Use a bounded floor region only when no distinctive route anchor is visible, and then name "
            "its exact direction and visual boundaries. Its spatial_relation must "
            "state how that waypoint advances the current sub-instruction using current observation, topology, "
            "memory, and previous execution progress. Determine from "
            "task_specification.goal whether the user explicitly names a final physical object. Such an explicit "
            "object must always populate target_query, even when Slow reasoning believes it is not currently visible; "
            "the independent Grounding Skill, not Slow reasoning, decides whether current visual evidence localizes it. "
            "If no final physical object is explicitly named, keep "
            "target_query null and complete the task through successive affordance queries. affordance_query must "
            "name the one audited current camera that visually supports the proposed traversable anchor. Grounding "
            "will receive only that selected image, so preferred_view cannot be null for affordance. target_query "
            "may use null only while the final object is not currently visible; once Slow claims target visibility, "
            "it must name the supporting audited camera. "
            "visual_description and excluded_labels should help the Grounding Skill "
            "distinguish current visual evidence from distractors. When a final target may be visible, keep the "
            "affordance query on a safe route toward it and describe the target independently in target_query. "
            "For object_goal, cot contains only target_localization_and_key_context and path_planning. This mode "
            "also handles natural-language route following through successive affordance queries when no final "
            "physical object is visible. For point_goal, cot is null. When topology_context is present, treat it as a semantic and "
            "connectivity prior rather than current visual evidence. Preserve the exact N1 CoT field set while "
            "making the relevant Topo-Memory evidence explicit inside those fields. For object_goal, "
            "target_localization_and_key_context must state the node-semantic hypothesis and path_planning must "
            "state the next connected-node or within-node search step. Cite topology node IDs only inside CoT; "
            "Each topology node may include goal_search records. These are current-task evidence, not fixed map "
            "semantics: no_target_observed means the named goal was not seen during that local search attempt, while "
            "target_confirmed is positive evidence. Use attempt_count and evidence to avoid repeating the same search "
            "without new information, but do not infer permanent object absence from one observation. "
            "When an object target may be visible, target_localization_and_key_context must describe current "
            "visibility, candidate camera view, appearance, nearby landmarks or spatial relations, and likely "
            "distractors that the current observation rules out. These details will guide a separate current-frame "
            "grounding Skill, so describe only evidence supported by the current observation. "
            "Do not add extra CoT fields. Topology alone cannot justify either grounding query. Do not output "
            "topology IDs as query labels, locations, viewpoint actions, completion decisions, controller "
            "actions, or verification."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in slow_input.reference_memory:
            content.extend(
                image_ref_content(
                    (
                        f"HISTORICAL_REFERENCE_MEMORY_{frame.slot + 1}_"
                        f"OBSERVATION_{frame.observation_id or 'unknown'}_FRONT_"
                        "NO_CURRENT_COORDINATES"
                    ),
                    frame.front_image_ref,
                )
            )
        reference = slow_input.reference_observation
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are the ABot-N1 Slow System. Consume reference front-view memory, the synchronized "
                    "single-frame current_view_audit, task specification g, the previous Slow decision, and the optional "
                    "bounded ABot-N0 semantic topology prior. In one strict JSON response, emit the task-specific "
                    "CoT plus one coordinate-free affordance query and an optional target query. Never localize "
                    "pixels or bounding boxes."
                ),
            ),
            ChatMessage(role="user", content=content),
        ]
        semantic_decision, response = await self._complete(
            messages,
            slow_input.task_specification.task_type,
            previous_decision=slow_input.previous_decision,
            previous_observation_id=self.session.previous_decision_observation_id,
            current_observation_id=reference.observation_id,
        )
        binding_started = time.monotonic()
        semantic_decision, binding_verification = await self._verify_view_bindings(
            reference,
            semantic_decision,
        )
        return (
            slow_input,
            semantic_decision,
            response,
            binding_verification,
            time.monotonic() - binding_started,
        )

    async def _verify_view_bindings(
        self,
        reference: SlowBrainReferenceObservation,
        decision: NavigationSlowBrainSemanticOutput,
    ) -> tuple[NavigationSlowBrainSemanticOutput, SlowBrainViewBindingVerification]:
        queries = [decision.affordance_query]
        if (
            decision.target_query is not None
            and decision.target_query.preferred_view is not None
        ):
            queries.append(decision.target_query)
        query_payload = [
            {
                "role": query.role,
                "label": query.label,
                "proposed_view": query.preferred_view,
                "visual_description_hypothesis": query.visual_description,
                "excluded_labels": query.excluded_labels,
            }
            for query in queries
        ]
        expected = {(query.role, query.label): query for query in queries}

        async def verify_one_view(
            view: str,
            ref: str,
        ) -> SlowBrainViewBindingVerification:
            prompt = (
                f"Verify the Slow-selected semantic anchors using only the robot-{view.upper()} "
                f"camera image from current observation {reference.observation_id}. Queries: "
                f"{json.dumps(query_payload, ensure_ascii=False, separators=(',', ':'))}. "
                "The Slow System has already decided which semantic anchors matter. Do not choose a "
                "new route, object, or traversable region. For each unchanged role and label, set "
                f"visually_supported=true and verified_view={view!r} only when that exact anchor is "
                "visibly supported in this one image. Otherwise set visually_supported=false, "
                "verified_view=null, and confidence=0. The proposed view and description are hypotheses, "
                "not evidence. confidence measures current-image support. visual_description must be a "
                "concise local appearance and image-relative-position description, without task, topology, "
                "history, route rationale, or another camera name. Return every supplied query exactly once."
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            content.extend(
                image_ref_content(
                    f"CURRENT_OBSERVATION_{reference.observation_id}_{view.upper()}_ONLY",
                    ref,
                )
            )
            response = await self.provider.complete(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            f"You are auditing only the robot-{view.upper()} camera. Verify whether "
                            "each unchanged semantic anchor is visibly supported in this single image."
                        ),
                    ),
                    ChatMessage(role="user", content=content),
                ],
                response_schema=SlowBrainViewBindingVerification.model_json_schema(),
            )
            if not isinstance(response.parsed_json, dict):
                raise ValueError("Slow view binding verifier did not return one JSON object")
            verification = SlowBrainViewBindingVerification.model_validate(
                response.parsed_json
            )
            received_keys = {
                (binding.role, binding.label) for binding in verification.bindings
            }
            if received_keys != set(expected):
                raise ValueError("Slow view binding verifier changed or omitted a semantic query")
            if any(
                binding.visually_supported and binding.verified_view != view
                for binding in verification.bindings
            ):
                raise ValueError("single-view verifier returned a different camera identity")
            return verification

        per_view = await asyncio.gather(
            *(
                verify_one_view(view, ref)
                for view, ref in (
                    ("left", reference.left_image_ref),
                    ("front", reference.front_image_ref),
                    ("right", reference.right_image_ref),
                )
            )
        )
        candidates: dict[tuple[str, str], list[SlowBrainViewBinding]] = {
            key: [] for key in expected
        }
        for per_view_verification in per_view:
            for binding in per_view_verification.bindings:
                if binding.visually_supported:
                    candidates[(binding.role, binding.label)].append(binding)
        selected_bindings: list[SlowBrainViewBinding] = []
        for key, query in expected.items():
            supported = candidates[key]
            if supported:
                selected_bindings.append(
                    max(
                        supported,
                        key=lambda binding: (
                            binding.confidence,
                            binding.verified_view == query.preferred_view,
                        ),
                    )
                )
            else:
                selected_bindings.append(
                    SlowBrainViewBinding(
                        role=query.role,
                        label=query.label,
                        visually_supported=False,
                        verified_view=None,
                        confidence=0.0,
                        visual_description="",
                        evidence_summary=(
                            "No synchronized current camera supports this anchor."
                        ),
                    )
                )
        verification = SlowBrainViewBindingVerification(bindings=selected_bindings)
        received = {(item.role, item.label): item for item in verification.bindings}

        affordance_binding = received[
            (decision.affordance_query.role, decision.affordance_query.label)
        ]
        if not affordance_binding.visually_supported:
            raise ValueError(
                "Slow-selected affordance is unsupported in the synchronized current views"
            )
        affordance = decision.affordance_query.model_copy(
            update={
                "preferred_view": affordance_binding.verified_view,
                "visual_description": affordance_binding.visual_description,
            }
        )
        target = decision.target_query
        if target is not None and target.preferred_view is not None:
            target_binding = received[(target.role, target.label)]
            target = target.model_copy(
                update={
                    "preferred_view": (
                        target_binding.verified_view
                        if target_binding.visually_supported
                        else None
                    ),
                    "visual_description": target_binding.visual_description,
                }
            )
        return (
            decision.model_copy(
                update={"affordance_query": affordance, "target_query": target}
            ),
            verification,
        )

    def _build_input(
        self,
        observation: Observation,
        delegation: AgentOSSkillDelegation,
    ) -> NavigationSlowBrainInput:
        specification = delegation.task_specification
        task_changed = self.session.bind_task(specification)
        handoff = self.session.handoff or {}
        selected = handoff.get("selected_keyframes") or []
        if not selected:
            selected = [
                {"front_image_ref": ref}
                for ref in handoff.get("selected_keyframe_refs") or []
            ]
        memory = [
            SlowBrainReferenceMemoryFrame(
                slot=index,
                front_image_ref=str(frame["front_image_ref"]),
                observation_id=(
                    str(frame["observation_id"])
                    if frame.get("observation_id")
                    else None
                ),
            )
            for index, frame in enumerate(([] if task_changed else selected)[-4:])
            if isinstance(frame, dict) and frame.get("front_image_ref")
        ]
        return NavigationSlowBrainInput(
            reference_memory=memory,
            reference_observation=SlowBrainReferenceObservation(
                observation_id=observation.trace.observation_id,
                left_image_ref=observation.rgb_refs["left"],
                front_image_ref=observation.rgb_refs["front"],
                right_image_ref=observation.rgb_refs["right"],
            ),
            task_specification=specification,
            previous_decision=self.session.previous_decision,
            topology_context=_slow_topology_context(handoff, delegation),
        )

    async def _complete(
        self,
        messages: list[ChatMessage],
        task_type: str,
        *,
        previous_decision: NavigationSlowBrainOutput | None,
        previous_observation_id: str | None,
        current_observation_id: str,
    ) -> tuple[
        NavigationSlowBrainSemanticOutput,
        LLMResponse,
    ]:
        response_model = {
            "object_goal": ObjectGoalSlowBrainSemanticOutput,
            "instruction_following": InstructionFollowingSlowBrainSemanticOutput,
            "point_goal": PixelOnlySlowBrainSemanticOutput,
        }[task_type]
        last_error = "unknown Slow-System protocol error"
        for attempt in range(3):
            response = await self.provider.complete(
                messages,
                response_schema=response_model.model_json_schema(),
            )
            if isinstance(response.parsed_json, dict):
                try:
                    typed_output = response_model.model_validate(response.parsed_json)
                    semantic_decision = NavigationSlowBrainSemanticOutput.model_validate(
                        typed_output.model_dump(mode="json")
                    )
                    self._validate_semantic_output(semantic_decision, task_type)
                    return semantic_decision, response
                except (ValidationError, ValueError) as exc:
                    last_error = str(exc)
            else:
                last_error = "response did not contain one complete JSON object"
            if attempt < 2:
                messages.extend(
                    [
                        ChatMessage(role="assistant", content=response.content[-800:] or "{}"),
                        ChatMessage(
                            role="user",
                            content=(
                                "Correct the Slow-System output. Emit one coordinate-free affordance query, "
                                "always emit target_query when the task explicitly names a physical target; otherwise "
                                "keep target_query null, "
                                "and use only the task-specific CoT fields. Do not emit coordinates or bboxes. "
                                f"The answer must be grounded in current observation {current_observation_id}; "
                                "historical coordinates are invalid. Preserve prior semantic state only when the "
                                "current evidence independently supports that no meaningful state changed. "
                                f"Previous error: {last_error}"
                            ),
                        ),
                    ]
                )
        raise ValueError(f"Navigation Slow System failed after 3 attempts: {last_error}")

    @staticmethod
    def _validate_semantic_output(
        output: NavigationSlowBrainSemanticOutput,
        task_type: str,
    ) -> None:
        if task_type == "object_goal":
            if not isinstance(output.cot, ObjectGoalCoT):
                raise ValueError("object_goal requires localization/path-planning CoT")
        elif task_type == "instruction_following":
            if not isinstance(output.cot, InstructionFollowingCoT):
                raise ValueError("instruction_following requires progress CoT")
        elif output.cot is not None:
            raise ValueError(f"{task_type} does not emit a natural-language CoT in ABot-N1")
        if task_type == "point_goal" and output.target_query is not None:
            raise ValueError("point_goal never emits an ABot-N1 target query")

    @staticmethod
    def _validate_output(output: NavigationSlowBrainOutput, task_type: str) -> None:
        if task_type == "object_goal":
            if not isinstance(output.cot, ObjectGoalCoT):
                raise ValueError("object_goal requires localization/path-planning CoT")
        elif task_type == "instruction_following":
            if not isinstance(output.cot, InstructionFollowingCoT):
                raise ValueError("instruction_following requires progress CoT")
        elif output.cot is not None:
            raise ValueError(f"{task_type} does not emit a natural-language CoT in ABot-N1")
        target = output.target_pixel.selected()
        if task_type == "point_goal" and target is not None:
            raise ValueError("point_goal never emits an ABot-N1 target pixel")

    @staticmethod
    def _image_size(observation: Observation) -> tuple[int, int]:
        ref = observation.rgb_refs.get("front") or next(iter(observation.rgb_refs.values()))
        if ref.startswith("mock://"):
            return 640, 480
        with Image.open(Path(ref)) as image:
            return image.size


class AgentOSHierarchicalNavigationSkill:
    """Navigation Skill with point, object, and instruction-following capabilities."""

    name = "NavigationSkill"
    descriptor = SkillDescriptor(
        name=name,
        description=(
            "Execute one point_goal, object_goal, or instruction_following task while retaining its complete "
            "subgoal: known points use deterministic navigation, object search uses Slow/Grounding/Fast, and "
            "natural-language route following uses the isolated OmniNav capability."
        ),
        input_schema={"type": "object", "required": ["delegation", "source_observation_id"]},
        output_schema={"type": "object"},
        required_modalities=["rgb", "depth", "pose", "map"],
        supported_embodiments=["habitat-gs", "ros2"],
        risk_level=RiskLevel.MEDIUM,
        side_effects=["moves_robot"],
        default_timeout_s=300.0,
    )

    def __init__(
        self,
        provider: LLMProvider,
        service: AgentOSHierarchicalService,
        session: N1NavigationSkillSession,
        *,
        max_local_cycles: int | None = None,
        visual_grounding_skill: VisualGroundingSkill | None = None,
        omninav_skill: PersistentOmniNavSkill | None = None,
    ) -> None:
        if max_local_cycles is not None and max_local_cycles < 1:
            raise ValueError("max_local_cycles must be positive")
        self.service = service
        self.session = session
        self.max_local_cycles = max_local_cycles
        self.visual_grounding_skill = visual_grounding_skill
        self.omninav_skill = omninav_skill
        self.slow_brain = QwenNavigationSlowBrain(provider, session)

    @staticmethod
    def main_agent_handoff(result: SkillResult) -> dict[str, Any]:
        """Expose bounded post-Skill state without leaking private execution history."""
        return compress_navigation_handoff(result.outputs or {})

    async def run(
        self,
        request: SkillRequest,
        observation: Observation,
        context: SkillContext,
    ) -> SkillResult:
        context.cancellation.raise_if_cancelled()
        delegation = AgentOSSkillDelegation.model_validate(request.inputs.get("delegation"))
        source_id = str(request.inputs.get("source_observation_id") or "")
        if source_id != observation.trace.observation_id:
            raise ValueError("Skill delegation source observation is stale")

        prepared = await asyncio.to_thread(
            self.service.prepare_agentic_plan,
            {
                "observation_id": observation.trace.observation_id,
                "planner_output": delegation.planner_output.model_dump(mode="json"),
                "replace_active_plan": bool(request.inputs.get("replace_active_plan")),
            },
        )
        self._merge_handoff(prepared)
        current_observation = _packet_observation(prepared) or observation
        if delegation.task_specification.task_type == "instruction_following":
            return await self._run_instruction_following(
                request,
                current_observation,
                delegation,
                context,
            )
        route_summaries: list[str] = []
        route_evidence: list[Evidence] = []
        route_distance_m = 0.0
        latest_route_packet: dict[str, Any] | None = None
        for route_hop in range(8):
            node_target = (
                _topology_point_goal(prepared)
                if delegation.task_specification.task_type == "point_goal"
                else None
            )
            if node_target is None:
                break
            await context.emit_progress(
                request,
                f"PointGoal route hop {route_hop + 1} to topology node {node_target}.",
                stage="fast_system",
                operation="point_goal",
                status="running",
                topology_node_id=node_target,
            )
            fast_started = time.monotonic()
            route_packet = await asyncio.to_thread(
                self.service.navigate_to_known_location,
                {
                    "slow_step_id": f"point-{uuid.uuid4().hex[:12]}",
                    "topology_node_id": node_target,
                    "observation_id": current_observation.trace.observation_id,
                    "semantic_map": _topology_semantic_map(prepared),
                    "goal_tolerance_m": _topology_goal_tolerance(prepared),
                },
            )
            fast_duration_s = time.monotonic() - fast_started
            self._merge_handoff(route_packet)
            result = route_packet.get("result") or {}
            telemetry = self._telemetry(route_packet)
            route_distance_m += telemetry.progress_m
            latest_route_packet = route_packet
            route_summary = (
                f"PointGoal node={node_target}; reached={telemetry.goal_reached}; "
                f"state={telemetry.state}; progress={telemetry.progress_m:.2f}m; "
                f"termination={telemetry.termination_reason}; "
                f"next_observation={route_packet.get('next_observation_id')}"
            )
            route_packet["execution_mode"] = "topology_node_point_goal"
            route_packet["fast_system_latency_s"] = round(fast_duration_s, 3)
            route_packet["fast_system_contract"] = {
                "implementation": "topology_point_goal_astar_habitat_steps",
                "topology_node_id": node_target,
                "task_specification": delegation.task_specification.model_dump(mode="json"),
            }
            route_summaries.append(route_summary)
            await context.emit_progress(
                request,
                f"PointGoal execution to {node_target} returned {telemetry.state}.",
                stage="fast_system",
                operation="point_goal",
                status="completed",
                topology_node_id=node_target,
                duration_s=round(fast_duration_s, 3),
                goal_reached=telemetry.goal_reached,
                progress_m=round(telemetry.progress_m, 3),
            )
            route_evidence.extend(self._evidence(route_packet))
            succeeded = bool(route_packet.get("ok", False)) and telemetry.goal_reached
            next_observation = _packet_observation(route_packet)
            if not succeeded or next_observation is None:
                self.session.skill_runs += 1
                return SkillResult(
                    trace=request.trace,
                    skill_name=self.name,
                    status=SkillStatus.RETRYABLE_FAILURE,
                    summary="; ".join(route_summaries),
                    evidence=route_evidence,
                    outputs=route_packet,
                    compressed_history="; ".join(route_summaries),
                    error=str(route_packet.get("error") or "PointGoal route hop failed"),
                    stop_reason=(
                        str(result.get("termination_reason"))
                        if result.get("termination_reason")
                        else None
                    ),
                )
            current_observation = next_observation
            prepared = await asyncio.to_thread(
                self.service.prepare_agentic_plan,
                {
                    "observation_id": current_observation.trace.observation_id,
                    "planner_output": delegation.planner_output.model_dump(mode="json"),
                    "replace_active_plan": True,
                },
            )
            self._merge_handoff(prepared)
        else:
            self.session.skill_runs += 1
            return SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=SkillStatus.RETRYABLE_FAILURE,
                summary="; ".join(route_summaries),
                evidence=route_evidence,
                outputs=prepared,
                compressed_history="; ".join(route_summaries),
                error="Topology route exceeded eight connected-node hops.",
                stop_reason="topology_route_hop_limit",
            )

        if route_summaries and delegation.task_specification.task_type == "point_goal":
            self.session.skill_runs += 1
            summary = "; ".join(route_summaries)
            route_packet = latest_route_packet or {}
            outputs = {
                **prepared,
                "result": route_packet.get("result") or {},
                "execution_telemetry": route_packet.get("execution_telemetry"),
                "execution_mode": route_packet.get("execution_mode"),
                "fast_system_contract": route_packet.get("fast_system_contract"),
                "distance_m": route_distance_m,
            }
            return SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=SkillStatus.SUCCESS,
                summary=summary,
                evidence=route_evidence,
                outputs=outputs,
                compressed_history=summary,
                stop_reason="topology_point_goal_reached",
            )

        packet: dict[str, Any] = {}
        cycle_summaries: list[str] = []
        scheduler_events: list[dict[str, Any]] = []
        scheduler_mode = "sequential"
        zero_progress_chunks = 0
        accepted_slow_cycles = 0

        active: _SlowDecisionBundle | None = None
        while self.max_local_cycles is None or accepted_slow_cycles < self.max_local_cycles:
            context.cancellation.raise_if_cancelled()
            try:
                active = await self._infer_slow(
                    request,
                    current_observation,
                    delegation,
                    context,
                )
            except (ValidationError, ValueError) as exc:
                self.session.skill_runs += 1
                summary = f"Slow/Grounding decision needs replanning: {type(exc).__name__}: {exc}"
                return SkillResult(
                    trace=request.trace,
                    skill_name=self.name,
                    status=SkillStatus.RETRYABLE_FAILURE,
                    summary=summary,
                    compressed_history=summary,
                    error=summary,
                    stop_reason="slow_decision_invalid",
                )
            active_audit_ref = self._accept_slow_decision(
                request,
                active,
                scheduler_mode=scheduler_mode,
            )
            accepted_slow_cycles += 1
            scheduler_events.append(
                {
                    "event": "slow_applied",
                    "source_observation_id": active.source_observation.trace.observation_id,
                    "latency_s": round(active.latency_s, 3),
                }
            )
            await context.emit_progress(
                request,
                f"ABot-N1 serial decision {accepted_slow_cycles}: observe, decide, execute.",
                local_cycle=accepted_slow_cycles,
                slow_brain_input=active.slow_input.model_dump(mode="json"),
                slow_brain_semantic_decision=active.semantic_decision.model_dump(
                    mode="json"
                ),
                slow_view_binding_verification=(
                    active.view_binding_verification.model_dump(mode="json")
                ),
                slow_brain_output=active.slow_output.model_dump(mode="json"),
                slow_brain_grounding=active.grounding.model_dump(mode="json"),
                visual_grounding_skill_result=(
                    active.visual_grounding_result.model_dump(mode="json")
                    if active.visual_grounding_result is not None
                    else None
                ),
                slow_inference_latency_s=round(active.slow_latency_s, 3),
                slow_view_binding_latency_s=round(
                    active.view_binding_latency_s, 3
                ),
                visual_grounding_latency_s=round(active.grounding_latency_s, 3),
                decision_pipeline_latency_s=round(active.latency_s, 3),
                slow_fast_mode=scheduler_mode,
            )

            fast_input = NavigationFastSystemInput(
                current_memory=active.slow_input.reference_memory,
                current_observation=_reference_observation(current_observation),
                slow_decision=active.slow_output,
                slow_reference_observation=active.slow_input.reference_observation,
                task_specification=active.slow_input.task_specification,
            )
            semantic_boundary = self._semantic_boundary_reason(
                delegation.task_specification.task_type,
                active.slow_input.previous_decision,
                active.slow_output,
            )
            if semantic_boundary is not None:
                packet = self._semantic_boundary_packet(
                    current_observation,
                    semantic_boundary,
                )
                self._attach_slow_contract(
                    packet,
                    active,
                    active_audit_ref,
                    fast_input,
                    scheduler_mode,
                )
                cycle_summaries.append(
                    f"Semantic boundary={semantic_boundary}; no next-anchor execution; "
                    f"observation={current_observation.trace.observation_id}"
                )
                scheduler_events.append(
                    {
                        "event": "control_returned",
                        "reason": semantic_boundary,
                        "source_observation_id": current_observation.trace.observation_id,
                    }
                )
                break
            if (
                active.slow_output.affordance_pixel.selected() is None
                and active.slow_output.target_pixel.selected() is None
            ):
                packet = self._no_pixel_goal_packet(current_observation)
            else:
                try:
                    await context.emit_progress(
                        request,
                        "Fast System is converting the selected Pixel Goal into a path and controls.",
                        stage="fast_system",
                        operation="pixel_goal",
                        status="running",
                    )
                    fast_started = time.monotonic()
                    packet = await asyncio.to_thread(
                        self._execute_fast,
                        active.slow_output,
                        active.grounding,
                        active.source_observation,
                        delegation,
                        allow_stale_slow_reference=False,
                    )
                    fast_duration_s = time.monotonic() - fast_started
                    packet["fast_system_latency_s"] = round(fast_duration_s, 3)
                except ValueError as exc:
                    self.session.skill_runs += 1
                    summary = f"Fast System rejected the grounded pixel goal: {exc}"
                    return SkillResult(
                        trace=request.trace,
                        skill_name=self.name,
                        status=SkillStatus.RETRYABLE_FAILURE,
                        summary=summary,
                        evidence=route_evidence,
                        compressed_history=summary,
                        error=summary,
                        stop_reason="fast_goal_invalid",
                    )
            self._attach_slow_contract(
                packet,
                active,
                active_audit_ref,
                fast_input,
                scheduler_mode,
            )
            cycle_summaries.append(self._summary(active.slow_output, packet))
            next_observation = _packet_observation(packet)
            execution = packet.get("result") or {}
            telemetry = self._telemetry(packet)
            if packet.get("fast_system_latency_s") is not None:
                await context.emit_progress(
                    request,
                    f"Fast System returned {telemetry.state} after Pixel Goal execution.",
                    stage="fast_system",
                    operation="pixel_goal",
                    status="completed",
                    duration_s=packet.get("fast_system_latency_s"),
                    goal_reached=telemetry.goal_reached,
                    progress_m=round(telemetry.progress_m, 3),
                    executed_role=packet.get("anchor_role"),
                )

            if (
                telemetry.motion_attempted
                and not telemetry.goal_reached
                and telemetry.progress_m < 0.05
            ):
                zero_progress_chunks += 1
            else:
                zero_progress_chunks = 0
            if zero_progress_chunks >= 2:
                execution.update(stalled=True, termination_reason="repeated_zero_progress")
                packet.update(
                    ok=False,
                    status="stalled",
                    error="Fast System repeated a local goal without measurable motion.",
                    result=execution,
                )
                cycle_summaries[-1] = self._summary(active.slow_output, packet)

            if next_observation is None:
                packet.update(
                    ok=False,
                    status="failed",
                    error="Fast System did not return a fresh observation.",
                )
            else:
                current_observation = next_observation
            await self._record_topology_search(
                packet,
                delegation,
                active,
                telemetry,
            )
            self._merge_handoff(packet)

            return_reason = self._return_control_reason(
                delegation.task_specification.task_type,
                active.slow_output,
                packet,
                topology_context=active.slow_input.topology_context,
            )
            if return_reason is not None or zero_progress_chunks >= 2:
                return_reason = return_reason or "repeated_zero_progress"
                packet["return_control_reason"] = return_reason
                scheduler_events.append(
                    {
                        "event": "control_returned",
                        "reason": return_reason,
                        "source_observation_id": current_observation.trace.observation_id,
                    }
                )
                break
            if next_observation is None:
                break
        assert active is not None
        slow_output = active.slow_output
        self.session.skill_runs += 1
        packet["local_slow_cycles"] = accepted_slow_cycles
        packet["slow_cycle_summaries"] = cycle_summaries[-4:]
        packet["slow_fast_scheduler"] = {
            "mode": scheduler_mode,
            "policy": "one_observation_one_decision_one_execution",
            "events": scheduler_events[-12:],
        }
        packet["semantic_completion"] = self._object_goal_completion(
            delegation.task_specification.task_type,
            active,
            packet,
        )

        summary = (
            ("route=" + " | ".join(route_summaries) + "; ")
            if route_summaries
            else ""
        ) + (
            f"slow_cycles={accepted_slow_cycles}; mode={scheduler_mode}; "
            f"{cycle_summaries[-1]}"
        )
        succeeded = bool(packet.get("ok", False)) and packet.get("status") != "failed"
        result = packet.get("result") or {}
        return SkillResult(
            trace=request.trace,
            skill_name=self.name,
            status=(SkillStatus.SUCCESS if succeeded else SkillStatus.RETRYABLE_FAILURE),
            summary=summary,
            evidence=[*route_evidence, *self._evidence(packet)],
            outputs=packet,
            compressed_history=summary,
            error=None if succeeded else str(packet.get("error") or "Fast System rollout failed"),
            stop_reason=(
                str(result.get("termination_reason"))
                if result.get("termination_reason")
                else None
            ),
        )

    async def _run_instruction_following(
        self,
        request: SkillRequest,
        observation: Observation,
        delegation: AgentOSSkillDelegation,
        context: SkillContext,
    ) -> SkillResult:
        if self.omninav_skill is None:
            summary = "NavigationSkill instruction_following capability is unavailable: OmniNav is not configured."
            return SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=SkillStatus.RETRYABLE_FAILURE,
                summary=summary,
                compressed_history=summary,
                error=summary,
                stop_reason="omninav_unavailable",
            )

        try:
            initial_bundle = await self._infer_slow(
                request,
                observation,
                delegation,
                context,
            )
            initial_guidance = self._omni_guidance(initial_bundle)
        except (ValidationError, ValueError) as exc:
            summary = f"Slow/Grounding could not establish visible Omni guidance: {type(exc).__name__}: {exc}"
            return SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=SkillStatus.RETRYABLE_FAILURE,
                summary=summary,
                compressed_history=summary,
                error=summary,
                stop_reason="slow_guidance_invalid",
            )

        async def refresh_guidance(current: Observation) -> OmniNavGuidance:
            bundle = await self._infer_slow(
                request,
                current,
                delegation,
                context,
            )
            return self._omni_guidance(bundle)

        async def accept_guidance(guidance: OmniNavGuidance) -> None:
            bundle = guidance.payload
            if not isinstance(bundle, _SlowDecisionBundle):
                raise ValueError("Omni guidance is missing its audited Slow decision")
            audit_ref = self._accept_slow_decision(
                request,
                bundle,
                scheduler_mode="asynchronous_omni_cached_guidance",
            )
            await context.emit_progress(
                request,
                "ABot-N1 Slow guidance accepted for asynchronous OmniNav consumption.",
                slow_guidance=guidance.instruction,
                slow_guidance_terminal_reason=guidance.terminal_reason,
                slow_brain_audit_ref=audit_ref,
                source_observation_id=bundle.source_observation.trace.observation_id,
                slow_inference_latency_s=round(bundle.slow_latency_s, 3),
                visual_grounding_latency_s=round(bundle.grounding_latency_s, 3),
            )

        try:
            outcome = await self.omninav_skill.run_guided(
                request,
                observation,
                context,
                initial_guidance=initial_guidance,
                refresh_guidance=refresh_guidance,
                on_guidance_accepted=accept_guidance,
            )
        except (ValidationError, ValueError) as exc:
            summary = f"Asynchronous Slow guidance refresh failed: {type(exc).__name__}: {exc}"
            return SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=SkillStatus.RETRYABLE_FAILURE,
                summary=summary,
                compressed_history=summary,
                error=summary,
                stop_reason="slow_guidance_refresh_invalid",
            )
        result = outcome.result
        terminal_reason = outcome.guidance.terminal_reason
        terminal_bundle = outcome.guidance.payload

        if terminal_reason == "target_ready" and isinstance(
            terminal_bundle, _SlowDecisionBundle
        ):
            object_delegation = self._target_handoff_delegation(delegation)
            try:
                packet = await asyncio.to_thread(
                    self._execute_fast,
                    terminal_bundle.slow_output,
                    terminal_bundle.grounding,
                    terminal_bundle.source_observation,
                    object_delegation,
                    allow_stale_slow_reference=False,
                )
            except ValueError as exc:
                summary = f"ObjectGoal target handoff rejected its pixel goal: {exc}"
                return result.model_copy(
                    update={
                        "status": SkillStatus.RETRYABLE_FAILURE,
                        "summary": summary,
                        "compressed_history": summary,
                        "error": summary,
                        "stop_reason": "target_handoff_invalid",
                    }
                )
            self._merge_handoff(packet)
            completion = self._object_goal_completion(
                "object_goal",
                terminal_bundle,
                packet,
            )
            target_summary = self._summary(terminal_bundle.slow_output, packet)
            outputs = {
                **result.outputs,
                **packet,
                "slow_brain_input": terminal_bundle.slow_input.model_dump(mode="json"),
                "slow_brain_output": terminal_bundle.slow_output.model_dump(mode="json"),
                "slow_brain_grounding": terminal_bundle.grounding.model_dump(mode="json"),
                "slow_guidance_source_observation_id": (
                    terminal_bundle.source_observation.trace.observation_id
                ),
                "navigation_task_type": "instruction_following",
                "navigation_capability": "slow_affordance_omni_then_object_goal",
                "omni_affordance_phase": result.outputs,
                "semantic_completion": completion,
            }
            target_succeeded = bool(completion and completion.get("satisfied"))
            summary = (
                f"Slow-guided Omni affordance phase completed; ObjectGoal target handoff: "
                f"{target_summary}"
            )
            return SkillResult(
                trace=request.trace,
                skill_name=self.name,
                status=(SkillStatus.SUCCESS if target_succeeded else SkillStatus.RETRYABLE_FAILURE),
                summary=summary,
                evidence=[*result.evidence, *self._evidence(packet)],
                outputs=outputs,
                compressed_history=summary,
                error=None if target_succeeded else "Terminal target was not reached.",
                stop_reason="target_pixel_reached" if target_succeeded else "target_handoff_incomplete",
                started_at=result.started_at,
                completed_at=utc_now(),
            )

        semantic_boundary = terminal_reason == "instruction_progress_boundary"
        outputs = {
            **result.outputs,
            "navigation_task_type": "instruction_following",
            "navigation_capability": "slow_affordance_omni_async",
            "semantic_completion": {
                "task_type": "instruction_following",
                "terminal_evidence": terminal_reason or result.stop_reason,
                "satisfied": False,
                "requires_skill_verifier": semantic_boundary,
            },
        }
        if terminal_reason == "no_visible_affordance":
            summary = "Slow System could not ground a clear visible affordance for OmniNav."
            return result.model_copy(
                update={
                    "status": SkillStatus.RETRYABLE_FAILURE,
                    "summary": summary,
                    "compressed_history": summary,
                    "outputs": outputs,
                    "error": summary,
                    "stop_reason": terminal_reason,
                }
            )
        return result.model_copy(
            update={
                "skill_name": self.name,
                "outputs": outputs,
                "status": (
                    SkillStatus.SUCCESS
                    if semantic_boundary
                    else result.status
                ),
            }
        )

    def _omni_guidance(self, bundle: _SlowDecisionBundle) -> OmniNavGuidance:
        previous = self.session.previous_decision
        semantic_boundary = self._semantic_boundary_reason(
            "instruction_following",
            previous,
            bundle.slow_output,
        )
        if semantic_boundary is not None:
            return OmniNavGuidance(
                instruction="Hold position for semantic progress verification.",
                source_observation=bundle.source_observation,
                payload=bundle,
                terminal_reason="instruction_progress_boundary",
            )

        target = bundle.grounding.target_grounding
        target_threshold = float(
            getattr(self.service, "target_confidence_threshold", 0.90)
        )
        if (
            target.view is not None
            and bool(target.bbox_normalized_1000)
            and target.confidence >= target_threshold
        ):
            return OmniNavGuidance(
                instruction=f"Hold for precise approach to the visible {target.label}.",
                source_observation=bundle.source_observation,
                payload=bundle,
                terminal_reason="target_ready",
            )

        affordance = bundle.grounding.affordance_grounding
        affordance_threshold = float(
            getattr(self.service, "anchor_confidence_threshold", 0.80)
        )
        if (
            affordance.view is None
            or not affordance.bbox_normalized_1000
            or affordance.confidence < affordance_threshold
            or not affordance.label
        ):
            return OmniNavGuidance(
                instruction="Hold position because no clear visible affordance is grounded.",
                source_observation=bundle.source_observation,
                payload=bundle,
                terminal_reason="no_visible_affordance",
            )
        return OmniNavGuidance(
            instruction=(
                f"Navigate toward the clearly visible {affordance.label} in the "
                f"{affordance.view.upper()} camera view. Stop at that local visual waypoint."
            ),
            source_observation=bundle.source_observation,
            payload=bundle,
        )

    @staticmethod
    def _target_handoff_delegation(
        delegation: AgentOSSkillDelegation,
    ) -> AgentOSSkillDelegation:
        specification = delegation.task_specification.model_copy(
            update={"task_type": "object_goal"}
        )
        return delegation.model_copy(update={"task_specification": specification})

    def _object_goal_completion(
        self,
        task_type: str,
        bundle: _SlowDecisionBundle,
        packet: dict[str, Any],
    ) -> dict[str, Any] | None:
        if task_type != "object_goal":
            return None
        telemetry = self._telemetry(packet)
        grounding = bundle.grounding.target_grounding
        threshold = float(
            getattr(self.service, "target_confidence_threshold", 0.90)
        )
        target_grounded = (
            grounding.view is not None
            and bool(grounding.bbox_normalized_1000)
            and grounding.confidence >= threshold
        )
        target_executed = packet.get("anchor_role") == "target"
        target_projection_limited = self._projection_depth_limited(packet)
        target_proxy_reached = (
            target_executed and telemetry.goal_reached and target_projection_limited
        )
        target_reached = (
            target_executed and telemetry.goal_reached and not target_projection_limited
        )
        return {
            "task_type": "object_goal",
            "required_terminal_role": "target",
            "target_label": grounding.label,
            "target_view": grounding.view,
            "target_confidence": grounding.confidence,
            "target_confidence_threshold": threshold,
            "target_grounded": target_grounded,
            "target_executed": target_executed,
            "target_projection_limited": target_projection_limited,
            "target_proxy_reached": target_proxy_reached,
            "target_reached": target_reached,
            "target_raw_depth_m": packet.get("raw_anchor_depth_m"),
            "target_max_depth_m": packet.get("anchor_max_depth_m"),
            "source_observation_id": bundle.source_observation.trace.observation_id,
            "post_observation_id": packet.get("next_observation_id"),
            "satisfied": target_grounded and target_reached,
        }

    async def _infer_slow(
        self,
        request: SkillRequest,
        observation: Observation,
        delegation: AgentOSSkillDelegation,
        context: SkillContext,
    ) -> _SlowDecisionBundle:
        started = time.monotonic()
        await context.emit_progress(
            request,
            "Slow Brain is reasoning from the global task, current observation, progress, and Topo-Memory.",
            stage="slow_brain",
            status="running",
            observation_id=observation.trace.observation_id,
        )
        (
            slow_input,
            semantic_decision,
            response,
            view_binding_verification,
            view_binding_latency_s,
        ) = await self.slow_brain.decide(request, observation, delegation)
        slow_finished = time.monotonic()
        await context.emit_progress(
            request,
            "Slow Brain produced the current CoT and semantic target/affordance queries; Visual Grounding is starting.",
            stage="visual_grounding",
            status="running",
            completed_stage="slow_brain",
            slow_inference_latency_s=round(slow_finished - started, 3),
            target_query=(
                semantic_decision.target_query.label
                if semantic_decision.target_query is not None
                else None
            ),
            affordance_query=semantic_decision.affordance_query.label,
        )
        visual_grounding_result = await self._ground_current_queries(
            request,
            observation,
            semantic_decision,
            context,
        )
        grounding_finished = time.monotonic()
        await context.emit_progress(
            request,
            "Visual Grounding localized the Slow-selected semantic queries in the bound current view.",
            stage="visual_grounding",
            status="completed",
            duration_s=round(grounding_finished - slow_finished, 3),
            decision_pipeline_latency_s=round(grounding_finished - started, 3),
        )
        grounding = self._grounding_from_batch(
            semantic_decision,
            visual_grounding_result,
        )
        slow_output = SlowPixelGoalAssembler.assemble(grounding)
        QwenNavigationSlowBrain._validate_output(
            slow_output,
            delegation.task_specification.task_type,
        )
        return _SlowDecisionBundle(
            source_observation=observation,
            slow_input=slow_input,
            semantic_decision=semantic_decision,
            view_binding_verification=view_binding_verification,
            slow_output=slow_output,
            response=response,
            grounding=grounding,
            visual_grounding_result=visual_grounding_result,
            slow_latency_s=slow_finished - started,
            view_binding_latency_s=view_binding_latency_s,
            grounding_latency_s=grounding_finished - slow_finished,
            latency_s=grounding_finished - started,
        )

    async def _ground_current_queries(
        self,
        request: SkillRequest,
        observation: Observation,
        semantic_decision: NavigationSlowBrainSemanticOutput,
        context: SkillContext,
    ) -> SkillResult:
        if self.visual_grounding_skill is None:
            raise ValueError(
                "VisualGroundingSkill is required to localize Slow-System semantic queries"
            )
        semantic_queries = [semantic_decision.affordance_query]
        if (
            semantic_decision.target_query is not None
            and semantic_decision.target_query.preferred_view is not None
        ):
            semantic_queries.append(semantic_decision.target_query)
        queries = [
            VisualGroundingQuery(
                role=query.role,
                query=query.label,
                constraints=VisualGroundingConstraints(
                    preferred_view=query.preferred_view,
                    visual_description=query.visual_description,
                    excluded_labels=query.excluded_labels,
                ),
                min_confidence=float(
                    getattr(
                        self.service,
                        (
                            "target_confidence_threshold"
                            if query.role == "target"
                            else "anchor_confidence_threshold"
                        ),
                        0.90 if query.role == "target" else 0.80,
                    )
                ),
            )
            for query in semantic_queries
        ]
        nested_request = SkillRequest(
            trace=observation.trace.model_copy(
                update={"skill_run_id": f"visual-grounding-{uuid.uuid4().hex[:12]}"}
            ),
            skill_name=self.visual_grounding_skill.name,
            command="Batch-ground the Slow System target and affordance queries.",
            success_conditions=[
                "Every returned bbox is grounded only in the supplied current observation."
            ],
            context_summary=(
                "Slow Brain supplied coordinate-free semantics; Grounding owns localization."
            ),
            inputs={
                "queries": [query.model_dump(mode="json") for query in queries],
                "candidate_views": ["left", "front", "right"],
                "max_results_per_query": 1,
            },
            timeout_s=min(request.timeout_s, 30.0),
        )
        return await self.visual_grounding_skill.run(
            nested_request,
            observation,
            context,
        )

    @staticmethod
    def _grounding_from_batch(
        semantic_decision: NavigationSlowBrainSemanticOutput,
        result: SkillResult,
    ) -> GroundedNavigationSlowBrainOutput:
        output = VisualGroundingBatchOutput.model_validate(result.outputs)
        results_by_role = {item.role: item for item in output.results}

        def grounded(role: str) -> SlowBrainVisualGrounding:
            query_result = results_by_role.get(role)
            if (
                result.status != SkillStatus.SUCCESS
                or not output.available
                or query_result is None
                or not query_result.detections
            ):
                return SlowBrainVisualGrounding(
                    label=None,
                    view=None,
                    bbox_normalized_1000=[],
                    confidence=0.0,
                    evidence_summary="",
                )
            detection = VisualDetection.model_validate(query_result.detections[0])
            evidence_summary = " ".join(
                (
                    detection.evidence_summary
                    or (
                        f"VisualGroundingSkill localized {detection.label!r} in current "
                        f"observation {output.observation_id}."
                    )
                ).split()
            )
            return SlowBrainVisualGrounding(
                label=query_result.query,
                view=detection.view,
                bbox_normalized_1000=detection.bbox_normalized_1000,
                confidence=detection.confidence,
                evidence_summary=evidence_summary[:240],
            )

        return GroundedNavigationSlowBrainOutput(
            cot=semantic_decision.cot,
            affordance_grounding=grounded("affordance"),
            target_grounding=grounded("target"),
        )

    def _accept_slow_decision(
        self,
        request: SkillRequest,
        bundle: _SlowDecisionBundle,
        *,
        scheduler_mode: str,
    ) -> str:
        audit_ref = self._write_slow_audit(
            request,
            bundle.source_observation,
            bundle.slow_input,
            bundle.semantic_decision,
            bundle.view_binding_verification,
            bundle.slow_output,
            bundle.grounding,
            bundle.response,
            visual_grounding_result=bundle.visual_grounding_result,
            invocation=self.session.decision_count + 1,
            slow_latency_s=bundle.slow_latency_s,
            view_binding_latency_s=bundle.view_binding_latency_s,
            grounding_latency_s=bundle.grounding_latency_s,
            pipeline_latency_s=bundle.latency_s,
            scheduler_mode=scheduler_mode,
        )
        self.session.record_slow_output(
            bundle.slow_output,
            source_observation_id=bundle.source_observation.trace.observation_id,
        )
        return audit_ref

    @staticmethod
    def _attach_slow_contract(
        packet: dict[str, Any],
        bundle: _SlowDecisionBundle,
        audit_ref: str,
        fast_input: NavigationFastSystemInput,
        scheduler_mode: str,
    ) -> None:
        packet["slow_brain_input"] = bundle.slow_input.model_dump(mode="json")
        packet["slow_brain_semantic_decision"] = bundle.semantic_decision.model_dump(
            mode="json"
        )
        packet["slow_view_binding_verification"] = (
            bundle.view_binding_verification.model_dump(mode="json")
        )
        packet["slow_brain_output"] = bundle.slow_output.model_dump(mode="json")
        packet["slow_brain_grounding"] = bundle.grounding.model_dump(mode="json")
        packet["visual_grounding_skill_result"] = (
            bundle.visual_grounding_result.model_dump(mode="json")
            if bundle.visual_grounding_result is not None
            else None
        )
        packet["slow_brain_model"] = bundle.response.model
        packet["slow_brain_latency_s"] = round(bundle.slow_latency_s, 3)
        packet["slow_view_binding_latency_s"] = round(
            bundle.view_binding_latency_s, 3
        )
        packet["visual_grounding_latency_s"] = round(bundle.grounding_latency_s, 3)
        packet["decision_pipeline_latency_s"] = round(bundle.latency_s, 3)
        packet["slow_brain_audit_ref"] = audit_ref
        packet["fast_system_input"] = fast_input.model_dump(mode="json")
        packet["fast_system_contract"] = {
            "implementation": "affordance_pixel_depth_astar_habitat_steps",
            "reference_observation_id": bundle.source_observation.trace.observation_id,
            "task_specification": bundle.slow_input.task_specification.model_dump(mode="json"),
            "cached_slow_decision": bundle.slow_output.model_dump(mode="json"),
            "pixel_semantics": (
                "target=explicit_task_object_only; affordance=semantic_intermediate_anchor"
            ),
            "scheduler_mode": scheduler_mode,
        }

    @staticmethod
    def _write_slow_audit(
        request: SkillRequest,
        observation: Observation,
        slow_input: NavigationSlowBrainInput,
        semantic_decision: NavigationSlowBrainSemanticOutput,
        view_binding_verification: SlowBrainViewBindingVerification,
        slow_output: NavigationSlowBrainOutput,
        grounded_output: GroundedNavigationSlowBrainOutput,
        response: LLMResponse,
        *,
        visual_grounding_result: SkillResult | None,
        invocation: int,
        slow_latency_s: float,
        view_binding_latency_s: float,
        grounding_latency_s: float,
        pipeline_latency_s: float,
        scheduler_mode: str,
    ) -> str:
        directory = Path(observation.rgb_refs["front"]).parent
        path = directory / (
            f"abot_n1_slow_{request.trace.skill_run_id}_{invocation:03d}.json"
        )
        path.write_text(
            json.dumps(
                {
                    "trace": request.trace.model_dump(mode="json"),
                    "input": slow_input.model_dump(mode="json"),
                    "semantic_decision": semantic_decision.model_dump(mode="json"),
                    "view_binding_verification": view_binding_verification.model_dump(
                        mode="json"
                    ),
                    "output": slow_output.model_dump(mode="json"),
                    "grounding": grounded_output.model_dump(mode="json"),
                    "visual_grounding_skill_result": (
                        visual_grounding_result.model_dump(mode="json")
                        if visual_grounding_result is not None
                        else None
                    ),
                    "model": response.model,
                    "slow_inference_latency_s": round(slow_latency_s, 3),
                    "slow_view_binding_latency_s": round(
                        view_binding_latency_s, 3
                    ),
                    "visual_grounding_latency_s": round(grounding_latency_s, 3),
                    "decision_pipeline_latency_s": round(pipeline_latency_s, 3),
                    "slow_fast_scheduler_mode": scheduler_mode,
                    "protocol": {
                        "n1_public_output": ["cot", "affordance_pixel", "target_pixel"],
                        "slow_semantic_output": [
                            "cot",
                            "affordance_query",
                            "target_query",
                        ],
                        "grounding_owner": "VisualGroundingSkill.batch_current_observation",
                        "topology_extension": "ABot-N0 prior expressed inside the task-specific N1 CoT fields",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(path)

    @staticmethod
    def _should_return_control(
        task_type: str,
        slow_output: NavigationSlowBrainOutput,
        packet: dict[str, Any],
    ) -> bool:
        return (
            AgentOSHierarchicalNavigationSkill._return_control_reason(
                task_type,
                slow_output,
                packet,
            )
            is not None
        )

    @staticmethod
    def _return_control_reason(
        task_type: str,
        slow_output: NavigationSlowBrainOutput,
        packet: dict[str, Any],
        *,
        topology_context: SlowBrainTopologyContext | None = None,
    ) -> str | None:
        del slow_output
        telemetry = AgentOSHierarchicalNavigationSkill._telemetry(packet)
        if telemetry.state in {"failed", "blocked", "stalled", "cancelled"}:
            return telemetry.termination_reason or telemetry.state
        if not telemetry.goal_reached:
            return None
        route_plan = packet.get("semantic_route_plan") or {}
        guidance = route_plan.get("regional_guidance") or {}
        before_node = (
            topology_context.current_topology_node_id
            if topology_context is not None
            else None
        )
        after_node = guidance.get("current_topology_node_id")
        if before_node and after_node and before_node != after_node:
            return f"topology_region_changed:{before_node}->{after_node}"
        reached_region = bool(guidance.get("target_region_reached"))
        previously_in_region = bool(
            topology_context is not None and topology_context.target_region_reached
        )
        if reached_region and not previously_in_region:
            return "target_topology_region_reached"
        if task_type == "point_goal" and reached_region:
            return "point_goal_region_reached"
        if packet.get("anchor_role") == "target":
            if AgentOSHierarchicalNavigationSkill._projection_depth_limited(packet):
                return None
            return "target_pixel_reached"
        if packet.get("anchor_role") == "affordance":
            return "affordance_execution_chunk_reached"
        return None

    @staticmethod
    def _projection_depth_limited(packet: dict[str, Any]) -> bool:
        generated = packet.get("generated_point") or {}
        return bool(
            packet.get("anchor_depth_limited")
            or (isinstance(generated, dict) and generated.get("depth_limited"))
        )

    @staticmethod
    def _telemetry(packet: dict[str, Any]) -> EmbodimentExecutionTelemetry:
        telemetry = packet.get("execution_telemetry")
        if not isinstance(telemetry, dict):
            raise ValueError(
                "Embodiment Adapter result is missing execution_telemetry"
            )
        return EmbodimentExecutionTelemetry.model_validate(telemetry)

    @staticmethod
    def _semantic_boundary_reason(
        task_type: str,
        previous_output: NavigationSlowBrainOutput | None,
        current_output: NavigationSlowBrainOutput,
    ) -> str | None:
        if task_type != "instruction_following" or previous_output is None:
            return None
        previous_cot = previous_output.cot
        current_cot = current_output.cot
        if not isinstance(previous_cot, InstructionFollowingCoT) or not isinstance(
            current_cot, InstructionFollowingCoT
        ):
            return None
        if (
            previous_cot.finished_sub_instruction.strip()
            != current_cot.finished_sub_instruction.strip()
        ):
            return "finished_sub_instruction_advanced"
        return None

    @staticmethod
    def _semantic_boundary_packet(
        observation: Observation,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "completed",
            "phase": "semantic_boundary",
            "return_control_reason": reason,
            "result": {
                "controller": "none",
                "reached": False,
                "distance_m": 0.0,
                "stalled": False,
                "termination_reason": reason,
            },
            "execution_telemetry": EmbodimentExecutionTelemetry(
                state="idle",
                goal_reached=False,
                motion_attempted=False,
                progress_m=0.0,
                stalled=False,
                termination_reason=reason,
                source="agentos-semantic-boundary",
            ).model_dump(mode="json"),
            "post_observation": observation.model_dump(mode="json"),
            "next_observation_id": observation.trace.observation_id,
        }

    def _merge_handoff(self, prepared: dict[str, Any]) -> None:
        previous = self.session.handoff or {}
        current = compress_navigation_handoff(prepared)
        if not any((current.get("execution_summary") or {}).values()):
            current["execution_summary"] = previous.get("execution_summary")
        if not current.get("selected_keyframe_refs"):
            current["selected_keyframe_refs"] = previous.get("selected_keyframe_refs", [])
        if not current.get("selected_keyframes"):
            current["selected_keyframes"] = previous.get("selected_keyframes", [])
        if not current.get("slow_context"):
            current["slow_context"] = previous.get("slow_context")
        if not current.get("semantic_route_plan"):
            current["semantic_route_plan"] = previous.get("semantic_route_plan")
        self.session.handoff = current

    async def _record_topology_search(
        self,
        packet: dict[str, Any],
        delegation: AgentOSSkillDelegation,
        bundle: _SlowDecisionBundle,
        telemetry: EmbodimentExecutionTelemetry,
    ) -> None:
        recorder = getattr(self.service, "record_topology_search", None)
        observation_id = str(packet.get("next_observation_id") or "")
        if recorder is None or not observation_id:
            return
        role = str(packet.get("anchor_role") or "")
        target = bundle.grounding.target_grounding
        if role == "target" and telemetry.goal_reached:
            status = "target_observed"
        elif target.view is not None and target.confidence >= 0.90:
            status = "target_observed"
        elif telemetry.termination_reason == "no_visible_pixel_goal":
            status = "no_target_observed"
        elif packet.get("status") in {"blocked", "failed", "stalled"} or telemetry.state in {
            "blocked",
            "failed",
            "stalled",
        }:
            status = "blocked"
        else:
            status = "no_target_observed"
        object_goal = delegation.planner_output.object_goal.strip()
        goal = object_goal or delegation.task_specification.goal
        evidence = (
            f"role={role or 'none'}; state={telemetry.state}; "
            f"reached={telemetry.goal_reached}; termination={telemetry.termination_reason}; "
            f"target={target.label or 'none'}; confidence={target.confidence:.2f}"
        )
        update = await asyncio.to_thread(
            recorder,
            {
                "observation_id": observation_id,
                "goal": goal,
                "status": status,
                "evidence": evidence,
                "planned_topology_node_id": (
                    delegation.planner_output.region_search.active_topology_node_id
                    if delegation.planner_output.region_search is not None
                    else None
                ),
            },
        )
        if not isinstance(update, dict):
            return
        packet["topology_goal_search"] = update.get("record")
        packet["topo_memory"] = update.get("topo_memory")
        packet["route_graph"] = update.get("route_graph")
        packet["slow_context"] = update.get("slow_context")

    def _execute_fast(
        self,
        output: NavigationSlowBrainOutput,
        grounded: GroundedNavigationSlowBrainOutput,
        observation: Observation,
        delegation: AgentOSSkillDelegation,
        *,
        allow_stale_slow_reference: bool,
    ) -> dict[str, Any]:
        width, height = QwenNavigationSlowBrain._image_size(observation)
        affordance = output.affordance_pixel.selected()
        target = output.target_pixel.selected()
        target_threshold = float(
            getattr(self.service, "target_confidence_threshold", 0.90)
        )
        target_is_executable = (
            target is not None
            and grounded.target_grounding.confidence >= target_threshold
        )
        if target_is_executable or affordance is None:
            executable = target
            grounding = grounded.target_grounding
            execution_role = "target"
        else:
            executable = affordance
            grounding = grounded.affordance_grounding
            execution_role = "affordance"
        if executable is None:
            raise ValueError("Slow decision contains no executable pixel goal")
        execution_view = executable[0]
        execution_pixel = _pixel_to_native(executable[1], width, height)
        execution_bbox = _bbox_to_native(
            grounding.bbox_normalized_1000,
            width,
            height,
        )
        target_view = target[0] if target is not None else None
        target_pixel = (
            _pixel_to_native(target[1], width, height) if target is not None else None
        )
        target_bbox = (
            _bbox_to_native(
                grounded.target_grounding.bbox_normalized_1000,
                width,
                height,
            )
            if target is not None
            else None
        )
        affordance_view = affordance[0] if affordance is not None else None
        affordance_pixel = (
            _pixel_to_native(affordance[1], width, height)
            if affordance is not None
            else None
        )
        affordance_bbox = (
            _bbox_to_native(
                grounded.affordance_grounding.bbox_normalized_1000,
                width,
                height,
            )
            if affordance is not None
            else None
        )
        affordance_annotation_complete = (
            affordance is not None
            and grounded.affordance_grounding.label is not None
            and affordance_bbox is not None
        )
        evidence = _cot_summary(output) or "ABot-N1 pixel guidance."
        packet = self.service.navigate_to_anchor(
            {
                "slow_step_id": f"slow-{uuid.uuid4().hex[:12]}",
                "observation_id": observation.trace.observation_id,
                "anchor": grounding.label or delegation.planner_output.object_goal,
                "anchor_role": execution_role,
                "view": execution_view,
                "semantic_map": _semantic_map(output),
                "target_pixel": execution_pixel,
                "target_bbox_px": execution_bbox,
                "target_pixel_confidence": grounding.confidence,
                "target_pixel_evidence": (grounding.evidence_summary or evidence)[:240],
                "n1_target_view": target_view,
                "n1_target_pixel": target_pixel,
                "n1_target_label": (
                    grounded.target_grounding.label if target is not None else None
                ),
                "n1_target_bbox_px": target_bbox,
                "n1_target_confidence": (
                    grounded.target_grounding.confidence if target is not None else None
                ),
                "n1_affordance_view": affordance_view,
                "n1_affordance_pixel": affordance_pixel,
                "n1_affordance_label": (
                    grounded.affordance_grounding.label
                    if affordance_annotation_complete
                    else None
                ),
                "n1_affordance_bbox_px": (
                    affordance_bbox if affordance_annotation_complete else None
                ),
                "n1_affordance_confidence": (
                    grounded.affordance_grounding.confidence
                    if affordance_annotation_complete
                    else None
                ),
                "goal_tolerance_m": 0.3,
                "anchor_max_depth_m": float(
                    getattr(
                        self.service,
                        "anchor_max_depth_m",
                        delegation.planner_output.anchor_max_depth_m,
                    )
                ),
            },
            allow_stale_slow_reference=allow_stale_slow_reference,
        )
        packet["anchor_role"] = execution_role
        return packet

    @staticmethod
    def _no_pixel_goal_packet(observation: Observation) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "blocked",
            "error": "Slow System found no visible affordance or target pixel.",
            "result": {
                "controller": "depth_astar_habitat_steps",
                "reached": False,
                "distance_m": 0.0,
                "stalled": True,
                "termination_reason": "no_visible_pixel_goal",
            },
            "execution_telemetry": EmbodimentExecutionTelemetry(
                state="blocked",
                goal_reached=False,
                motion_attempted=False,
                progress_m=0.0,
                stalled=True,
                termination_reason="no_visible_pixel_goal",
                source="navigation-skill",
            ).model_dump(mode="json"),
            "post_observation": observation.model_dump(mode="json"),
            "next_observation_id": observation.trace.observation_id,
        }

    @staticmethod
    def _summary(output: NavigationSlowBrainOutput, packet: dict[str, Any]) -> str:
        telemetry = AgentOSHierarchicalNavigationSkill._telemetry(packet)
        affordance = output.affordance_pixel.selected()
        target = output.target_pixel.selected()
        cot = _cot_summary(output) or "pixel-only task"
        return (
            f"Slow decision={cot[:240]}; executed_role={packet.get('anchor_role')}; "
            f"affordance={affordance}; target={target}; "
            f"reached={telemetry.goal_reached}; state={telemetry.state}; "
            f"progress={telemetry.progress_m:.2f}m; "
            f"termination={telemetry.termination_reason}; "
            f"next_observation={packet.get('next_observation_id')}"
        )

    @staticmethod
    def _evidence(packet: dict[str, Any]) -> list[Evidence]:
        evidence: list[Evidence] = []
        generated = packet.get("generated_point") or {}
        if generated.get("annotated_ref"):
            evidence.append(
                Evidence(
                    kind="slow_brain_pixel_goal",
                    summary=str(generated.get("evidence_summary") or "ABot-N1 pixel guidance."),
                    ref=str(generated["annotated_ref"]),
                )
            )
        next_id = packet.get("next_observation_id")
        if next_id:
            evidence.append(
                Evidence(
                    kind="post_observation",
                    summary="Fresh synchronized observation after Fast System execution.",
                    ref=f"observation:{next_id}",
                )
            )
        for index, frame in enumerate(
            ((packet.get("result") or {}).get("trajectory_keyframes") or [])[-4:],
            start=1,
        ):
            ref = (frame.get("rgb_refs") or {}).get("front")
            if ref:
                evidence.append(
                    Evidence(
                        kind="trajectory_keyframe",
                        summary=f"Fast System trajectory keyframe {index}.",
                        ref=str(ref),
                        data={
                            "pose": frame.get("pose"),
                            "yaw": frame.get("yaw"),
                            "location_id": frame.get("location_id"),
                        },
                    )
                )
        return evidence[-8:]


def _pixel_to_native(pixel: list[int], width: int, height: int) -> list[int]:
    return [
        round(pixel[0] * (width - 1) / 1000.0),
        round(pixel[1] * (height - 1) / 1000.0),
    ]


def _bbox_to_native(bbox: list[int], width: int, height: int) -> list[int]:
    if len(bbox) != 4:
        raise ValueError("normalized bbox must contain four coordinates")
    xmin, ymin, xmax, ymax = bbox
    return [
        round(xmin * (width - 1) / 1000.0),
        round(ymin * (height - 1) / 1000.0),
        round(xmax * (width - 1) / 1000.0),
        round(ymax * (height - 1) / 1000.0),
    ]


def _topology_point_goal(packet: dict[str, Any]) -> str | None:
    route_plan = packet.get("semantic_route_plan") or {}
    guidance = route_plan.get("regional_guidance") or {}
    if guidance.get("mode") != "route_to_region":
        return None
    node_id = guidance.get("next_topology_node_id")
    return str(node_id) if node_id else None


def _topology_goal_tolerance(packet: dict[str, Any]) -> float:
    route_plan = packet.get("semantic_route_plan") or {}
    guidance = route_plan.get("regional_guidance") or {}
    radius = float(guidance.get("node_arrival_radius_m") or 0.5)
    return max(0.3, min(radius, 2.0))


def _topology_semantic_map(packet: dict[str, Any]) -> dict[str, Any]:
    route_plan = packet.get("semantic_route_plan") or {}
    guidance = route_plan.get("regional_guidance") or {}
    return {
        "scene_summary": str(
            guidance.get("reason") or "PointGoal navigation along the topology route."
        )[:160],
        "left": [],
        "front": [],
        "right": [],
        "route_node_type": "none",
        "traversable_directions": list(guidance.get("allowed_anchor_views") or [])[:3],
    }


def _slow_topology_context(
    handoff: dict[str, Any],
    delegation: AgentOSSkillDelegation,
) -> SlowBrainTopologyContext | None:
    """Select a small connected semantic subgraph for one Slow decision."""

    slow_context = handoff.get("slow_context") or {}
    planner_context = slow_context.get("topology_context") or {}
    topo = planner_context.get("topo_memory") or {}
    if not isinstance(topo, dict) or not topo.get("map_id"):
        return None

    route_plan = handoff.get("semantic_route_plan") or {}
    guidance = route_plan.get("regional_guidance") or {}
    selected = delegation.planner_output.node_selection
    current_id = topo.get("current_topology_node_id")
    target_id = (
        guidance.get("target_topology_node_id")
        or (selected.selected_topology_node_id if selected is not None else None)
        or delegation.delegated_subtask.topology_node_id
    )
    next_id = guidance.get("next_topology_node_id")
    route_ids = [
        str(node_id)
        for node_id in (guidance.get("route_node_ids") or [])
        if node_id
    ][:12]

    roads = [item for item in (topo.get("road_layer") or []) if isinstance(item, dict)]
    block_items = [
        item for item in (topo.get("block_layer") or []) if isinstance(item, dict)
    ]
    relevant_ids = _unique_ids([current_id, target_id, next_id, *route_ids])
    if not relevant_ids:
        relevant_ids = _unique_ids(
            [item.get("topology_node_id") for item in block_items]
        )[:8]
    for road in roads:
        source = road.get("source")
        target = road.get("target")
        if source in relevant_ids or target in relevant_ids:
            relevant_ids = _unique_ids([*relevant_ids, source, target])[:8]

    blocks = {
        str(item.get("topology_node_id")): item
        for item in block_items
        if item.get("topology_node_id")
    }
    functions = {
        str(item.get("topology_node_id")): item
        for item in (topo.get("function_layer") or [])
        if isinstance(item, dict) and item.get("topology_node_id")
    }
    objects = {
        str(item.get("topology_node_id")): item
        for item in (topo.get("object_poi_layer") or [])
        if isinstance(item, dict) and item.get("topology_node_id")
    }
    nodes: list[SlowBrainSemanticTopologyNode] = []
    for node_id in relevant_ids[:8]:
        block = blocks.get(node_id)
        if block is None:
            continue
        function = functions.get(node_id) or {}
        object_poi = objects.get(node_id) or {}
        nodes.append(
            SlowBrainSemanticTopologyNode(
                topology_node_id=node_id,
                place_type=str(block.get("place_type") or "unknown region"),
                scene_summary=str(
                    block.get("scene_summary")
                    or block.get("place_type")
                    or "Unknown mapped region"
                ),
                function=str(function.get("function") or ""),
                task_affordances=list(function.get("task_affordances") or [])[:8],
                objects=list(object_poi.get("objects") or [])[:12],
                landmarks=list(object_poi.get("landmarks") or [])[:8],
                visited=bool(block.get("visited")),
                visit_count=int(block.get("visit_count") or 0),
                goal_search=list(block.get("goal_search") or [])[-4:],
            )
        )
    retained_ids = {node.topology_node_id for node in nodes}
    edges = [
        SlowBrainTopologyEdge.model_validate(road)
        for road in roads
        if road.get("source") in retained_ids and road.get("target") in retained_ids
    ][:16]
    return SlowBrainTopologyContext(
        map_id=str(topo["map_id"]),
        revision=int(topo.get("revision") or 0),
        current_topology_node_id=current_id,
        target_topology_node_id=target_id,
        next_topology_node_id=next_id,
        route_node_ids=route_ids,
        nodes=nodes,
        edges=edges,
        target_region_reached=bool(guidance.get("target_region_reached")),
        guidance_reason=str(guidance.get("reason") or ""),
    )


def _unique_ids(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and str(value) not in result:
            result.append(str(value))
    return result


def _cot_summary(output: NavigationSlowBrainOutput) -> str:
    if isinstance(output.cot, InstructionFollowingCoT):
        return (
            f"Finished: {output.cot.finished_sub_instruction} "
            f"Ongoing: {output.cot.ongoing_sub_instruction}"
        )
    if isinstance(output.cot, ObjectGoalCoT):
        return (
            f"Target: {output.cot.target_localization_and_key_context} "
            f"Path: {output.cot.path_planning}"
        )
    return ""


def _semantic_map(output: NavigationSlowBrainOutput) -> dict[str, Any]:
    labels = {"left": [], "front": [], "right": []}
    affordance = output.affordance_pixel.selected()
    if affordance is not None:
        labels[affordance[0]].append("visible traversable affordance")
    target = output.target_pixel.selected()
    if target is not None:
        labels[target[0]].append("visible final target")
    return {
        "scene_summary": (_cot_summary(output) or "ABot-N1 pixel guidance.")[:160],
        **labels,
        "route_node_type": "none",
        "traversable_directions": [],
    }


def _packet_observation(packet: dict[str, Any]) -> Observation | None:
    raw = packet.get("post_observation") or packet.get("observation")
    return Observation.model_validate(raw) if isinstance(raw, dict) else None


def _reference_observation(observation: Observation) -> SlowBrainReferenceObservation:
    return SlowBrainReferenceObservation(
        observation_id=observation.trace.observation_id,
        left_image_ref=observation.rgb_refs["left"],
        front_image_ref=observation.rgb_refs["front"],
        right_image_ref=observation.rgb_refs["right"],
    )
