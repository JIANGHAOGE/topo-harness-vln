from __future__ import annotations

import json
from itertools import combinations
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator

from agentos.memory.store import GraphMemoryStore
from agentos.providers.base import LLMProvider
from agentos.providers.vision import (
    audit_observation_views,
    audited_observation_content,
    image_ref_content,
)
from agentos.schemas.base import SchemaModel
from agentos.schemas.llm import ChatMessage
from agentos.schemas.memory import MemoryNode, MemoryNodeType
from agentos.schemas.observation import Observation


ShortText = Annotated[str, Field(min_length=1, max_length=180)]
ConciseText = Annotated[str, Field(min_length=1, max_length=360)]
NavigationCommand = Annotated[
    str,
    Field(min_length=18, max_length=180, pattern=r"^Navigate towards [^\n]+$"),
]
AnchorView = Literal["left", "front", "right"]


class OmniTaskPlan(SchemaModel):
    decision: Literal["run_omninav", "stop"]
    task_summary: ConciseText
    task_success_conditions: list[ShortText] = Field(min_length=1, max_length=6)
    subtasks: list[NavigationCommand] = Field(min_length=1, max_length=1)
    initial_visible_anchor: ShortText
    initial_anchor_view: AnchorView
    initial_subtask: NavigationCommand
    initial_subtask_success_conditions: list[ShortText] = Field(min_length=1, max_length=4)
    initial_scene_description: ConciseText
    rationale_summary: ConciseText

    @model_validator(mode="after")
    def validate_visible_navigation(self):
        if self.decision == "run_omninav" and self.subtasks != [self.initial_subtask]:
            raise ValueError("subtasks must contain only the current visible-anchor command")
        if self.initial_visible_anchor and self.initial_subtask:
            if self.initial_visible_anchor.lower() not in self.initial_subtask.lower():
                raise ValueError("initial_subtask must name initial_visible_anchor")
        return self


class OmniTrajectoryDescription(SchemaModel):
    frame_index: int = Field(ge=0)
    observation_id: str = Field(min_length=1)
    description: ConciseText


class OmniPostDecision(SchemaModel):
    previous_subtask_status: Literal["success", "partial", "failed", "uncertain"]
    evidence_summary: ConciseText
    current_scene_description: ConciseText
    omni_action_description: ConciseText
    trajectory_descriptions: list[OmniTrajectoryDescription] = Field(min_length=1, max_length=4)
    decision: Literal["continue_subtask", "next_subtask", "stop", "failed"]
    next_visible_anchor: ShortText
    next_anchor_view: AnchorView
    next_subtask: NavigationCommand
    next_subtask_success_conditions: list[ShortText] = Field(min_length=1, max_length=4)
    rationale_summary: ConciseText

    @model_validator(mode="after")
    def validate_visible_navigation(self):
        if self.next_visible_anchor and self.next_subtask:
            if self.next_visible_anchor.lower() not in self.next_subtask.lower():
                raise ValueError("next_subtask must name next_visible_anchor")
        return self


class OmniFinishVerification(SchemaModel):
    verdict: Literal["finish", "continue"]
    evidence_view: Literal["left", "front", "right", "multiple", "none"]
    evidence_summary: ConciseText
    next_visible_anchor: ShortText
    next_anchor_view: AnchorView
    next_subtask: NavigationCommand
    next_subtask_success_conditions: list[ShortText] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_visible_navigation(self):
        if self.next_visible_anchor and self.next_subtask:
            if self.next_visible_anchor.lower() not in self.next_subtask.lower():
                raise ValueError("next_subtask must name next_visible_anchor")
        return self


class TrajectoryMemoryFrame(SchemaModel):
    frame_index: int = Field(ge=0)
    observation_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    scene_description: str = Field(min_length=1)
    omni_action_description: str = Field(min_length=1)
    subtask: str = Field(min_length=1)
    subtask_status: str = Field(min_length=1)
    pose: dict[str, Any] | None = None


class BoundedTrajectoryMemory:
    """Keep start, latest, and two temporally spread trajectory descriptions."""

    def __init__(self, max_frames: int = 4) -> None:
        if max_frames != 4:
            raise ValueError("OmniNav trajectory memory is fixed at four frames")
        self.max_frames = max_frames
        self.frames: list[TrajectoryMemoryFrame] = []

    def add(self, frame: TrajectoryMemoryFrame) -> None:
        candidates = sorted(
            {item.frame_index: item for item in [*self.frames, frame]}.values(),
            key=lambda item: item.frame_index,
        )
        if len(candidates) <= self.max_frames:
            self.frames = candidates
            return

        first, latest = candidates[0], candidates[-1]
        interior = candidates[1:-1]

        def score(pair: tuple[TrajectoryMemoryFrame, TrajectoryMemoryFrame]) -> tuple[int, int]:
            indices = [first.frame_index, pair[0].frame_index, pair[1].frame_index, latest.frame_index]
            gaps = [right - left for left, right in zip(indices, indices[1:])]
            return min(gaps), -sum(abs(left - right) for left, right in combinations(gaps, 2))

        middle = max(combinations(interior, 2), key=score)
        self.frames = [first, *middle, latest]

    def as_json(self) -> list[dict[str, Any]]:
        return [frame.model_dump(mode="json") for frame in self.frames]


class OmniNavSupervisor:
    """Deliberative Agent around each atomic OmniNav invocation."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def initialize(
        self,
        global_task: str,
        observation: Observation,
        declared_success_conditions: list[str],
    ) -> OmniTaskPlan:
        prompt = (
            f"Global task: {global_task}\n"
            f"Declared success conditions: {declared_success_conditions or ['infer visible success conditions']}\n"
            "Use the current robot-left, robot-front, and robot-right views to ground the next local visual waypoint. "
            "Long-range navigation must be a sequence of currently visible anchors: approach one visible object, "
            "doorway, corridor opening, or room boundary, observe again, then choose the next visible anchor. Never "
            "invent an unseen bedroom, hallway, door, or object as the immediate target. Set initial_visible_anchor "
            "to a uniquely identifiable thing that is actually visible now and initial_anchor_view to its camera. "
            "Choose the visible anchor most likely to advance the global task: prefer a traversable doorway, corridor "
            "opening, or unexplored room boundary over nearby furniture. Do not target a person, robot, chair, table, "
            "or other obstacle unless that object is itself the requested goal. The local success conditions must "
            "mean the robot materially approached the anchor and exposed a new useful view; merely seeing an anchor "
            "that is already visible is not success. "
            "Set initial_subtask exactly in the form 'Navigate towards <visible anchor>' and set subtasks to a list "
            "containing exactly that one current command. Decomposition happens again after this anchor; never list "
            "future anchors or speculate that the unseen global target is behind the current anchor. Explain only why "
            "the anchor is useful for continued exploration. Set decision=run_omninav whenever the global task is "
            "not yet visibly complete. Set decision=stop only when it is already complete. Do not emit velocities, "
            "turn angles, waypoints, frame controls, analysis, self-correction, or quoted instructions."
        )
        plan = await self._complete(
            OmniTaskPlan,
            "You supervise an atomic visual navigation skill. Return only the strict JSON object. Never expose reasoning.",
            prompt,
            observation,
        )
        if plan.decision == "run_omninav":
            plan = plan.model_copy(
                update={
                    "initial_subtask_success_conditions": [
                        _approach_success_condition(str(plan.initial_visible_anchor))
                    ]
                }
            )
        return plan

    async def after_omninav(
        self,
        global_task: str,
        plan: OmniTaskPlan,
        previous_subtask: str,
        previous_success_conditions: list[str],
        omni_run: dict[str, Any],
        trajectory_memory: BoundedTrajectoryMemory,
        trajectory_keyframes: list[dict[str, Any]],
        observation: Observation,
    ) -> OmniPostDecision:
        keyframe_metadata = [
            {key: value for key, value in frame.items() if key != "front_ref"}
            for frame in trajectory_keyframes
        ]
        prompt = (
            f"Global task: {global_task}\n"
            f"Initial decomposition: {plan.model_dump(mode='json')}\n"
            f"Previous subtask: {previous_subtask}\n"
            f"Previous subtask success conditions: {previous_success_conditions}\n"
            f"Atomic OmniNav subtask run summary: {omni_run}\n"
            f"Prior trajectory memory, maximum four descriptions: {trajectory_memory.as_json()}\n"
            f"Selected keyframes spanning this complete OmniNav run: {keyframe_metadata}\n"
            "Use the current three views and the evidence above to assess the previous subtask. Describe what the "
            "current images show, what OmniNav did, and every supplied trajectory keyframe. The keyframes are ordered "
            "from the start to the end of the complete local run; return exactly one trajectory_descriptions item per "
            "keyframe with matching frame_index and observation_id. Then choose the best next semantic subtask for the "
            "global task. If navigation must continue, select one anchor that is actually visible in the current "
            "left/front/right images, record its camera in next_anchor_view, and write next_subtask exactly as "
            "'Navigate towards <visible anchor>'. Link progress through successive visible anchors rather than naming "
            "an unseen destination. Prefer traversable openings that reveal unexplored space, and avoid people, "
            "robots, furniture, and obstacles unless they are the requested target. Define success as approaching "
            "the anchor enough to reveal a new useful view, not as the anchor merely remaining visible. OmniNav local "
            "STOP is evidence only and never forces global completion. Return "
            "decision=stop only when the global task itself is complete. Do not include analysis or self-correction, "
            "and do not emit velocities or per-frame controls. Even for stop or failed, populate next_visible_anchor, "
            "next_anchor_view, and next_subtask with the strongest currently visible evidence anchor; those fields "
            "are recorded but not executed after termination."
        )
        view_audit = await audit_observation_views(self.provider, observation)
        content = audited_observation_content(prompt, observation, view_audit)
        for order, frame in enumerate(trajectory_keyframes, start=1):
            content.extend(
                image_ref_content(
                    f"OMNINAV_TRAJECTORY_KEYFRAME_{order} "
                    f"frame_index={frame['frame_index']} observation_id={frame['observation_id']}",
                    str(frame["front_ref"]),
                )
            )
        post = await self._complete_content(
            OmniPostDecision,
            "You are the post-execution Agent supervising one completed atomic OmniNav subtask. Return strict JSON.",
            content,
        )
        expected = [(int(frame["frame_index"]), str(frame["observation_id"])) for frame in trajectory_keyframes]
        actual = [(item.frame_index, item.observation_id) for item in post.trajectory_descriptions]
        if actual != expected:
            raise ValueError("OmniPostDecision trajectory descriptions do not match the supplied keyframes")
        if post.decision in {"continue_subtask", "next_subtask"}:
            post = post.model_copy(
                update={
                    "next_subtask_success_conditions": [
                        _approach_success_condition(str(post.next_visible_anchor))
                    ]
                }
            )
        return post

    async def verify_finish(
        self,
        global_task: str,
        task_success_conditions: list[str],
        candidate: OmniPostDecision,
        observation: Observation,
    ) -> OmniFinishVerification:
        prompt = (
            f"Original global task: {global_task}\n"
            f"Required success conditions: {task_success_conditions}\n"
            f"Candidate completion claim: {candidate.model_dump(mode='json')}\n"
            "Independently verify every success condition using only the current robot-left, robot-front, and "
            "robot-right images. Set verdict=finish only when the target and required place are visually "
            "unambiguous now. For a bed-search task, a sofa, couch, bench, recliner, or distant ambiguous furniture "
            "is not a bed. If any condition is uncertain, set verdict=continue and provide the single best next "
            "semantic navigation subtask. On continue, choose one object, doorway, corridor opening, or room boundary "
            "that is clearly visible now, record next_visible_anchor and next_anchor_view, and write next_subtask "
            "exactly as 'Navigate towards <visible anchor>'. Prefer a traversable opening that reveals unexplored "
            "space and avoid people, robots, furniture, or obstacles unless they are the requested target. Never "
            "target an unseen destination. Even for finish, populate the next-anchor fields with the visible target "
            "that proves completion; they are evidence only and will not be executed. Keep every string to "
            "one concise sentence."
        )
        finish = await self._complete(
            OmniFinishVerification,
            "You are the independent and skeptical global finish verifier. Return only strict JSON.",
            prompt,
            observation,
        )
        if finish.verdict == "continue":
            finish = finish.model_copy(
                update={
                    "next_subtask_success_conditions": [
                        _approach_success_condition(str(finish.next_visible_anchor))
                    ]
                }
            )
        return finish

    async def _complete(self, schema, system: str, prompt: str, observation: Observation):
        view_audit = await audit_observation_views(self.provider, observation)
        return await self._complete_content(
            schema,
            system,
            audited_observation_content(prompt, observation, view_audit),
        )

    async def _complete_content(self, schema, system: str, content: list[dict[str, Any]]):
        response_schema = schema.model_json_schema()
        response = await self.provider.complete(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=content),
            ],
            response_schema=response_schema,
        )
        try:
            if not isinstance(response.parsed_json, dict):
                raise ValueError(f"{schema.__name__} did not return a JSON object")
            return schema.model_validate(response.parsed_json)
        except (ValueError, ValidationError) as first_error:
            invalid = response.parsed_json if isinstance(response.parsed_json, dict) else response.content[:2000]
            repair = await self.provider.complete(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "Repair one structured AgentOS response. Return only one complete strict JSON object, "
                            "include every required field exactly once, preserve valid visual grounding, and do not "
                            "add analysis or self-correction."
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=(
                            f"Target schema: {schema.__name__}\n"
                            f"Validation error: {first_error}\n"
                            f"Incomplete object: {json.dumps(invalid, ensure_ascii=True)}"
                        ),
                    ),
                ],
                response_schema=response_schema,
            )
            if not isinstance(repair.parsed_json, dict):
                raise ValueError(f"{schema.__name__} repair did not return a JSON object") from first_error
            original_fields = {
                key: value
                for key, value in invalid.items()
                if isinstance(invalid, dict) and key in schema.model_fields
            }
            repaired_fields = {
                **original_fields,
                **{
                    key: value
                    for key, value in repair.parsed_json.items()
                    if key in schema.model_fields
                },
            }
            return schema.model_validate(_normalize_repaired_fields(schema, repaired_fields))


def _approach_success_condition(anchor: str) -> str:
    return f"The robot has approached {anchor} enough to reveal a new useful view beyond or around it."


def _normalize_repaired_fields(schema, fields: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(fields)
    if schema is OmniTaskPlan and normalized.get("decision") == "run_omninav":
        anchor = normalized.get("initial_visible_anchor")
        view = normalized.get("initial_anchor_view")
        command = normalized.get("initial_subtask")
        if anchor and view and command:
            normalized["subtasks"] = [command]
            normalized["initial_subtask_success_conditions"] = [_approach_success_condition(str(anchor))]
            normalized.setdefault(
                "initial_scene_description",
                f"The selected anchor {anchor} is visible in the robot-{view} view.",
            )
            normalized.setdefault(
                "rationale_summary",
                f"Approaching {anchor} supports continued exploration without assuming an unseen destination.",
            )
    elif schema is OmniPostDecision and normalized.get("decision") in {"continue_subtask", "next_subtask"}:
        anchor = normalized.get("next_visible_anchor")
        if anchor:
            normalized["next_subtask_success_conditions"] = [_approach_success_condition(str(anchor))]
    elif schema is OmniFinishVerification and normalized.get("verdict") == "continue":
        anchor = normalized.get("next_visible_anchor")
        if anchor:
            normalized["next_subtask_success_conditions"] = [_approach_success_condition(str(anchor))]
    return normalized


def persist_trajectory_memory(
    memory: GraphMemoryStore,
    *,
    episode_id: str,
    task_id: str,
    trajectory_memory: BoundedTrajectoryMemory,
) -> None:
    """Persist four stable slots; changing representatives overwrite prior slots."""
    for slot, frame in enumerate(trajectory_memory.frames):
        memory.upsert_node(
            MemoryNode(
                node_id=f"semantic_event:{task_id}:omninav-trajectory:{slot}",
                node_type=MemoryNodeType.SEMANTIC_EVENT,
                title=f"OmniNav trajectory memory {slot + 1}/4",
                content=(
                    f"scene={frame.scene_description}; omni_action={frame.omni_action_description}; "
                    f"subtask={frame.subtask}; status={frame.subtask_status}"
                ),
                source_ref=f"observation:{frame.observation_id}",
                episode_id=episode_id,
                task_id=task_id,
                metadata={"trajectory_slot": slot, **frame.model_dump(mode="json")},
            )
        )
