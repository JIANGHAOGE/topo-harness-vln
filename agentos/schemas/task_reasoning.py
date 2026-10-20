from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, WithJsonSchema, model_validator

from agentos.schemas.base import SchemaModel
from agentos.schemas.hierarchical import (
    CameraView,
    SemanticMapUpdate,
    ViewpointAction,
)
from agentos.schemas.semantic_map import NodeSemantics


NavigationActionType = Literal["visible_anchor", "known_location", "viewpoint_adjustment"]
NavigationIntent = Literal["goal_pursuit", "explore_route", "backtrack"]
AnchorRole = Literal["affordance", "target"]
ABotNavigationTaskType = Literal[
    "point_goal",
    "object_goal",
    "instruction_following",
]
ABotSubtaskStage = Literal["approaching", "reaching", "interaction"]
ABotSubtaskStatus = Literal["pending", "active", "completed", "failed"]
AgentOSHighLevelSkillName = Literal[
    "MappingSkill",
    "NavigationSkill",
]
LowConfidenceRecovery = Literal["adjust_viewpoint", "previous_location"]
SemanticRoutePhase = Literal["approach_node", "local_anchor_search", "verify_goal", "explore_nodes"]
RegionalGuidanceMode = Literal["route_to_region", "search_within_region", "unavailable"]
NormalizedBBox1000 = Annotated[
    list[Annotated[int, Field(ge=0, le=1000)]],
    Field(max_length=4),
    WithJsonSchema(
        {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "maxItems": 0,
                },
                {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    "minItems": 4,
                    "maxItems": 4,
                },
            ]
        }
    ),
]


class SlowBrainTaskSpecification(SchemaModel):
    """The stable free-form language goal g used by one ABot-N1 Skill task."""

    task_type: ABotNavigationTaskType = Field(
        description=(
            "point_goal for a known map node, coordinate, or saved location; object_goal for "
            "continuous visual-semantic search using affordance waypoints and a terminal target; "
            "instruction_following for a natural-language route executed by the OmniNav capability"
        )
    )
    goal: str = Field(
        min_length=1,
        max_length=240,
        description="Complete navigation instruction retained unchanged across N1 Slow invocations.",
    )


class SemanticTaskDecompositionInput(SchemaModel):
    """Coarse-to-fine split of one natural-language navigation task."""

    global_goal: str = Field(min_length=1, max_length=240)
    destination_query: str = Field(
        min_length=1,
        max_length=120,
        description="Place or functional scene that constrains the object goal.",
    )
    object_goal: str = Field(
        min_length=1,
        max_length=120,
        description="Object to find specifically within or associated with destination_query.",
    )
    node_local_radius_m: float = Field(default=5.0, gt=0.0, le=20.0)
    anchor_max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)


class AgenticSubtask(SchemaModel):
    """One ABot-N0 typed subtask in a coarse-to-fine navigation plan."""

    subtask_id: str = Field(pattern=r"^subtask-[0-9]{3,}$")
    stage: ABotSubtaskStage
    task_type: ABotNavigationTaskType
    instruction: str = Field(min_length=1, max_length=240)
    target: str = Field(min_length=1, max_length=120)
    topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    success_condition: str = Field(min_length=1, max_length=240)
    status: ABotSubtaskStatus = "pending"


class AgenticPlannerSubtask(SchemaModel):
    """One Skill subtask proposed by the AgentOS Main Agent."""

    skill_name: AgentOSHighLevelSkillName = "NavigationSkill"
    stage: ABotSubtaskStage
    task_type: ABotNavigationTaskType
    instruction: str = Field(min_length=1, max_length=240)
    target: str = Field(min_length=1, max_length=120)
    topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    success_condition: str = Field(min_length=1, max_length=240)


class SemanticNodeSelection(SchemaModel):
    """One model-grounded semantic choice over the fixed topology nodes."""

    selected_topology_node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    confidence: float = Field(ge=0.0, le=1.0)
    matched_concepts: list[str] = Field(default_factory=list, max_length=12)
    selection_reason: str = Field(min_length=1, max_length=240)


class RegionHypothesis(SchemaModel):
    """One model-ranked place hypothesis for a task-scoped object search."""

    topology_node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    confidence: float = Field(ge=0.0, le=1.0)
    place_evidence: str = Field(min_length=1, max_length=240)
    object_evidence: str = Field(min_length=1, max_length=240)
    relation_reason: str = Field(min_length=1, max_length=240)


class RegionSearchPlan(SchemaModel):
    """Optional Main-Agent state for a node-guided object-search task."""

    candidates: list[RegionHypothesis] = Field(min_length=1, max_length=4)
    active_topology_node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    update_reason: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_candidates(self) -> "RegionSearchPlan":
        candidate_ids = [item.topology_node_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("region-search candidate topology_node_id values must be unique")
        if self.active_topology_node_id not in candidate_ids:
            raise ValueError("active_topology_node_id must reference a region-search candidate")
        return self


class AgenticPlannerOutput(SchemaModel):
    """ABot-N0 Planner result after observing topology, episodic memory, and RGB."""

    global_instruction: str = Field(min_length=1, max_length=240)
    destination_query: str = Field(min_length=1, max_length=120)
    object_goal: str = Field(min_length=1, max_length=120)
    node_selection: SemanticNodeSelection | None = None
    region_search: RegionSearchPlan | None = Field(
        default=None,
        description=(
            "Task-scoped ranked region hypotheses, present only when the Main Agent chooses "
            "node-guided search for an unresolved object goal."
        ),
    )
    subtasks: list[AgenticPlannerSubtask] = Field(min_length=1, max_length=8)
    node_local_radius_m: float = Field(default=5.0, gt=0.0, le=20.0)
    anchor_max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)


class AgentOSSkillDelegation(SchemaModel):
    """The complete model-facing output of the AgentOS Main Agent."""

    skill_name: AgentOSHighLevelSkillName = "NavigationSkill"
    task_specification: SlowBrainTaskSpecification
    planner_output: AgenticPlannerOutput
    delegated_subtask_index: int = Field(default=0, ge=0)
    rationale_summary: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_delegated_subtask(self) -> "AgentOSSkillDelegation":
        if self.delegated_subtask_index >= len(self.planner_output.subtasks):
            raise ValueError("delegated_subtask_index must reference planner_output.subtasks")
        if self.skill_name != self.delegated_subtask.skill_name:
            raise ValueError("skill_name must match the delegated subtask skill_name")
        return self

    @property
    def delegated_subtask(self) -> AgenticPlannerSubtask:
        return self.planner_output.subtasks[self.delegated_subtask_index]


class AgenticPlan(SchemaModel):
    """ABot-N0 Planner output consumed by the Actor one subtask at a time."""

    plan_id: str = Field(min_length=1)
    global_instruction: str = Field(min_length=1, max_length=240)
    subtasks: list[AgenticSubtask] = Field(min_length=1, max_length=8)
    active_subtask_index: int = Field(default=0, ge=0)
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_active_subtask(self) -> "AgenticPlan":
        if self.active_subtask_index >= len(self.subtasks):
            raise ValueError("active_subtask_index must reference an existing subtask")
        active = [
            index
            for index, subtask in enumerate(self.subtasks)
            if subtask.status == "active"
        ]
        terminal = all(
            subtask.status in {"completed", "failed"} for subtask in self.subtasks
        )
        if not terminal and active != [self.active_subtask_index]:
            raise ValueError(
                "a non-terminal Agentic Plan must have exactly one indexed active subtask"
            )
        return self


class SubtaskReflection(SchemaModel):
    """ABot-N0 Self-Reflector output at a macro-subtask boundary."""

    subtask_id: str = Field(pattern=r"^subtask-[0-9]{3,}$")
    completed: bool
    outcome: Literal["completed", "partial", "failed"]
    feedback: str = Field(min_length=1, max_length=240)
    should_replan: bool = False

    @model_validator(mode="after")
    def validate_outcome_contract(self) -> "SubtaskReflection":
        if self.completed != (self.outcome == "completed"):
            raise ValueError("completed must agree with outcome=completed")
        if self.outcome == "failed" and not self.should_replan:
            raise ValueError("outcome=failed requires should_replan=true")
        return self


class FinishVerifierAssessment(SchemaModel):
    """Independent semantic evidence judgment before Runtime authorization."""

    evidence_satisfies_task: bool
    terminal_failure_supported: bool
    observed_evidence: list[str] = Field(min_length=1, max_length=8)


class FinishVerifierDecision(SchemaModel):
    """Independent AgentOS authorization for global task termination."""

    requested_outcome: Literal["done", "failed"]
    authorized: bool
    recommended_action: Literal["finish", "replan", "fail"]
    summary: str = Field(min_length=1, max_length=240)
    observed_evidence: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_authorization_evidence(self) -> "FinishVerifierDecision":
        if self.requested_outcome == "done" and self.authorized:
            if not self.observed_evidence:
                raise ValueError("authorized done requires observed_evidence")
            if self.recommended_action != "finish":
                raise ValueError("authorized done requires recommended_action=finish")
        return self


class RegionalAnchorGuidance(SchemaModel):
    """Topology-derived directional prior for the next visual anchor."""

    mode: RegionalGuidanceMode
    current_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    target_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    next_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    route_node_ids: list[str] = Field(default_factory=list, max_length=24)
    route_progress_index: int = Field(default=0, ge=0)
    guidance_point_xyz_m: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )
    distance_to_next_node_m: float | None = Field(default=None, ge=0.0)
    distance_to_target_node_m: float | None = Field(default=None, ge=0.0)
    node_arrival_radius_m: float | None = Field(default=None, gt=0.0)
    target_region_reached: bool = False
    relative_bearing_deg: float | None = Field(default=None, ge=-180.0, le=180.0)
    allowed_anchor_views: list[CameraView] = Field(default_factory=list, max_length=3)
    recommended_viewpoint_action: ViewpointAction | None = None
    direction_constraint_active: bool = False
    reason: str = Field(min_length=1, max_length=240)


class SemanticRoutePlan(SchemaModel):
    """Regional node context for a chain of visually grounded navigation subtasks."""

    plan_id: str = Field(min_length=1)
    global_goal: str = Field(min_length=1, max_length=240)
    destination_query: str = Field(min_length=1, max_length=120)
    object_goal: str = Field(min_length=1, max_length=120)
    execution_strategy: Literal["anchor_chain"] = "anchor_chain"
    phase: SemanticRoutePhase
    selected_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    destination_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    destination_selection: SemanticNodeSelection | None = None
    evidence_location_id: str | None = Field(
        default=None, pattern=r"^location-[0-9]{3,}$"
    )
    selected_node_semantics: NodeSemantics | None = None
    match_score: float | None = Field(default=None, ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list, max_length=12)
    selection_reason: str = Field(min_length=1, max_length=240)
    regional_guidance: RegionalAnchorGuidance | None = None
    node_local_radius_m: float = Field(default=5.0, gt=0.0, le=20.0)
    anchor_max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)
    agentic_plan: AgenticPlan


class PreviousSubtaskAssessment(SchemaModel):
    execution_status: Literal["not_started", "completed", "partial", "failed"]
    task_effect: Literal["progress", "no_progress", "hypothesis_rejected", "blocked", "unknown"]
    evidence: str = Field(min_length=1, max_length=240)


class RouteHypothesis(SchemaModel):
    statement: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0.0, le=1.0)


class InstructionFollowingCoT(SchemaModel):
    """ABot-N1 instruction-following CoT with done/current progress only."""

    finished_sub_instruction: str = Field(
        min_length=1, max_length=512
    )
    ongoing_sub_instruction: str = Field(
        min_length=1, max_length=512
    )


class ObjectGoalCoT(SchemaModel):
    """ABot-N1 object-goal CoT with localization and path planning only."""

    target_localization_and_key_context: str = Field(min_length=1, max_length=512)
    path_planning: str = Field(
        min_length=1,
        max_length=512,
    )


ABotNavigationCoT = InstructionFollowingCoT | ObjectGoalCoT


class SlowBrainSemanticTopologyNode(SchemaModel):
    """One bounded ABot-N0 semantic node supplied as an N1 reasoning prior."""

    topology_node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    place_type: str = Field(min_length=1, max_length=64)
    scene_summary: str = Field(min_length=1, max_length=160)
    function: str = Field(default="", max_length=200)
    task_affordances: list[str] = Field(default_factory=list, max_length=8)
    objects: list[str] = Field(default_factory=list, max_length=12)
    landmarks: list[str] = Field(default_factory=list, max_length=8)
    visited: bool = False
    visit_count: int = Field(default=0, ge=0)
    goal_search: list[dict[str, object]] = Field(default_factory=list, max_length=4)


class SlowBrainTopologyEdge(SchemaModel):
    """One traversable relation retained in the bounded Slow-System context."""

    source: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    target: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    distance_m: float | None = Field(default=None, ge=0.0)
    transition: str = Field(default="connected route", max_length=120)


class SlowBrainTopologyContext(SchemaModel):
    """Task-relevant semantic topology used to constrain N1 CoT and pixel guidance."""

    map_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    current_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    target_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    next_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    route_node_ids: list[str] = Field(default_factory=list, max_length=12)
    nodes: list[SlowBrainSemanticTopologyNode] = Field(
        default_factory=list, max_length=8
    )
    edges: list[SlowBrainTopologyEdge] = Field(default_factory=list, max_length=16)
    target_region_reached: bool = False
    guidance_reason: str = Field(default="", max_length=240)


class SlowBrainTriViewPixelGoal(SchemaModel):
    """ABot-N1 per-view pixel slots, serialized on a normalized 0-1000 grid."""

    left: list[Annotated[int, Field(ge=0, le=1000)]] = Field(
        default_factory=list, max_length=2
    )
    front: list[Annotated[int, Field(ge=0, le=1000)]] = Field(
        default_factory=list, max_length=2
    )
    right: list[Annotated[int, Field(ge=0, le=1000)]] = Field(
        default_factory=list, max_length=2
    )

    @model_validator(mode="after")
    def validate_pixel_slots(self) -> "SlowBrainTriViewPixelGoal":
        populated = 0
        for view in ("left", "front", "right"):
            pixel = getattr(self, view)
            if pixel and len(pixel) != 2:
                raise ValueError(f"{view} pixel must be [] or [x, y]")
            populated += bool(pixel)
        if populated > 1:
            raise ValueError("a pixel goal may populate only one synchronized camera view")
        return self

    def selected(self) -> tuple[CameraView, list[int]] | None:
        for view in ("left", "front", "right"):
            pixel = getattr(self, view)
            if pixel:
                return view, list(pixel)
        return None


class NavigationSlowBrainOutput(SchemaModel):
    """ABot-N1 Slow output consumed by the deterministic Fast substitute."""

    cot: ABotNavigationCoT | None = None
    affordance_pixel: SlowBrainTriViewPixelGoal = Field(
        default_factory=SlowBrainTriViewPixelGoal
    )
    target_pixel: SlowBrainTriViewPixelGoal = Field(
        default_factory=SlowBrainTriViewPixelGoal
    )


class InstructionFollowingSlowBrainOutput(NavigationSlowBrainOutput):
    cot: InstructionFollowingCoT


class ObjectGoalSlowBrainOutput(NavigationSlowBrainOutput):
    cot: ObjectGoalCoT


class PixelOnlySlowBrainOutput(NavigationSlowBrainOutput):
    cot: None = None


class SlowBrainVisualGrounding(SchemaModel):
    """One current-frame bbox returned by the independent Grounding Skill."""

    label: str | None = Field(min_length=1, max_length=120)
    view: CameraView | None
    bbox_normalized_1000: NormalizedBBox1000
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = Field(max_length=240)

    @model_validator(mode="after")
    def validate_grounding(self) -> "SlowBrainVisualGrounding":
        if not self.bbox_normalized_1000:
            if self.label is not None or self.view is not None or self.confidence != 0.0:
                raise ValueError(
                    "empty grounding requires null label/view, confidence 0, and bbox []"
                )
            return self
        if len(self.bbox_normalized_1000) != 4:
            raise ValueError("grounding bbox must be [] or [xmin, ymin, xmax, ymax]")
        if self.label is None or self.view is None:
            raise ValueError("populated grounding requires label and view")
        if re.fullmatch(
            r"(?:backbone|location)-[0-9]{3,}", self.label.strip(), re.IGNORECASE
        ):
            raise ValueError("grounding label must be visual semantics, not a navigation ID")
        xmin, ymin, xmax, ymax = self.bbox_normalized_1000
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("grounding bbox must have positive width and height")
        if self.confidence <= 0.0:
            raise ValueError("populated grounding requires positive confidence")
        return self

    def bottom_center(self) -> list[int] | None:
        if not self.bbox_normalized_1000:
            return None
        xmin, _, xmax, ymax = self.bbox_normalized_1000
        return [round((xmin + xmax) / 2), ymax]

class GroundedNavigationSlowBrainOutput(SchemaModel):
    """Grounding result converted to the public N1 CoT-and-pixels contract."""

    cot: ABotNavigationCoT | None = None
    affordance_grounding: SlowBrainVisualGrounding
    target_grounding: SlowBrainVisualGrounding


class SlowBrainSemanticGroundingQuery(SchemaModel):
    """Coordinate-free semantic request emitted by the Slow System."""

    role: AnchorRole
    label: str = Field(min_length=1, max_length=120)
    preferred_view: CameraView | None = None
    visual_description: str = Field(default="", max_length=512)
    spatial_relation: str = Field(min_length=1, max_length=512)
    excluded_labels: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_visual_semantics(self) -> "SlowBrainSemanticGroundingQuery":
        if self.label.strip().casefold() in {"left", "front", "right"}:
            raise ValueError(
                "grounding query label must name a visual object or transition, not a camera view"
            )
        if re.fullmatch(
            r"(?:backbone|location)-[0-9]{3,}", self.label, re.IGNORECASE
        ):
            raise ValueError("grounding query label must be visual semantics, not a navigation ID")
        return self


class SlowBrainAffordanceGroundingQuery(SlowBrainSemanticGroundingQuery):
    """A Slow-selected traversable anchor with one supporting current camera."""

    role: Literal["affordance"] = "affordance"
    preferred_view: CameraView

    @model_validator(mode="after")
    def validate_navigable_semantic_label(self) -> "SlowBrainAffordanceGroundingQuery":
        normalized = re.sub(r"[_-]+", " ", self.label.strip().casefold())
        camera_words = {
            "left",
            "front",
            "right",
            "camera",
            "view",
            "image",
            "current",
            "robot",
        }
        if normalized and set(normalized.split()) <= camera_words:
            raise ValueError(
                "affordance label must name a visible navigable semantic waypoint, not a camera or view"
            )
        return self


class SlowBrainTargetGroundingQuery(SlowBrainSemanticGroundingQuery):
    """A final object query whose view is optional until Slow sees the target."""

    role: Literal["target"] = "target"


class SlowBrainViewBinding(SchemaModel):
    """Current-image support for one semantic query selected by Slow Brain."""

    role: AnchorRole
    label: str = Field(min_length=1, max_length=120)
    visually_supported: bool
    verified_view: CameraView | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_description: str = Field(default="", max_length=512)
    evidence_summary: str = Field(default="", max_length=240)

    @model_validator(mode="after")
    def validate_support(self) -> "SlowBrainViewBinding":
        if self.visually_supported and self.verified_view is None:
            raise ValueError("visually supported query requires verified_view")
        if not self.visually_supported and self.verified_view is not None:
            raise ValueError("unsupported query must keep verified_view null")
        if not self.visually_supported and self.confidence != 0.0:
            raise ValueError("unsupported query must use confidence 0")
        return self


class SlowBrainViewBindingVerification(SchemaModel):
    """Internal Slow-side camera binding check before pixel grounding."""

    bindings: list[SlowBrainViewBinding] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_unique_roles(self) -> "SlowBrainViewBindingVerification":
        roles = [binding.role for binding in self.bindings]
        if len(roles) != len(set(roles)):
            raise ValueError("view binding verification requires unique roles")
        return self


class NavigationSlowBrainSemanticOutput(SchemaModel):
    """Slow reasoning output before current-frame visual grounding."""

    cot: ABotNavigationCoT | None = None
    affordance_query: SlowBrainAffordanceGroundingQuery
    target_query: SlowBrainTargetGroundingQuery | None = None

    @model_validator(mode="after")
    def validate_query_roles(self) -> "NavigationSlowBrainSemanticOutput":
        if self.affordance_query.role != "affordance":
            raise ValueError("affordance_query.role must be affordance")
        if self.affordance_query.preferred_view is None:
            raise ValueError(
                "affordance_query.preferred_view must identify its supporting current view"
            )
        if self.target_query is not None and self.target_query.role != "target":
            raise ValueError("target_query.role must be target")
        return self


class InstructionFollowingSlowBrainSemanticOutput(NavigationSlowBrainSemanticOutput):
    cot: InstructionFollowingCoT


class ObjectGoalSlowBrainSemanticOutput(NavigationSlowBrainSemanticOutput):
    cot: ObjectGoalCoT


class PixelOnlySlowBrainSemanticOutput(NavigationSlowBrainSemanticOutput):
    cot: None = None


class SlowBrainReferenceMemoryFrame(SchemaModel):
    """One historical front-view frame in ABot-N1 reference memory."""

    slot: int = Field(ge=0, le=3)
    front_image_ref: str = Field(min_length=1)
    observation_id: str | None = Field(default=None, min_length=1)


class SlowBrainReferenceObservation(SchemaModel):
    """The synchronized tri-view captured for one Slow-System invocation."""

    observation_id: str = Field(min_length=1)
    left_image_ref: str = Field(min_length=1)
    front_image_ref: str = Field(min_length=1)
    right_image_ref: str = Field(min_length=1)


class SlowBrainSingleViewAudit(SchemaModel):
    """Task-neutral visual facts extracted from exactly one current camera image."""

    view: CameraView
    visible_objects: list[str] = Field(default_factory=list, max_length=16)
    navigable_semantic_anchors: list[str] = Field(default_factory=list, max_length=8)
    target_relevant_evidence: list[str] = Field(default_factory=list, max_length=8)
    scene_summary: str = Field(min_length=1, max_length=180)


class SlowBrainTriViewAudit(SchemaModel):
    """Synchronized single-frame audits keyed by physical robot camera identity."""

    left: SlowBrainSingleViewAudit
    front: SlowBrainSingleViewAudit
    right: SlowBrainSingleViewAudit

    @model_validator(mode="after")
    def validate_camera_identity(self) -> "SlowBrainTriViewAudit":
        for view in ("left", "front", "right"):
            if getattr(self, view).view != view:
                raise ValueError(f"{view} audit must retain view={view}")
        return self


class NavigationSlowBrainInput(SchemaModel):
    """The four N1 inputs plus an explicit, bounded ABot-N0 topology extension."""

    reference_memory: list[SlowBrainReferenceMemoryFrame] = Field(
        default_factory=list, max_length=4
    )
    reference_observation: SlowBrainReferenceObservation
    current_view_audit: SlowBrainTriViewAudit | None = None
    task_specification: SlowBrainTaskSpecification
    previous_decision: NavigationSlowBrainOutput | None = None
    topology_context: SlowBrainTopologyContext | None = None


class NavigationFastSystemInput(SchemaModel):
    """Bounded inputs for the deterministic target-pixel Fast substitute."""

    current_memory: list[SlowBrainReferenceMemoryFrame] = Field(
        default_factory=list, max_length=4
    )
    current_observation: SlowBrainReferenceObservation
    slow_decision: NavigationSlowBrainOutput
    slow_reference_observation: SlowBrainReferenceObservation
    task_specification: SlowBrainTaskSpecification


class NavigationCandidate(SchemaModel):
    action_type: NavigationActionType
    intent: NavigationIntent = "goal_pursuit"
    target: str = Field(min_length=1, max_length=120)
    task_relevance: float = Field(ge=0.0, le=1.0)
    information_gain: float = Field(ge=0.0, le=1.0)
    revisit_penalty: float = Field(ge=0.0, le=1.0)


class SelectedNavigationAction(SchemaModel):
    action_type: NavigationActionType
    intent: NavigationIntent = "goal_pursuit"
    target: str = Field(min_length=1, max_length=120)
    view: CameraView | None = None
    anchor_role: AnchorRole = "affordance"
    topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    location_id: str | None = Field(default=None, pattern=r"^location-[0-9]{3,}$")
    viewpoint_action: ViewpointAction | None = None
    low_confidence_recovery: LowConfidenceRecovery = "adjust_viewpoint"
    recovery_viewpoint_action: ViewpointAction | None = None
    expected_evidence: str = Field(min_length=1, max_length=240)
    success_condition: str = Field(min_length=1, max_length=240)
    fallback: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "SelectedNavigationAction":
        if self.action_type == "visible_anchor":
            if self.view is None:
                raise ValueError("visible_anchor requires view")
            if re.search(r"\b(?:backbone|location)-[0-9]{3,}\b", self.target, re.IGNORECASE):
                raise ValueError(
                    "visible_anchor target must be a visually groundable semantic phrase, not a navigation ID"
                )
            if self.topology_node_id is not None or self.location_id is not None:
                raise ValueError("visible_anchor cannot include a known-location target ID")
            if self.viewpoint_action is not None:
                raise ValueError("visible_anchor cannot include viewpoint_action")
        if self.action_type != "visible_anchor" and self.anchor_role != "affordance":
            raise ValueError("target anchor_role is valid only for visible_anchor")
        if self.action_type == "known_location":
            target_count = sum(
                value is not None for value in (self.topology_node_id, self.location_id)
            )
            if target_count != 1:
                raise ValueError(
                    "known_location requires exactly one topology_node_id or location_id"
                )
            if self.view is not None or self.viewpoint_action is not None:
                raise ValueError("known_location cannot include camera or viewpoint fields")
        if self.action_type == "viewpoint_adjustment":
            if self.viewpoint_action is None:
                raise ValueError("viewpoint_adjustment requires viewpoint_action")
            if any(
                value is not None
                for value in (self.view, self.topology_node_id, self.location_id)
            ):
                raise ValueError("viewpoint_adjustment cannot include navigation target fields")
        return self


class AdvanceNavigationInput(SchemaModel):
    observation_id: str = Field(min_length=1)
    global_goal: str = Field(min_length=1, max_length=240)
    target_evidence: list[str] = Field(default_factory=list, max_length=8)
    navigation_task_type: ABotNavigationTaskType = "object_goal"
    cot: ABotNavigationCoT | None = Field(
        default_factory=lambda: ObjectGoalCoT(
            target_localization_and_key_context="No target evidence is available yet.",
            path_planning="Continue visual-semantic navigation toward the active object goal.",
        )
    )
    self_reflection: SubtaskReflection | None = None
    planner_output: AgenticPlannerOutput | None = None
    previous_assessment: PreviousSubtaskAssessment
    route_hypothesis: RouteHypothesis
    candidates: list[NavigationCandidate] = Field(min_length=1, max_length=3)
    action: SelectedNavigationAction
    semantic_map: SemanticMapUpdate
    node_semantics: NodeSemantics | None = None
    rejected_branch: str | None = Field(default=None, max_length=160)
    max_control_steps: int = Field(default=64, ge=1, le=1000)
    goal_tolerance_m: float = Field(default=0.3, gt=0.0, le=2.0)
    anchor_max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)

    @model_validator(mode="after")
    def validate_selected_candidate(self) -> "AdvanceNavigationInput":
        if self.navigation_task_type == "object_goal" and not isinstance(
            self.cot, ObjectGoalCoT
        ):
            raise ValueError("object_goal requires the ABot-N1 localization/path CoT")
        if self.navigation_task_type == "point_goal" and self.cot is not None:
            raise ValueError(
                f"{self.navigation_task_type} does not emit linguistic CoT in ABot-N1"
            )
        if not any(
            candidate.action_type == self.action.action_type
            and candidate.target.casefold() == self.action.target.casefold()
            for candidate in self.candidates
        ):
            raise ValueError("selected action must match one proposed candidate")
        return self
