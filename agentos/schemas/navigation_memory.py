from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now
from agentos.schemas.semantic_map import NodeSemantics


BranchStatus = Literal["unexplored", "exploring", "deprioritized", "confirmed"]
TopologyGoalSearchStatus = Literal[
    "no_target_observed",
    "target_observed",
    "target_confirmed",
    "blocked",
]
PlaceSource = Literal["observation", "trajectory_waypoint", "trajectory_keyframe", "map_backbone"]
TopologyNodeType = Literal[
    "none",
    "global_anchor",
    "open_space_center",
    "junction",
    "semantic_landmark",
]
VisualMemoryKind = Literal["global_anchor", "trajectory_keyframe"]
VisualMemoryScope = Literal["task_overview", "latest_subtask", "task"]


class MetricPoseEvidence(SchemaModel):
    x: float
    y: float
    yaw: float
    frame: str = Field(min_length=1)
    source: str = Field(min_length=1)
    map_id: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=utc_now)


class DirectionalSemanticObservation(SchemaModel):
    yaw: float
    scene_summary: str = Field(min_length=1, max_length=160)
    left: list[str] = Field(default_factory=list, max_length=4)
    front: list[str] = Field(default_factory=list, max_length=4)
    right: list[str] = Field(default_factory=list, max_length=4)
    traversable_directions: list[Literal["left", "front", "right"]] = Field(
        default_factory=list, max_length=3
    )
    observation_id: str = Field(min_length=1)


class PlaceMemory(SchemaModel):
    place_id: str = Field(pattern=r"^location-[0-9]{3,}$")
    environment_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    pose: MetricPoseEvidence
    navigation_point_xyz_m: list[float] = Field(min_length=3, max_length=3)
    scene_summary: str = Field(min_length=1, max_length=160)
    semantic_labels: list[str] = Field(default_factory=list, max_length=12)
    semantic_observations: list[DirectionalSemanticObservation] = Field(default_factory=list, max_length=4)
    topology_node_id: str | None = Field(default=None, pattern=r"^backbone-[0-9]{3,}$")
    node_semantics: NodeSemantics | None = None
    source: PlaceSource = "observation"
    trajectory_frame_index: int | None = Field(default=None, ge=0)
    front_image_ref: str | None = None
    image_refs: dict[str, str] = Field(default_factory=dict)
    is_global_anchor: bool = False
    is_route_node: bool = False
    route_node_type: TopologyNodeType = "none"
    semantic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    task_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    traversable_directions: list[Literal["left", "front", "right"]] = Field(
        default_factory=list, max_length=3
    )
    anchor_instruction: str = Field(default="", max_length=240)
    visit_count: int = Field(default=1, ge=1)
    branch_status: BranchStatus = "unexplored"
    last_observation_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class PlaceMemoryView(SchemaModel):
    place_id: str
    scene_summary: str
    semantic_labels: list[str] = Field(default_factory=list, max_length=6)
    topology_node_id: str | None = Field(default=None, pattern=r"^backbone-[0-9]{3,}$")
    node_semantics: NodeSemantics | None = None
    visit_count: int = Field(ge=1)
    branch_status: BranchStatus
    pose: MetricPoseEvidence
    source: PlaceSource = "observation"
    trajectory_frame_index: int | None = Field(default=None, ge=0)
    is_global_anchor: bool = False
    is_route_node: bool = False
    route_node_type: TopologyNodeType = "none"
    semantic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    task_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    traversable_directions: list[Literal["left", "front", "right"]] = Field(
        default_factory=list, max_length=3
    )
    anchor_instruction: str = Field(default="", max_length=240)
    relation: Literal["current", "relevant", "recent"]
    relevance: float = Field(ge=0.0, le=1.0)


class BackboneExplorationCandidate(SchemaModel):
    location_id: str = Field(pattern=r"^location-[0-9]{3,}$")
    topology_node_id: str | None = Field(default=None, pattern=r"^backbone-[0-9]{3,}$")
    route_node_type: Literal["open_space_center", "junction"]
    distance_m: float = Field(ge=0.0)
    visited: bool = False
    visit_count: int = Field(default=0, ge=0)
    scene_summary: str = Field(min_length=1, max_length=160)
    semantic_labels: list[str] = Field(default_factory=list, max_length=6)
    semantic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    task_relevance: float | None = Field(default=None, ge=0.0, le=1.0)


class NavigationVisualMemory(SchemaModel):
    visual_id: str = Field(min_length=1)
    subtask_id: str = Field(min_length=1)
    kind: VisualMemoryKind
    environment_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    location_id: str = Field(pattern=r"^location-[0-9]{3,}$")
    view: Literal["left", "front", "right"]
    image_ref: str = Field(min_length=1)
    source_target: str = Field(default="", max_length=160)
    observation_id: str = Field(min_length=1)
    frame_index: int | None = Field(default=None, ge=0)
    pose: MetricPoseEvidence
    navigation_point_xyz_m: list[float] = Field(min_length=3, max_length=3)
    created_at: datetime = Field(default_factory=utc_now)


class NavigationVisualRecallInput(SchemaModel):
    scope: VisualMemoryScope = "task_overview"
    location_id: str | None = Field(default=None, pattern=r"^location-[0-9]{3,}$")
    query: str = Field(default="", max_length=160)
    limit: int = Field(default=4, ge=1, le=4)


class TopologyGoalSearchRecord(SchemaModel):
    """Task-scoped evidence about searching one semantic topology region."""

    topology_node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    planned_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    observed_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    goal: str = Field(min_length=1, max_length=160)
    status: TopologyGoalSearchStatus
    attempt_count: int = Field(default=1, ge=1)
    last_observation_id: str = Field(min_length=1)
    evidence: str = Field(default="", max_length=240)
    updated_at: datetime = Field(default_factory=utc_now)


class TopoMemoryBlock(SchemaModel):
    """ABot-N0 Block layer: stable regions or room-like navigation areas."""

    topology_node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    place_type: str = Field(min_length=1, max_length=64)
    scene_summary: str = Field(min_length=1, max_length=160)
    position_xy_m: list[float] = Field(min_length=2, max_length=2)
    visited: bool = False
    visit_count: int = Field(default=0, ge=0)
    goal_search: list[TopologyGoalSearchRecord] = Field(
        default_factory=list, max_length=4
    )


class TopoMemoryRoad(SchemaModel):
    """ABot-N0 Road layer: traversable connectivity between stable regions."""

    source: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    target: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    distance_m: float | None = Field(default=None, ge=0.0)
    transition: str = Field(default="connected navmesh route", max_length=120)


class TopoMemoryFunction(SchemaModel):
    """ABot-N0 Function layer: what a region is used for and affords."""

    topology_node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    function: str = Field(default="", max_length=200)
    task_affordances: list[str] = Field(default_factory=list, max_length=8)


class TopoMemoryObjectPOI(SchemaModel):
    """ABot-N0 Object/POI layer: semantic landmarks attached to map nodes."""

    topology_node_id: str = Field(pattern=r"^backbone-[0-9]{3,}$")
    objects: list[str] = Field(default_factory=list, max_length=12)
    landmarks: list[str] = Field(default_factory=list, max_length=8)


class TopoMemoryView(SchemaModel):
    """Bounded four-layer Topo-Memory supplied to the Agentic Planner."""

    map_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    current_topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    block_layer: list[TopoMemoryBlock] = Field(default_factory=list, max_length=32)
    road_layer: list[TopoMemoryRoad] = Field(default_factory=list, max_length=64)
    function_layer: list[TopoMemoryFunction] = Field(default_factory=list, max_length=32)
    object_poi_layer: list[TopoMemoryObjectPOI] = Field(
        default_factory=list, max_length=32
    )


class TaskWorkingMemory(SchemaModel):
    task_id: str = Field(min_length=1)
    global_goal: str = Field(default="", max_length=240)
    target_evidence: list[str] = Field(default_factory=list, max_length=8)
    route_hypothesis: str = Field(default="", max_length=240)
    route_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rejected_branches: list[str] = Field(default_factory=list, max_length=8)
    current_place_id: str | None = None
    previous_place_id: str | None = None
    current_route_node_id: str | None = None
    previous_route_node_id: str | None = None
    global_anchor_place_id: str | None = None
    previous_assessment: dict[str, object] = Field(default_factory=dict)
    navigation_task_type: str = "instruction_following"
    cot: dict[str, object] = Field(default_factory=dict)
    last_slow_output: dict[str, object] = Field(default_factory=dict)
    last_action: dict[str, object] = Field(default_factory=dict)
    topology_goal_search: list[TopologyGoalSearchRecord] = Field(
        default_factory=list, max_length=64
    )
    revision: int = Field(default=0, ge=0)


class MemoryDelta(SchemaModel):
    revision: int = Field(ge=0)
    current_place: PlaceMemoryView | None = None
    created: bool = False
    revisited: bool = False
    distance_to_place_m: float | None = Field(default=None, ge=0.0)
    updated_fields: list[str] = Field(default_factory=list, max_length=8)


class NavigationMemoryView(SchemaModel):
    revision: int = Field(ge=0)
    task_id: str
    global_goal: str = ""
    route_hypothesis: str = ""
    route_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    global_anchor: PlaceMemoryView | None = None
    current_place: PlaceMemoryView | None = None
    backtrack_location: PlaceMemoryView | None = None
    relevant_places: list[PlaceMemoryView] = Field(default_factory=list, max_length=5)
    nearby_backbone_nodes: list[BackboneExplorationCandidate] = Field(
        default_factory=list, max_length=4
    )
    rejected_branches: list[str] = Field(default_factory=list, max_length=8)
    total_places: int = Field(default=0, ge=0)
    route_node_count: int = Field(default=0, ge=0)
    visual_archive_count: int = Field(default=0, ge=0)
    latest_subtask_id: str | None = None
    latest_subtask_keyframe_count: int = Field(default=0, ge=0, le=4)
