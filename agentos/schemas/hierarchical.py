from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from agentos.schemas.base import SchemaModel
from agentos.schemas.observation import Pose2D


CameraView = Literal["left", "front", "right"]
ViewpointAction = Literal["turn_left", "turn_right", "move_forward", "move_backward"]
AnchorRole = Literal["affordance", "target"]
RouteNodeType = Literal["none", "open_space_center", "junction"]
SemanticLabel = Annotated[str, Field(min_length=1, max_length=48)]


class SemanticMapUpdate(SchemaModel):
    scene_summary: str = Field(min_length=1, max_length=160)
    left: list[SemanticLabel] = Field(default_factory=list, max_length=4)
    front: list[SemanticLabel] = Field(default_factory=list, max_length=4)
    right: list[SemanticLabel] = Field(default_factory=list, max_length=4)
    route_node_type: RouteNodeType = "none"
    traversable_directions: list[CameraView] = Field(default_factory=list, max_length=3)


class SceneSemanticDescription(SchemaModel):
    room_hint: str = ""
    left: list[str] = Field(default_factory=list, max_length=6)
    front: list[str] = Field(default_factory=list, max_length=6)
    right: list[str] = Field(default_factory=list, max_length=6)
    distinctive_landmarks: list[str] = Field(default_factory=list, max_length=8)


class AnchorGroundingInput(SchemaModel):
    anchor: str = Field(min_length=1)
    view: CameraView
    image_width: int = Field(default=640, ge=1)
    image_height: int = Field(default=480, ge=1)
    allowed_views: list[CameraView] = Field(default_factory=list, max_length=3)
    direction_hint: str = Field(default="", max_length=240)


class AnchorPixel(SchemaModel):
    anchor: str = Field(min_length=1)
    view: CameraView
    x_px: int = Field(ge=0)
    y_px: int = Field(ge=0)
    anchor_visible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = Field(min_length=1)
    model_coordinate_space: Literal["pixels", "normalized_1000"] = "pixels"
    bbox_px: list[int] | None = Field(default=None, min_length=4, max_length=4)
    annotated_ref: str | None = None


class QwenAnchorMark(SchemaModel):
    anchor: str = Field(min_length=1)
    panel_index: int = Field(ge=0, le=2)
    x_min: int = Field(ge=0, le=1000)
    y_min: int = Field(ge=0, le=1000)
    x_max: int = Field(ge=0, le=1000)
    y_max: int = Field(ge=0, le=1000)
    anchor_visible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = Field(min_length=1)


class PixelToPointInput(SchemaModel):
    source_observation_id: str = Field(min_length=1)
    view: CameraView
    x_px: int = Field(ge=0)
    y_px: int = Field(ge=0)
    image_width: int = Field(default=640, ge=1)
    image_height: int = Field(default=480, ge=1)
    max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)


class SpatialAnchorPoint(SchemaModel):
    source_observation_id: str = Field(min_length=1)
    view: CameraView
    pixel: list[int] = Field(min_length=2, max_length=2)
    depth_pixel: list[int] = Field(min_length=2, max_length=2)
    depth_m: float = Field(gt=0.0)
    camera_point_right_up_forward_m: list[float] = Field(min_length=3, max_length=3)
    robot_point_forward_left_m: list[float] = Field(min_length=2, max_length=2)
    world_point_xyz_m: list[float] = Field(min_length=3, max_length=3)
    navigation_candidate_xyz_m: list[float] = Field(min_length=3, max_length=3)
    raw_depth_m: float | None = Field(default=None, gt=0.0)
    depth_limited: bool = False
    max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)


class PathPlanningInput(SchemaModel):
    world_point_xyz_m: list[float] = Field(min_length=3, max_length=3)


class PlannedPath(SchemaModel):
    requested_goal_xyz_m: list[float] = Field(min_length=3, max_length=3)
    snapped_goal_xyz_m: list[float] = Field(min_length=3, max_length=3)
    world_path_xyz_m: list[list[float]] = Field(min_length=1)
    robot_path_forward_left_m: list[list[float]] = Field(min_length=1)
    geodesic_distance_m: float = Field(ge=0.0)
    waypoint_count: int = Field(ge=1)


class TriViewPixelGoal(SchemaModel):
    """ABot-N1 pixel notation: one selected pixel or an empty list per view."""

    left: list[int] = Field(default_factory=list, max_length=2)
    front: list[int] = Field(default_factory=list, max_length=2)
    right: list[int] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_pixel_slots(self) -> "TriViewPixelGoal":
        populated = 0
        for view in ("left", "front", "right"):
            pixel = getattr(self, view)
            if pixel and len(pixel) != 2:
                raise ValueError(f"{view} pixel must be [] or [x, y]")
            if pixel:
                if any(value < 0 for value in pixel):
                    raise ValueError(f"{view} pixel coordinates must be non-negative")
                populated += 1
        if populated > 1:
            raise ValueError("a tri-view pixel goal may populate only one camera view")
        return self

    @classmethod
    def from_view(cls, view: CameraView | None, pixel: list[int] | None) -> "TriViewPixelGoal":
        if view is None or pixel is None:
            return cls()
        return cls.model_validate({view: pixel})


class DualPixelGoal(SchemaModel):
    """ABot-N1 affordance and target pixels referenced to one synchronized tri-view."""

    source_observation_id: str = Field(min_length=1)
    affordance_pixel: TriViewPixelGoal = Field(default_factory=TriViewPixelGoal)
    target_pixel: TriViewPixelGoal = Field(default_factory=TriViewPixelGoal)
    affordance_world_point_xyz_m: list[float] | None = Field(
        default=None, min_length=3, max_length=3
    )


class PathExecutionInput(SchemaModel):
    world_path_xyz_m: list[list[float]] = Field(min_length=1)
    max_control_steps: int = Field(default=64, ge=1, le=1000)
    goal_tolerance_m: float = Field(default=0.3, gt=0.0, le=2.0)
    control_period_s: float = Field(default=0.1, ge=0.0, le=1.0)


class TrajectoryKeyframe(SchemaModel):
    frame_index: int = Field(ge=0)
    observation_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    pose: dict[str, object]
    map_id: str = Field(min_length=1)
    pose_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    navigation_point_xyz_m: list[float] = Field(min_length=3, max_length=3)
    rgb_refs: dict[CameraView, str]
    location_id: str | None = Field(default=None, pattern=r"^location-[0-9]{3,}$")


class TrajectoryWaypoint(SchemaModel):
    frame_index: int = Field(ge=0)
    observation_id: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    pose: dict[str, object]
    map_id: str = Field(min_length=1)
    pose_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    navigation_point_xyz_m: list[float] = Field(min_length=3, max_length=3)


class PathExecutionReport(SchemaModel):
    controller: Literal["habitat_greedy_follower"] = "habitat_greedy_follower"
    reached: bool
    control_steps: int = Field(ge=0)
    distance_m: float = Field(ge=0.0)
    remaining_distance_m: float = Field(ge=0.0)
    collisions: int = Field(ge=0)
    stalled: bool
    pose_after: dict[str, float]
    last_control: dict[str, object] | None = None
    trajectory_waypoints: list[TrajectoryWaypoint] = Field(default_factory=list, max_length=1001)
    trajectory_keyframes: list[TrajectoryKeyframe] = Field(default_factory=list, max_length=4)


class HierarchicalNavigateInput(SchemaModel):
    slow_step_id: str | None = Field(default=None, min_length=1)
    anchor: str = Field(min_length=1)
    anchor_role: AnchorRole = "affordance"
    view: CameraView
    observation_id: str | None = None
    semantic_map: SemanticMapUpdate
    task_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    scene_memory: SceneSemanticDescription | None = None
    enforce_regional_direction: bool = False
    low_confidence_recovery: Literal["adjust_viewpoint", "previous_location"] = "adjust_viewpoint"
    recovery_viewpoint_action: ViewpointAction | None = None
    max_control_steps: int = Field(default=64, ge=1, le=1000)
    goal_tolerance_m: float = Field(default=0.3, gt=0.0, le=2.0)
    anchor_max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)
    target_pixel: list[int] | None = Field(default=None, min_length=2, max_length=2)
    target_bbox_px: list[int] | None = Field(default=None, min_length=4, max_length=4)
    target_pixel_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    target_pixel_evidence: str | None = Field(default=None, min_length=1, max_length=240)
    n1_target_view: CameraView | None = None
    n1_target_pixel: list[int] | None = Field(default=None, min_length=2, max_length=2)
    n1_target_label: str | None = Field(default=None, min_length=1, max_length=120)
    n1_target_bbox_px: list[int] | None = Field(default=None, min_length=4, max_length=4)
    n1_target_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    n1_affordance_view: CameraView | None = None
    n1_affordance_pixel: list[int] | None = Field(default=None, min_length=2, max_length=2)
    n1_affordance_label: str | None = Field(default=None, min_length=1, max_length=120)
    n1_affordance_bbox_px: list[int] | None = Field(default=None, min_length=4, max_length=4)
    n1_affordance_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_slow_brain_target_pixel(self) -> "HierarchicalNavigateInput":
        if re.fullmatch(
            r"(?:backbone|location)-[0-9]{3,}",
            self.anchor.strip(),
            re.IGNORECASE,
        ):
            raise ValueError(
                "anchor must describe a visual target or traversable affordance, not a topology/location ID"
            )
        supplied = (
            self.target_pixel is not None,
            self.target_pixel_confidence is not None,
            self.target_pixel_evidence is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "target_pixel, target_pixel_confidence, and target_pixel_evidence must be supplied together"
            )
        if self.target_bbox_px is not None and self.target_pixel is None:
            raise ValueError("target_bbox_px requires target_pixel")
        if (self.n1_target_view is None) != (self.n1_target_pixel is None):
            raise ValueError("n1_target_view and n1_target_pixel must be supplied together")
        target_annotation = (
            self.n1_target_label,
            self.n1_target_bbox_px,
            self.n1_target_confidence,
        )
        if any(value is not None for value in target_annotation) and not all(
            value is not None for value in target_annotation
        ):
            raise ValueError(
                "n1_target_label, n1_target_bbox_px, and n1_target_confidence must be supplied together"
            )
        if any(value is not None for value in target_annotation) and self.n1_target_view is None:
            raise ValueError("N1 target annotation requires n1_target_view and n1_target_pixel")
        if (self.n1_affordance_view is None) != (self.n1_affordance_pixel is None):
            raise ValueError(
                "n1_affordance_view and n1_affordance_pixel must be supplied together"
            )
        affordance_annotation = (
            self.n1_affordance_label,
            self.n1_affordance_bbox_px,
            self.n1_affordance_confidence,
        )
        if any(value is not None for value in affordance_annotation) and not all(
            value is not None for value in affordance_annotation
        ):
            raise ValueError(
                "n1_affordance_label, n1_affordance_bbox_px, and "
                "n1_affordance_confidence must be supplied together"
            )
        if (
            any(value is not None for value in affordance_annotation)
            and self.n1_affordance_view is None
        ):
            raise ValueError(
                "N1 affordance annotation requires n1_affordance_view and n1_affordance_pixel"
            )
        return self


class ViewpointAdjustmentInput(SchemaModel):
    slow_step_id: str | None = Field(default=None, min_length=1)
    observation_id: str = Field(min_length=1)
    action: ViewpointAction
    rotation_degrees: float = Field(default=90.0, ge=15.0, le=180.0)
    semantic_map: SemanticMapUpdate


class ViewpointAdjustmentReport(SchemaModel):
    slow_step_id: str | None = Field(default=None, min_length=1)
    action: ViewpointAction
    collided: bool
    distance_m: float = Field(ge=0.0)
    control_steps: int = Field(default=1, ge=1)
    requested_rotation_degrees: float = Field(default=0.0, ge=0.0, le=180.0)
    rotation_degrees: float = Field(default=0.0, ge=0.0, le=180.0)
    pose_before: dict[str, float]
    pose_after: dict[str, float]


class LocalizationMapReport(SchemaModel):
    observation_id: str = Field(min_length=1)
    pose: Pose2D
    topdown_ref: str = Field(min_length=1)
    map_frame: Literal["habitat_world_xz"] = "habitat_world_xz"
    legend: dict[str, str]


LocalizationMapInput = SemanticMapUpdate


class KnownLocationNavigateInput(SchemaModel):
    slow_step_id: str | None = Field(default=None, min_length=1)
    topology_node_id: str | None = Field(
        default=None, pattern=r"^backbone-[0-9]{3,}$"
    )
    location_id: str | None = Field(default=None, pattern=r"^location-[0-9]{3,}$")
    observation_id: str = Field(min_length=1)
    semantic_map: SemanticMapUpdate
    max_control_steps: int = Field(default=64, ge=1, le=1000)
    goal_tolerance_m: float = Field(default=0.3, gt=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_navigation_target(self) -> "KnownLocationNavigateInput":
        target_count = sum(
            value is not None for value in (self.topology_node_id, self.location_id)
        )
        if target_count != 1:
            raise ValueError(
                "exactly one topology_node_id or location_id is required"
            )
        return self


class AnchorGenerationInput(SchemaModel):
    source_observation_id: str = Field(min_length=1)
    anchor: str = Field(min_length=1)
    anchor_role: AnchorRole = "affordance"
    view: CameraView
    image_width: int = Field(default=640, ge=1)
    image_height: int = Field(default=480, ge=1)
    max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)
    allowed_views: list[CameraView] = Field(default_factory=list, max_length=3)
    direction_hint: str = Field(default="", max_length=240)


class GeneratedNavigationPoint(SchemaModel):
    target_id: str = Field(min_length=1)
    source: Literal["visible_anchor"] = "visible_anchor"
    source_observation_id: str = Field(min_length=1)
    anchor: str = Field(min_length=1)
    anchor_role: AnchorRole = "affordance"
    view: CameraView
    grounding_pixel: list[int] | None = Field(default=None, min_length=2, max_length=2)
    target_pixel: list[int] | None = Field(default=None, min_length=2, max_length=2)
    target_view: CameraView | None = None
    target_label: str | None = Field(default=None, min_length=1, max_length=120)
    target_bbox_px: list[int] | None = Field(default=None, min_length=4, max_length=4)
    target_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    annotation_view: CameraView | None = None
    annotation_role: AnchorRole | None = None
    bbox_px: list[int] | None = Field(default=None, min_length=4, max_length=4)
    world_point_xyz_m: list[float] | None = Field(default=None, min_length=3, max_length=3)
    anchor_visible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = Field(min_length=1)
    annotated_ref: str | None = None
    depth_m: float | None = Field(default=None, gt=0.0)
    raw_depth_m: float | None = Field(default=None, gt=0.0)
    depth_limited: bool = False
    max_depth_m: float = Field(default=5.0, gt=0.0, le=20.0)


class PointNavigationInput(SchemaModel):
    slow_step_id: str | None = Field(default=None, min_length=1)
    target_id: str = Field(min_length=1)
    world_point_xyz_m: list[float] = Field(min_length=3, max_length=3)
    max_control_steps: int = Field(default=64, ge=1, le=1000)
    goal_tolerance_m: float = Field(default=0.3, gt=0.0, le=2.0)
    control_period_s: float = Field(default=0.1, ge=0.0, le=1.0)
    target_yaw: float | None = None
    source_observation_id: str | None = None
    target_view: CameraView | None = None
    target_pixel: list[int] | None = Field(default=None, min_length=2, max_length=2)
    affordance_lookahead_m: float = Field(default=3.0, gt=0.0, le=10.0)

    @model_validator(mode="after")
    def validate_target_pixel(self) -> "PointNavigationInput":
        if (self.target_view is None) != (self.target_pixel is None):
            raise ValueError("target_view and target_pixel must be supplied together")
        if self.source_observation_id is None and self.target_pixel is not None:
            raise ValueError("target pixels require source_observation_id")
        return self


class PointNavigationReport(PathExecutionReport):
    slow_step_id: str | None = Field(default=None, min_length=1)
    target_id: str = Field(min_length=1)
    requested_goal_xyz_m: list[float] = Field(min_length=3, max_length=3)
    snapped_goal_xyz_m: list[float] = Field(min_length=3, max_length=3)
    geodesic_distance_m: float = Field(ge=0.0)
    waypoint_count: int = Field(ge=1)
    target_yaw: float | None = None
    final_yaw: float | None = None
    orientation_error_rad: float | None = Field(default=None, ge=0.0)
    orientation_steps: int = Field(default=0, ge=0)
    pixel_goal: DualPixelGoal | None = None


class HierarchicalTaskFinishInput(SchemaModel):
    outcome: Literal["done", "failed", "cancelled"]
    observation_id: str | None = None
    scene_memory: SceneSemanticDescription | None = None
