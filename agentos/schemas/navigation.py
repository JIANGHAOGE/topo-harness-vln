from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now


class NavigationMode(str, Enum):
    INSTRUCTION = "instruction"
    OBJECT_GOAL = "object_goal"
    POINT_GOAL = "point_goal"
    ROUTE = "route"
    EXPLORE = "explore"


class AnchorType(str, Enum):
    LANDMARK = "landmark"
    IMAGE_POINT = "image_point"
    ROBOT_POINT = "robot_point"
    WORLD_POINT = "world_point"
    ROUTE_CORRIDOR = "route_corridor"
    TRACKED_ENTITY = "tracked_entity"
    FRONTIER = "frontier"


class ObservationPolicy(SchemaModel):
    """Task-scoped sensor selection without putting raw history in the prompt."""

    required_views: list[str] = Field(default_factory=lambda: ["front"])
    required_modalities: list[str] = Field(default_factory=lambda: ["rgb", "pose"])
    history_horizon_s: float = Field(default=5.0, ge=0.0)
    max_frames: int = Field(default=8, ge=1, le=128)
    token_budget: int = Field(default=4096, ge=128)
    max_observation_age_s: float = Field(default=1.0, gt=0.0)
    camera_weights: dict[str, float] = Field(default_factory=dict)
    sampling_mode: str = Field(default="recent", min_length=1)


class SpatialAnchor(SchemaModel):
    anchor_type: AnchorType
    value: dict[str, Any]
    frame: str = Field(min_length=1)
    source_observation_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=utc_now)
    valid_until: datetime | None = None


class NavigationIntent(SchemaModel):
    """Structured boundary between semantic planning and a navigation skill."""

    mode: NavigationMode
    semantic_goal: str = Field(min_length=1)
    success_conditions: list[str] = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    anchors: list[SpatialAnchor] = Field(default_factory=list)
    observation_policy: ObservationPolicy = Field(default_factory=ObservationPolicy)
    priority: int = Field(default=5, ge=0, le=10)
    plan_revision: int = Field(default=0, ge=0)


class ExecutionReport(SchemaModel):
    """Measured embodiment outcome; predictions alone are not execution evidence."""

    command_ids: list[str] = Field(default_factory=list)
    controller: str = Field(min_length=1)
    distance_m: float = Field(default=0.0, ge=0.0)
    yaw_change_rad: float = 0.0
    collision_count: int = Field(default=0, ge=0)
    stale_observation_count: int = Field(default=0, ge=0)
    observation_age_s: float | None = Field(default=None, ge=0.0)
    final_pose: dict[str, Any] = Field(default_factory=dict)
    stopped: bool = False
    stop_source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbodimentExecutionTelemetry(SchemaModel):
    """Simulator- and robot-neutral motion outcome exposed above an adapter."""

    state: Literal[
        "idle",
        "progressing",
        "reached",
        "blocked",
        "stalled",
        "cancelled",
        "failed",
    ]
    goal_reached: bool = False
    motion_attempted: bool = False
    progress_m: float = Field(default=0.0, ge=0.0)
    remaining_distance_m: float | None = Field(default=None, ge=0.0)
    stalled: bool = False
    termination_reason: str = Field(default="", max_length=120)
    source: str = Field(min_length=1)
