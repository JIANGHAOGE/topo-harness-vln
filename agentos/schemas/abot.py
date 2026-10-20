from __future__ import annotations

from datetime import datetime

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now
from agentos.schemas.hierarchical import CameraView, DualPixelGoal
from agentos.schemas.navigation_memory import TopoMemoryView
from agentos.schemas.task_reasoning import (
    ABotNavigationCoT,
    ABotNavigationTaskType,
    AgenticPlan,
    AgenticSubtask,
)


class ABotSlowOutput(SchemaModel):
    """ABot-N1 Slow-System decision cached before deterministic Fast execution."""

    slow_step_id: str = Field(min_length=1)
    navigation_task_type: ABotNavigationTaskType
    cot: ABotNavigationCoT | None = None
    pixel_goal: DualPixelGoal | None = None
    reference_observation_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)


class ABotPlannerContext(SchemaModel):
    """ABot-N0 long-term spatial memory and currently active typed plan."""

    topo_memory: TopoMemoryView
    agentic_plan: AgenticPlan | None = None
    active_subtask: AgenticSubtask | None = None


class ABotTargetEvidence(SchemaModel):
    """Sticky evidence for the active directly visible task target."""

    target: str = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0.0, le=1.0)
    source_observation_id: str = Field(min_length=1)
    post_observation_id: str = Field(min_length=1)
    view: CameraView
    target_pixel: list[int] = Field(min_length=2, max_length=2)
    affordance_pixel: list[int] = Field(min_length=2, max_length=2)
    source_pose: dict[str, object] | None = None
    execution_reached: bool
    remaining_distance_m: float = Field(ge=0.0)
    evidence_summary: str = Field(min_length=1, max_length=240)


class ABotSlowContext(SchemaModel):
    """Bounded ABot-N1 Slow-System input assembled at one macro boundary."""

    current_observation_id: str = Field(min_length=1)
    historical_front_observation_ids: list[str] = Field(
        default_factory=list, max_length=4
    )
    task_specification: str = Field(min_length=1, max_length=240)
    previous_decision: ABotSlowOutput | None = None
    active_target_evidence: ABotTargetEvidence | None = None
    topology_context: ABotPlannerContext


class ABotFastExecution(SchemaModel):
    """Training-free Fast-System result produced by A* and the step controller."""

    slow_step_id: str = Field(min_length=1)
    reference_observation_id: str = Field(min_length=1)
    controller: str = Field(min_length=1)
    reached: bool
    control_steps: int = Field(ge=0)
    distance_m: float = Field(ge=0.0)
    remaining_distance_m: float = Field(ge=0.0)
    stalled: bool
