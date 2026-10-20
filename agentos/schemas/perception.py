from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from agentos.schemas.base import SchemaModel


CameraView = Literal["left", "front", "right"]
GroundingRole = Literal["affordance", "target"]


class VisualGroundingConstraints(SchemaModel):
    """Current-image localization hints allowed to enter a grounding provider."""

    preferred_view: CameraView | None = None
    visual_description: str = Field(default="", max_length=512)
    excluded_labels: list[str] = Field(default_factory=list, max_length=8)


class VisualGroundingInput(SchemaModel):
    query: str = Field(min_length=1)
    candidate_views: list[CameraView] = Field(
        default_factory=lambda: ["left", "front", "right"],
        min_length=1,
    )
    constraints: VisualGroundingConstraints = Field(
        default_factory=VisualGroundingConstraints
    )
    min_confidence: float = Field(default=0.90, ge=0.0, le=1.0)
    max_results: int = Field(default=3, ge=1, le=20)


class VisualGroundingProviderRequest(SchemaModel):
    observation_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    image_refs: dict[str, str] = Field(min_length=1)
    candidate_views: list[CameraView] = Field(min_length=1)
    constraints: VisualGroundingConstraints = Field(
        default_factory=VisualGroundingConstraints
    )
    max_results: int = Field(default=3, ge=1, le=20)


class VisualDetection(SchemaModel):
    label: str = Field(min_length=1)
    view: CameraView
    bbox_normalized_1000: list[int] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_summary: str = ""

    @model_validator(mode="after")
    def validate_bbox(self) -> "VisualDetection":
        xmin, ymin, xmax, ymax = self.bbox_normalized_1000
        if any(value < 0 or value > 1000 for value in self.bbox_normalized_1000):
            raise ValueError("bbox coordinates must be normalized to [0, 1000]")
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("bbox must have positive width and height")
        return self

    def bottom_center(self) -> list[int]:
        xmin, _, xmax, ymax = self.bbox_normalized_1000
        return [round((xmin + xmax) / 2.0), ymax]


class VisualGroundingProviderResponse(SchemaModel):
    provider: str = Field(min_length=1)
    detections: list[VisualDetection] = Field(default_factory=list)


class VisualGroundingOutput(SchemaModel):
    available: bool
    query: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    provider: str | None = None
    found: bool = False
    detections: list[VisualDetection] = Field(default_factory=list)
    reason: str | None = None


class VisualGroundingQuery(SchemaModel):
    role: GroundingRole
    query: str = Field(min_length=1, max_length=120)
    constraints: VisualGroundingConstraints = Field(
        default_factory=VisualGroundingConstraints
    )
    min_confidence: float = Field(default=0.90, ge=0.0, le=1.0)


class VisualGroundingBatchInput(SchemaModel):
    queries: list[VisualGroundingQuery] = Field(min_length=1, max_length=2)
    candidate_views: list[CameraView] = Field(
        default_factory=lambda: ["left", "front", "right"],
        min_length=1,
    )
    max_results_per_query: int = Field(default=1, ge=1, le=5)

    @model_validator(mode="after")
    def validate_unique_roles(self) -> "VisualGroundingBatchInput":
        roles = [query.role for query in self.queries]
        if len(roles) != len(set(roles)):
            raise ValueError("batch grounding queries must use unique roles")
        return self


class VisualGroundingBatchProviderRequest(SchemaModel):
    observation_id: str = Field(min_length=1)
    queries: list[VisualGroundingQuery] = Field(min_length=1, max_length=2)
    image_refs: dict[str, str] = Field(min_length=1)
    candidate_views: list[CameraView] = Field(min_length=1)
    max_results_per_query: int = Field(default=1, ge=1, le=5)


class VisualGroundingQueryResult(SchemaModel):
    role: GroundingRole
    query: str = Field(min_length=1, max_length=120)
    detections: list[VisualDetection] = Field(default_factory=list, max_length=5)


class VisualGroundingBatchProviderResponse(SchemaModel):
    provider: str = Field(min_length=1)
    results: list[VisualGroundingQueryResult] = Field(min_length=1, max_length=2)


class VisualGroundingBatchOutput(SchemaModel):
    available: bool
    observation_id: str = Field(min_length=1)
    provider: str | None = None
    results: list[VisualGroundingQueryResult] = Field(default_factory=list, max_length=2)
    reason: str | None = None
