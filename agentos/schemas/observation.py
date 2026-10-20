from __future__ import annotations

from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel
from agentos.schemas.trace import TraceContext


class Pose2D(SchemaModel):
    x: float
    y: float
    yaw: float
    source: str = Field(default="unknown", min_length=1)
    frame: str = Field(default="unknown", min_length=1)


class Observation(SchemaModel):
    trace: TraceContext
    rgb_refs: dict[str, str] = Field(default_factory=dict)
    depth_refs: dict[str, str] = Field(default_factory=dict)
    topdown_ref: str | None = None
    pose: Pose2D | None = None
    objects: list[dict[str, Any]] = Field(default_factory=list)
    text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

