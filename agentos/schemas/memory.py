from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel, utc_now


class MemoryNodeType(str, Enum):
    SOURCE = "source"
    EVIDENCE = "evidence"
    ENTITY = "entity"
    PLACE = "place"
    SESSION = "session"
    TASK = "task"
    SKILL_RUN = "skill_run"
    SEMANTIC_EVENT = "semantic_event"


class MemoryEdgeType(str, Enum):
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    IDENTITY = "identity"
    INTERACTION = "interaction"
    PROVENANCE = "provenance"
    CONTAINS = "contains"
    OBSERVED = "observed"
    PARTICIPATED_IN = "participated_in"
    LOCATED_AT = "located_at"
    SUPERSEDES = "supersedes"


class MemoryNode(SchemaModel):
    node_id: str = Field(min_length=1)
    node_type: MemoryNodeType
    title: str = Field(min_length=1)
    content: str = ""
    source_ref: str = Field(min_length=1)
    episode_id: str | None = None
    task_id: str | None = None
    valid_from: datetime = Field(default_factory=utc_now)
    valid_to: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryEdge(SchemaModel):
    edge_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    edge_type: MemoryEdgeType
    source_ref: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class MemoryQuery(SchemaModel):
    text: str = ""
    node_types: list[MemoryNodeType] = Field(default_factory=list)
    episode_id: str | None = None
    task_id: str | None = None
    persistent_only: bool = False
    limit: int = Field(default=8, ge=1, le=1000)


class MemoryHit(SchemaModel):
    node: MemoryNode
    score: float = Field(ge=0.0)
    matched_terms: list[str] = Field(default_factory=list)
