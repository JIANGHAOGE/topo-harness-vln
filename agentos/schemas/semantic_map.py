from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from agentos.schemas.base import SchemaModel


SemanticTerm = Annotated[str, Field(min_length=1, max_length=64)]


class NodeSemantics(SchemaModel):
    """Orientation-invariant human-style knowledge attached to one map node."""

    place_type: str = Field(min_length=1, max_length=64)
    function: str = Field(default="", max_length=200)
    objects: list[SemanticTerm] = Field(default_factory=list, max_length=12)
    landmarks: list[SemanticTerm] = Field(default_factory=list, max_length=8)
    spatial_character: str = Field(default="", max_length=200)
    task_affordances: list[SemanticTerm] = Field(default_factory=list, max_length=8)
    aliases: list[SemanticTerm] = Field(default_factory=list, max_length=8)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: Literal["model", "human", "merged"] = "model"
    notes: str = Field(default="", max_length=200)

    def summary(self) -> str:
        detail = self.function or self.spatial_character
        return f"{self.place_type}: {detail}"[:160] if detail else self.place_type

    def search_terms(self) -> list[str]:
        return _unique(
            [
                self.place_type,
                *self.aliases,
                *self.objects,
                *self.landmarks,
                *self.task_affordances,
            ],
            24,
        )


class EditableSemanticMap(SchemaModel):
    schema_version: Literal["node-semantic-map-v1"] = "node-semantic-map-v1"
    map_id: str = Field(min_length=1)
    geometry_revision: str = Field(min_length=1)
    topology_cache_key: str = Field(default="", max_length=80)
    nodes: dict[str, NodeSemantics] = Field(default_factory=dict)

    @field_validator("nodes")
    @classmethod
    def validate_node_ids(cls, value: dict[str, NodeSemantics]) -> dict[str, NodeSemantics]:
        invalid = [node_id for node_id in value if not _valid_node_id(node_id)]
        if invalid:
            raise ValueError(f"invalid backbone node ids: {', '.join(sorted(invalid))}")
        return value


class SemanticMapSearchInput(SchemaModel):
    query: str = Field(min_length=1, max_length=160)
    limit: int = Field(default=5, ge=1, le=10)


def _valid_node_id(node_id: str) -> bool:
    return (
        len(node_id) == len("backbone-000")
        and node_id.startswith("backbone-")
        and node_id.removeprefix("backbone-").isdigit()
    )


def _unique(items: list[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key and key not in seen:
            result.append(item)
            seen.add(key)
        if len(result) >= limit:
            break
    return result
