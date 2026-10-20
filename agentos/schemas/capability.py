from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SkillDescriptor(SchemaModel):
    """Machine-readable contract exposed to the Hermes skill catalog."""

    name: str = Field(min_length=1)
    version: str = Field(default="1.0.0", min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    required_modalities: list[str] = Field(default_factory=list)
    supported_embodiments: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    side_effects: list[str] = Field(default_factory=list)
    supports_cancel: bool = True
    default_timeout_s: float = Field(default=30.0, gt=0.0)

