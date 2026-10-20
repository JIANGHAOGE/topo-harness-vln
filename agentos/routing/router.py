from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import Field

from agentos.schemas.base import SchemaModel


class ModelRoute(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


class ModelRouteRequest(SchemaModel):
    purpose: str = Field(min_length=1)
    complexity: int = Field(default=1, ge=0, le=10)
    modalities: list[str] = Field(default_factory=list)
    latency_budget_ms: int | None = Field(default=None, gt=0)
    privacy_sensitive: bool = False
    edge_available: bool = True
    cloud_available: bool = True


class ModelRouteDecision(SchemaModel):
    route: ModelRoute
    reason: str = Field(min_length=1)
    fallback: ModelRoute | None = None


class ModelRouter(Protocol):
    def route(self, request: ModelRouteRequest) -> ModelRouteDecision:
        ...


class RuleBasedModelRouter:
    """Deterministic edge/cloud policy matching ABot-AgentOS routing boundaries."""

    def __init__(self, *, cloud_complexity_threshold: int = 6, tight_latency_ms: int = 800) -> None:
        self.cloud_complexity_threshold = cloud_complexity_threshold
        self.tight_latency_ms = tight_latency_ms

    def route(self, request: ModelRouteRequest) -> ModelRouteDecision:
        if request.privacy_sensitive and request.edge_available:
            return ModelRouteDecision(route=ModelRoute.EDGE, reason="Privacy-sensitive request stays on edge.")
        if request.latency_budget_ms is not None and request.latency_budget_ms <= self.tight_latency_ms and request.edge_available:
            return ModelRouteDecision(route=ModelRoute.EDGE, reason="Latency budget requires edge execution.")
        if request.complexity >= self.cloud_complexity_threshold and request.cloud_available:
            fallback = ModelRoute.EDGE if request.edge_available else None
            return ModelRouteDecision(route=ModelRoute.CLOUD, fallback=fallback, reason="Complex reasoning is routed to cloud.")
        if request.edge_available:
            fallback = ModelRoute.CLOUD if request.cloud_available else None
            return ModelRouteDecision(route=ModelRoute.EDGE, fallback=fallback, reason="Routine turn handled by edge model.")
        if request.cloud_available:
            return ModelRouteDecision(route=ModelRoute.CLOUD, reason="Edge model unavailable.")
        raise RuntimeError("No model route is available")

