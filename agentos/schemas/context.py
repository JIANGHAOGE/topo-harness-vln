from __future__ import annotations

from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel


class ContextEvidence(SchemaModel):
    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    score: float = Field(default=0.0, ge=0.0)
    source_ref: str | None = None


class EmbodiedContext(SchemaModel):
    """Bounded context handed to one isolated skill run."""

    instruction: str = Field(min_length=1)
    plan_summary: str = ""
    observation_summary: str = ""
    robot_state: dict[str, Any] = Field(default_factory=dict)
    recent_events: list[str] = Field(default_factory=list)
    prior_results: list[str] = Field(default_factory=list)
    retrieved_evidence: list[ContextEvidence] = Field(default_factory=list)

    def render(self) -> str:
        sections = [f"Task: {self.instruction}"]
        if self.plan_summary:
            sections.append(f"Plan: {self.plan_summary}")
        if self.observation_summary:
            sections.append(f"Observation: {self.observation_summary}")
        if self.robot_state:
            sections.append(f"Robot state: {self.robot_state}")
        if self.prior_results:
            sections.append("Prior skill results: " + " | ".join(self.prior_results))
        if self.retrieved_evidence:
            evidence = " | ".join(item.summary for item in self.retrieved_evidence)
            sections.append(f"Retrieved evidence: {evidence}")
        if self.recent_events:
            sections.append("Recent events: " + " | ".join(self.recent_events))
        return "\n".join(sections)

