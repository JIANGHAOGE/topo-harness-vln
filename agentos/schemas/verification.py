from __future__ import annotations

from typing import Any

from pydantic import Field

from agentos.schemas.base import SchemaModel
from agentos.schemas.enums import VerificationAction, VerificationLevel, VerificationStatus
from agentos.schemas.skill import Evidence
from agentos.schemas.trace import TraceContext


class VerificationResult(SchemaModel):
    trace: TraceContext
    level: VerificationLevel
    status: VerificationStatus
    summary: str = Field(min_length=1)
    reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    recommended_action: VerificationAction | None = None

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.SUCCESS

    @property
    def action(self) -> VerificationAction:
        if self.recommended_action is not None:
            return self.recommended_action
        if self.status == VerificationStatus.SUCCESS:
            return VerificationAction.CONTINUE
        if self.status == VerificationStatus.RETRY:
            return VerificationAction.RETRY
        return VerificationAction.FAIL
