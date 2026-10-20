from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentos.core.events import InMemoryEventBus
from agentos.schemas.enums import EventType, RuntimeState
from agentos.schemas.observation import Observation
from agentos.schemas.skill import SkillRequest, SkillResult


class CancellationToken:
    def __init__(self) -> None:
        self.cancelled = False
        self.reason = ""

    def cancel(self, reason: str = "cancelled") -> None:
        self.cancelled = True
        self.reason = reason

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError(self.reason or "cancelled")


@dataclass
class SkillContext:
    event_bus: InMemoryEventBus
    cancellation: CancellationToken
    attempt: int = 0

    async def emit_progress(self, request: SkillRequest, summary: str, **payload: object) -> None:
        await self.event_bus.emit(
            event_type=EventType.SKILL_PROGRESS,
            trace=request.trace,
            state=RuntimeState.EXECUTE_SKILL,
            payload={"summary": summary, **payload},
        )


class Skill(Protocol):
    name: str

    async def run(self, request: SkillRequest, observation: Observation, context: SkillContext) -> SkillResult:
        ...

