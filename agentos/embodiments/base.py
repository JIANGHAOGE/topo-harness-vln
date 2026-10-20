from __future__ import annotations

from typing import Protocol

from agentos.schemas.observation import Observation
from agentos.schemas.skill import SkillResult
from agentos.schemas.trace import TraceContext


class EmbodimentAdapter(Protocol):
    async def observe(self, trace: TraceContext) -> Observation:
        ...

    async def apply_skill_result(self, result: SkillResult) -> dict[str, object]:
        ...

