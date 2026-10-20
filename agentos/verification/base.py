from __future__ import annotations

from typing import Protocol

from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan
from agentos.schemas.skill import SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult


class RuntimeVerifier(Protocol):
    async def verify_runtime(
        self,
        request: SkillRequest,
        result: SkillResult,
        observation: Observation,
    ) -> VerificationResult:
        ...


class SkillVerifier(Protocol):
    async def verify_skill(
        self,
        request: SkillRequest,
        result: SkillResult,
        observation: Observation,
    ) -> VerificationResult:
        ...


class FinishVerifier(Protocol):
    async def verify_finish(
        self,
        task: TaskRequest,
        plan: Plan,
        skill_results: list[SkillResult],
        observation: Observation,
    ) -> VerificationResult:
        ...
