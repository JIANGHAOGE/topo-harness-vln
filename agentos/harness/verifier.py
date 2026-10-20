from __future__ import annotations

from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan
from agentos.schemas.skill import SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult
from agentos.verification.base import FinishVerifier, RuntimeVerifier, SkillVerifier


class Verifier:
    """One AgentOS Verifier applying checks at three execution stages."""

    def __init__(
        self,
        *,
        runtime: RuntimeVerifier,
        skill: SkillVerifier,
        finish: FinishVerifier,
    ) -> None:
        self.runtime = runtime
        self.skill = skill
        self.finish = finish

    async def verify_runtime(
        self,
        request: SkillRequest,
        result: SkillResult,
        observation: Observation,
    ) -> VerificationResult:
        return await self.runtime.verify_runtime(request, result, observation)

    async def verify_skill(
        self,
        request: SkillRequest,
        result: SkillResult,
        observation: Observation,
    ) -> VerificationResult:
        return await self.skill.verify_skill(request, result, observation)

    async def verify_finish(
        self,
        task: TaskRequest,
        plan: Plan,
        skill_results: list[SkillResult],
        observation: Observation,
    ) -> VerificationResult:
        return await self.finish.verify_finish(task, plan, skill_results, observation)


# Compatibility for callers created before the public AgentOS terminology was
# aligned. This is one Verifier component, not three independent authorities.
VerifierStack = Verifier
