from __future__ import annotations

from agentos.schemas.enums import SkillStatus, VerificationLevel, VerificationStatus
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan
from agentos.schemas.skill import Evidence, SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult


class SimpleRuntimeVerifier:
    async def verify_runtime(
        self,
        request: SkillRequest,
        result: SkillResult,
        observation: Observation,
    ) -> VerificationResult:
        del request, observation
        if result.status in {SkillStatus.SUCCESS, SkillStatus.RETRYABLE_FAILURE}:
            status = VerificationStatus.SUCCESS if result.status == SkillStatus.SUCCESS else VerificationStatus.RETRY
            return VerificationResult(
                trace=result.trace,
                level=VerificationLevel.RUNTIME,
                status=status,
                summary="Runtime result is structurally valid and current.",
                evidence=result.evidence,
            )
        return VerificationResult(
            trace=result.trace,
            level=VerificationLevel.RUNTIME,
            status=VerificationStatus.FAILURE,
            summary="Runtime execution failed before skill-level acceptance.",
            reasons=[result.error or result.summary],
        )


class SimpleSkillVerifier:
    async def verify_skill(
        self,
        request: SkillRequest,
        result: SkillResult,
        observation: Observation,
    ) -> VerificationResult:
        del request, observation
        if result.status == SkillStatus.SUCCESS:
            return VerificationResult(
                trace=result.trace,
                level=VerificationLevel.SKILL,
                status=VerificationStatus.SUCCESS,
                summary="Skill result accepted.",
                evidence=result.evidence,
            )
        if result.status == SkillStatus.RETRYABLE_FAILURE:
            return VerificationResult(
                trace=result.trace,
                level=VerificationLevel.SKILL,
                status=VerificationStatus.RETRY,
                summary="Skill requested retry.",
                reasons=[result.error or result.summary],
            )
        return VerificationResult(
            trace=result.trace,
            level=VerificationLevel.SKILL,
            status=VerificationStatus.FAILURE,
            summary="Skill result failed verification.",
            reasons=[result.error or result.summary],
        )


class SimpleFinishVerifier:
    async def verify_finish(
        self,
        task: TaskRequest,
        plan: Plan,
        skill_results: list[SkillResult],
        observation: Observation,
    ) -> VerificationResult:
        del plan, observation
        if task.metadata.get("force_finish_failure"):
            return VerificationResult(
                trace=task.trace,
                level=VerificationLevel.FINISH,
                status=VerificationStatus.FAILURE,
                summary="Finish verifier was forced to fail by task metadata.",
                reasons=["force_finish_failure=true"],
            )
        if not skill_results:
            return VerificationResult(
                trace=task.trace,
                level=VerificationLevel.FINISH,
                status=VerificationStatus.FAILURE,
                summary="No skill result is available for finish verification.",
            )
        if all(result.status == SkillStatus.SUCCESS for result in skill_results):
            stop_evidence = [
                Evidence(
                    kind="finish",
                    summary="All skill runs succeeded; local STOP is evidence, not authority.",
                    data={"skill_count": len(skill_results)},
                )
            ]
            return VerificationResult(
                trace=task.trace,
                level=VerificationLevel.FINISH,
                status=VerificationStatus.SUCCESS,
                summary="Global task accepted by finish verifier.",
                evidence=stop_evidence,
            )
        return VerificationResult(
            trace=task.trace,
            level=VerificationLevel.FINISH,
            status=VerificationStatus.FAILURE,
            summary="At least one skill failed before finish verification.",
        )
