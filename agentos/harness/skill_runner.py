from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agentos.core.events import InMemoryEventBus
from agentos.schemas.base import utc_now
from agentos.schemas.enums import EventType, RuntimeState, SkillStatus
from agentos.schemas.harness import SkillRunnerSession
from agentos.schemas.observation import Observation
from agentos.schemas.skill import SkillRequest, SkillResult
from agentos.skills.base import CancellationToken, SkillContext
from agentos.skills.registry import SkillRegistry


@dataclass
class SkillRunnerOutcome:
    raw_result: SkillResult
    main_agent_result: SkillResult
    session: SkillRunnerSession


class IsolatedSkillRunner:
    """Run a procedural skill with private progress history and bounded output."""

    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills

    async def run(
        self,
        request: SkillRequest,
        observation: Observation,
        *,
        event_bus: InMemoryEventBus,
        cancellation: CancellationToken,
        attempt: int,
    ) -> SkillRunnerOutcome:
        session = SkillRunnerSession(
            task_id=request.trace.task_id,
            skill_run_id=request.trace.skill_run_id,
            skill_name=request.skill_name,
            subgoal=request.command,
            success_conditions=request.success_conditions,
            attempt=attempt,
            source_observation_id=observation.trace.observation_id,
        )
        await event_bus.emit(
            event_type=EventType.SKILL_CONTEXT_OPENED,
            trace=request.trace,
            state=RuntimeState.START_SKILL,
            payload={"session": session.model_dump(mode="json")},
        )

        local_bus = InMemoryEventBus()

        async def forward_local_event(event) -> None:
            if event.event_type == EventType.SKILL_PROGRESS:
                summary = str(event.payload.get("summary", "")).strip()
                if summary:
                    session.progress_summaries.append(summary)
            await event_bus.publish(event)

        local_bus.subscribe(forward_local_event)
        context = SkillContext(event_bus=local_bus, cancellation=cancellation, attempt=attempt)
        skill = self.skills.get(request.skill_name)
        try:
            raw_result = await asyncio.wait_for(
                skill.run(request, observation, context),
                timeout=request.timeout_s,
            )
        except asyncio.TimeoutError:
            raw_result = self._failure_result(request, "Skill Runner timed out.", SkillStatus.TIMEOUT)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raw_result = self._failure_result(
                request,
                f"{type(exc).__name__}: {exc}",
                SkillStatus.FAILURE,
            )

        compact_summary = raw_result.compressed_history or raw_result.summary
        main_agent_outputs = {}
        handoff_builder = getattr(skill, "main_agent_handoff", None)
        if callable(handoff_builder):
            handoff = handoff_builder(raw_result)
            if not isinstance(handoff, dict):
                raise TypeError("Skill main_agent_handoff must return a dictionary")
            main_agent_outputs = handoff
        main_agent_result = raw_result.model_copy(
            update={
                "summary": compact_summary,
                "compressed_history": compact_summary,
                "outputs": main_agent_outputs,
            }
        )
        session.completed_at = utc_now()
        await event_bus.emit(
            event_type=EventType.SKILL_CONTEXT_CLOSED,
            trace=request.trace,
            state=RuntimeState.EXECUTE_SKILL,
            payload={
                "session": session.model_dump(mode="json"),
                "outcome": main_agent_result.model_dump(mode="json"),
            },
        )
        return SkillRunnerOutcome(
            raw_result=raw_result,
            main_agent_result=main_agent_result,
            session=session,
        )

    @staticmethod
    def _failure_result(
        request: SkillRequest,
        summary: str,
        status: SkillStatus,
    ) -> SkillResult:
        return SkillResult(
            trace=request.trace,
            skill_name=request.skill_name,
            status=status,
            summary=summary,
            compressed_history=summary,
            error=summary,
        )
