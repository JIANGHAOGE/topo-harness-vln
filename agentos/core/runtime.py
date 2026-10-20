from __future__ import annotations

import uuid
import time
from dataclasses import dataclass
from datetime import timedelta
from collections.abc import Awaitable, Callable

from agentos.context.manager import BoundedContextManager, ContextManager
from agentos.core.events import InMemoryEventBus
from agentos.core.state_machine import StateMachine
from agentos.core.trace import is_current_trace
from agentos.evolution.runtime import SkillEvolvement
from agentos.harness.main_agent import MainAgent
from agentos.harness.skill_runner import IsolatedSkillRunner, SkillRunnerOutcome
from agentos.harness.verifier import Verifier
from agentos.memory.writer import MemoryWriter
from agentos.planning.base import Planner, Replanner
from agentos.schemas.base import utc_now
from agentos.schemas.enums import (
    EventType,
    RuntimeState,
    SkillStatus,
    VerificationAction,
    VerificationLevel,
    VerificationStatus,
)
from agentos.schemas.events import Event
from agentos.schemas.observation import Observation
from agentos.schemas.planning import Plan, PlanStep
from agentos.schemas.skill import RetryPolicy, SkillRequest, SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.trace import TraceContext
from agentos.schemas.verification import VerificationResult
from agentos.skills.base import CancellationToken
from agentos.skills.registry import SkillRegistry
from agentos.verification.base import FinishVerifier, RuntimeVerifier, SkillVerifier
from agentos.verification.simple import SimpleRuntimeVerifier


@dataclass
class RuntimeResult:
    state: RuntimeState
    task: TaskRequest
    plan: Plan | None
    skill_results: list[SkillResult]
    main_agent_results: list[SkillResult]
    finish_verification: VerificationResult | None
    events: list[Event]


@dataclass
class StepExecution:
    result: SkillResult
    main_agent_result: SkillResult
    verification: VerificationResult
    observation: Observation


class AgentRuntime:
    """Verification-aware Agent Harness above embodiment-specific skills."""

    def __init__(
        self,
        *,
        planner: Planner,
        skills: SkillRegistry,
        skill_verifier: SkillVerifier | None = None,
        finish_verifier: FinishVerifier | None = None,
        runtime_verifier: RuntimeVerifier | None = None,
        context_manager: ContextManager | None = None,
        memory_writer: MemoryWriter | None = None,
        replanner: Replanner | None = None,
        max_plan_revisions: int = 2,
        event_bus: InMemoryEventBus | None = None,
        observation_provider: Callable[[TraceContext], Awaitable[Observation]] | None = None,
        main_agent: MainAgent | None = None,
        skill_runner: IsolatedSkillRunner | None = None,
        verifier: Verifier | None = None,
        verifier_stack: Verifier | None = None,
        skill_evolvement: SkillEvolvement | None = None,
    ) -> None:
        if max_plan_revisions < 0:
            raise ValueError("max_plan_revisions must be non-negative")
        self.planner = planner
        self.replanner = replanner
        self.skills = skills
        runtime_stage = runtime_verifier or SimpleRuntimeVerifier()
        if verifier is None and verifier_stack is None:
            if skill_verifier is None or finish_verifier is None:
                raise ValueError("A Verifier or both skill and finish verification stages are required")
            verifier = Verifier(
                runtime=runtime_stage,
                skill=skill_verifier,
                finish=finish_verifier,
            )
        self.verifier = verifier or verifier_stack
        assert self.verifier is not None
        self.runtime_verifier = self.verifier.runtime
        self.skill_verifier = self.verifier.skill
        self.finish_verifier = self.verifier.finish
        self.main_agent = main_agent or MainAgent(planner, replanner)
        self.skill_runner = skill_runner or IsolatedSkillRunner(skills)
        self.memory_writer = memory_writer
        memory = memory_writer.store if memory_writer is not None else None
        self.context_manager = context_manager or BoundedContextManager(memory)
        self.skill_evolvement = skill_evolvement
        self.max_plan_revisions = max_plan_revisions
        self.event_bus = event_bus or InMemoryEventBus()
        self.observation_provider = observation_provider
        self.state_machine = StateMachine()
        self.cancellation = CancellationToken()
        self._active_trace: TraceContext | None = None

    def cancel(self, reason: str = "cancelled") -> None:
        self.cancellation.cancel(reason)

    def accepts_trace(self, trace: TraceContext) -> bool:
        return is_current_trace(self._active_trace, trace)

    async def run_task(self, task: TaskRequest, observation: Observation) -> RuntimeResult:
        if self.state_machine.state in {RuntimeState.DONE, RuntimeState.FAILED, RuntimeState.CANCELLED}:
            self.state_machine.reset()
        self.cancellation = CancellationToken()
        self._active_trace = None
        plan: Plan | None = None
        skill_results: list[SkillResult] = []
        main_agent_results: list[SkillResult] = []
        plan_results: list[SkillResult] = []
        finish_result: VerificationResult | None = None
        revisions_used = 0
        current_observation = observation

        await self._transition(RuntimeState.PLAN, task.trace)
        await self.event_bus.emit(event_type=EventType.TASK_RECEIVED, trace=task.trace, state=self.state_machine.state)
        await self._remember(task.trace, "task", lambda: self.memory_writer.record_task(task) if self.memory_writer else None)
        await self._remember(
            task.trace,
            "observation",
            lambda: self.memory_writer.record_observation(task, observation) if self.memory_writer else None,
        )

        try:
            plan = await self._create_plan(task, current_observation)

            while True:
                replan_reason: str | None = None
                for step in plan.steps:
                    if self.cancellation.cancelled:
                        await self._cancel(task.trace)
                        break
                    execution = await self._run_step(
                        task,
                        current_observation,
                        plan,
                        step,
                        main_agent_results,
                    )
                    current_observation = execution.observation
                    skill_results.append(execution.result)
                    main_agent_results.append(execution.main_agent_result)
                    plan_results.append(execution.main_agent_result)
                    if execution.verification.status == VerificationStatus.SUCCESS:
                        continue
                    if execution.verification.action in {VerificationAction.REPLAN, VerificationAction.RETRY}:
                        replan_reason = execution.verification.summary
                        break
                    await self._fail(execution.result.trace, execution.verification.summary)
                    return self._result(task, plan, skill_results, main_agent_results, finish_result)

                if self.state_machine.state == RuntimeState.CANCELLED:
                    return self._result(task, plan, skill_results, main_agent_results, finish_result)

                if replan_reason is not None:
                    revised = await self._replan(
                        task,
                        current_observation,
                        plan,
                        main_agent_results,
                        reason=replan_reason,
                        revisions_used=revisions_used,
                    )
                    if revised is None:
                        await self._fail(task.trace, f"Recovery unavailable: {replan_reason}")
                        return self._result(task, plan, skill_results, main_agent_results, finish_result)
                    plan = revised
                    plan_results = []
                    revisions_used += 1
                    continue

                if self._requires_main_agent_followup(plan):
                    mapping_completed = (
                        plan.metadata.get("delegated_skill") == "MappingSkill"
                    )
                    revised = await self._replan(
                        task,
                        current_observation,
                        plan,
                        main_agent_results,
                        reason=(
                            "MappingSkill succeeded. Replan from its fresh spatial evidence before "
                            "delegating navigation or attempting global finish."
                            if mapping_completed
                            else "Delegated Skill succeeded. Advance the accepted high-level plan to "
                            "its earliest pending Skill subtask before attempting global finish."
                        ),
                        revisions_used=revisions_used,
                        enforce_revision_budget=False,
                    )
                    if revised is None:
                        await self._fail(
                            task.trace,
                            "Main Agent follow-up is required after the delegated Skill, but plan "
                            "advancement is unavailable.",
                        )
                        return self._result(
                            task,
                            plan,
                            skill_results,
                            main_agent_results,
                            finish_result,
                        )
                    plan = revised
                    plan_results = []
                    continue

                await self._transition(RuntimeState.VERIFY_FINISH, task.trace)
                finish_started = time.monotonic()
                finish_result = await self.verifier.verify_finish(
                    task, plan, plan_results, current_observation
                )
                finish_duration_s = time.monotonic() - finish_started
                await self.event_bus.emit(
                    event_type=EventType.FINISH_VERIFIED,
                    trace=finish_result.trace,
                    state=self.state_machine.state,
                    payload={
                        "verification": finish_result.model_dump(mode="json"),
                        "duration_s": round(finish_duration_s, 3),
                    },
                )
                await self._remember(
                    finish_result.trace,
                    "finish_verification",
                    lambda: self.memory_writer.record_verification(task, finish_result) if self.memory_writer else None,
                )
                if finish_result.status == VerificationStatus.SUCCESS:
                    await self._transition(RuntimeState.DONE, finish_result.trace)
                    await self.event_bus.emit(
                        event_type=EventType.TASK_DONE,
                        trace=finish_result.trace,
                        state=self.state_machine.state,
                    )
                    break
                if finish_result.action == VerificationAction.REPLAN:
                    revised = await self._replan(
                        task,
                        current_observation,
                        plan,
                        main_agent_results,
                        reason=finish_result.summary,
                        revisions_used=revisions_used,
                    )
                    if revised is not None:
                        plan = revised
                        plan_results = []
                        revisions_used += 1
                        continue
                await self._fail(finish_result.trace, finish_result.summary)
                break
        finally:
            self._active_trace = None

        return self._result(task, plan, skill_results, main_agent_results, finish_result)

    async def _create_plan(self, task: TaskRequest, observation: Observation) -> Plan:
        started = time.monotonic()
        plan = await self.main_agent.plan(task, observation)
        duration_s = time.monotonic() - started
        await self.event_bus.emit(
            event_type=EventType.PLAN_CREATED,
            trace=task.trace,
            state=self.state_machine.state,
            payload={
                "plan": plan.model_dump(mode="json"),
                "duration_s": round(duration_s, 3),
            },
        )
        return plan

    async def _replan(
        self,
        task: TaskRequest,
        observation: Observation,
        previous_plan: Plan,
        skill_results: list[SkillResult],
        *,
        reason: str,
        revisions_used: int,
        enforce_revision_budget: bool = True,
    ) -> Plan | None:
        if self.main_agent.replanner is None:
            return None
        if enforce_revision_budget and revisions_used >= self.max_plan_revisions:
            return None
        next_revision = previous_plan.plan_revision + 1
        await self._transition(RuntimeState.PLAN, task.trace)
        started = time.monotonic()
        plan = await self.main_agent.replan(
            task,
            observation,
            previous_plan,
            skill_results,
            reason=reason,
            next_revision=next_revision,
        )
        if plan is None:
            return None
        duration_s = time.monotonic() - started
        if plan.plan_revision != next_revision:
            plan = plan.model_copy(update={"plan_revision": next_revision})
        await self.event_bus.emit(
            event_type=EventType.PLAN_REVISED,
            trace=task.trace,
            state=self.state_machine.state,
            payload={
                "reason": reason,
                "plan": plan.model_dump(mode="json"),
                "duration_s": round(duration_s, 3),
            },
        )
        return plan

    @staticmethod
    def _requires_main_agent_followup(plan: Plan) -> bool:
        if plan.metadata.get("delegated_skill") == "MappingSkill":
            return True
        subtasks = plan.metadata.get("planned_skill_subtasks")
        delegated_index = plan.metadata.get("delegated_subtask_index")
        return (
            isinstance(subtasks, list)
            and isinstance(delegated_index, int)
            and 0 <= delegated_index < len(subtasks) - 1
        )

    async def _run_step(
        self,
        task: TaskRequest,
        observation: Observation,
        plan: Plan,
        step: PlanStep,
        prior_results: list[SkillResult],
    ) -> StepExecution:
        attempts = step.max_retries + 1
        last_result: SkillResult | None = None
        last_main_agent_result: SkillResult | None = None
        last_verification: VerificationResult | None = None
        for attempt in range(attempts):
            trace = self._skill_trace(task.trace, observation.trace, plan.plan_revision, step.timeout_s)
            context = self.context_manager.build(
                task=task,
                observation=observation,
                plan=plan,
                skill_results=prior_results,
                events=self.event_bus.snapshot(),
            )
            request = SkillRequest(
                trace=trace,
                skill_name=step.skill_name,
                command=step.command,
                success_conditions=step.success_conditions,
                context_summary=context.render(),
                inputs=step.inputs,
                timeout_s=step.timeout_s,
                retry_policy=RetryPolicy(max_retries=step.max_retries),
            )
            await self.event_bus.emit(
                event_type=EventType.CONTEXT_BUILT,
                trace=trace,
                state=self.state_machine.state,
                payload={
                    "skill_name": step.skill_name,
                    "evidence_node_ids": [item.node_id for item in context.retrieved_evidence],
                },
            )
            self._active_trace = trace
            await self._transition(RuntimeState.START_SKILL, trace)
            await self.event_bus.emit(
                event_type=EventType.SKILL_STARTED,
                trace=trace,
                state=self.state_machine.state,
                payload={"skill_name": step.skill_name, "attempt": attempt},
            )
            await self._transition(RuntimeState.EXECUTE_SKILL, trace)
            runner_outcome = await self._execute_skill(request, observation, attempt)
            result = runner_outcome.raw_result
            main_agent_result = runner_outcome.main_agent_result
            post_observation = await self._observe_after_skill(task, result.trace, observation)
            await self._transition(RuntimeState.VERIFY_RUNTIME, trace)

            if not self.accepts_trace(result.trace):
                await self.event_bus.emit(
                    event_type=EventType.STALE_RESULT_DROPPED,
                    trace=result.trace,
                    state=self.state_machine.state,
                    payload={"active_trace": trace.model_dump(mode="json")},
                )
                last_result = result.model_copy(update={"status": SkillStatus.STALE})
                last_main_agent_result = last_result.model_copy(update={"outputs": {}})
                last_verification = self._verification_failure(
                    result.trace,
                    VerificationLevel.RUNTIME,
                    "Stale skill result was rejected.",
                )
                continue

            await self.event_bus.emit(
                event_type=EventType.SKILL_RESULT,
                trace=result.trace,
                state=self.state_machine.state,
                payload={"result": result.model_dump(mode="json")},
            )
            await self._remember(
                result.trace,
                "skill_result",
                lambda: self.memory_writer.record_skill_result(task, result) if self.memory_writer else None,
            )
            runtime_started = time.monotonic()
            runtime_verification = await self.verifier.verify_runtime(request, result, post_observation)
            runtime_duration_s = time.monotonic() - runtime_started
            await self.event_bus.emit(
                event_type=EventType.RUNTIME_VERIFIED,
                trace=runtime_verification.trace,
                state=self.state_machine.state,
                payload={
                    "verification": runtime_verification.model_dump(mode="json"),
                    "duration_s": round(runtime_duration_s, 3),
                },
            )
            await self._remember(
                runtime_verification.trace,
                "runtime_verification",
                lambda: self.memory_writer.record_verification(task, runtime_verification) if self.memory_writer else None,
            )
            last_result = result
            last_main_agent_result = main_agent_result
            last_verification = runtime_verification
            if runtime_verification.status != VerificationStatus.SUCCESS:
                if runtime_verification.status == VerificationStatus.RETRY and attempt + 1 < attempts:
                    continue
                break

            await self._transition(RuntimeState.VERIFY_SKILL, trace)
            skill_verify_started = time.monotonic()
            skill_verification = await self.verifier.verify_skill(request, result, post_observation)
            skill_verify_duration_s = time.monotonic() - skill_verify_started
            await self.event_bus.emit(
                event_type=EventType.SKILL_VERIFIED,
                trace=skill_verification.trace,
                state=self.state_machine.state,
                payload={
                    "verification": skill_verification.model_dump(mode="json"),
                    "skill_name": result.skill_name,
                    "semantic_completion": (result.outputs or {}).get(
                        "semantic_completion"
                    ),
                    "topology_goal_search": (result.outputs or {}).get(
                        "topology_goal_search"
                    ),
                    "duration_s": round(skill_verify_duration_s, 3),
                },
            )
            await self._remember(
                skill_verification.trace,
                "skill_verification",
                lambda: self.memory_writer.record_verification(task, skill_verification) if self.memory_writer else None,
            )
            last_verification = skill_verification
            if skill_verification.status == VerificationStatus.SUCCESS:
                self._active_trace = None
                return StepExecution(
                    result=result,
                    main_agent_result=main_agent_result,
                    verification=skill_verification,
                    observation=post_observation,
                )
            if skill_verification.status != VerificationStatus.RETRY:
                break

        self._active_trace = None
        result = last_result or self._failure_result(task.trace, step.skill_name, "Skill did not produce a result")
        main_agent_result = last_main_agent_result or self._compact_result(result)
        verification = last_verification or self._verification_failure(
            result.trace,
            VerificationLevel.SKILL,
            "Skill did not produce a verifiable result.",
        )
        return StepExecution(
            result=result,
            main_agent_result=main_agent_result,
            verification=verification,
            observation=post_observation,
        )

    async def _observe_after_skill(
        self,
        task: TaskRequest,
        result_trace: TraceContext,
        fallback: Observation,
    ) -> Observation:
        if self.observation_provider is None:
            return fallback
        observation = await self.observation_provider(result_trace)
        await self._remember(
            result_trace,
            "observation",
            lambda: self.memory_writer.record_observation(task, observation) if self.memory_writer else None,
        )
        return observation

    async def _execute_skill(
        self,
        request: SkillRequest,
        observation: Observation,
        attempt: int,
    ) -> SkillRunnerOutcome:
        return await self.skill_runner.run(
            request,
            observation,
            event_bus=self.event_bus,
            cancellation=self.cancellation,
            attempt=attempt,
        )

    def _skill_trace(
        self,
        task_trace: TraceContext,
        observation_trace: TraceContext,
        plan_revision: int,
        timeout_s: float,
    ) -> TraceContext:
        now = utc_now()
        deadline = min(task_trace.deadline, now + timedelta(seconds=timeout_s))
        return task_trace.for_skill(
            skill_run_id=uuid.uuid4().hex,
            plan_revision=plan_revision,
            observation_id=observation_trace.observation_id,
            frame_timestamp=observation_trace.frame_timestamp,
            pose_timestamp=observation_trace.pose_timestamp,
            deadline=deadline,
        )

    def _failure_result(
        self,
        trace: TraceContext,
        skill_name: str,
        summary: str,
        status: SkillStatus = SkillStatus.FAILURE,
    ) -> SkillResult:
        return SkillResult(trace=trace, skill_name=skill_name, status=status, summary=summary, error=summary)

    @staticmethod
    def _compact_result(result: SkillResult) -> SkillResult:
        summary = result.compressed_history or result.summary
        return result.model_copy(
            update={
                "summary": summary,
                "compressed_history": summary,
                "outputs": {},
            }
        )

    @staticmethod
    def _verification_failure(
        trace: TraceContext,
        level: VerificationLevel,
        summary: str,
    ) -> VerificationResult:
        return VerificationResult(
            trace=trace,
            level=level,
            status=VerificationStatus.FAILURE,
            summary=summary,
        )

    async def _remember(self, trace: TraceContext, record_type: str, write: Callable[[], str | None]) -> None:
        if self.memory_writer is None:
            return
        try:
            node_id = write()
        except Exception as exc:
            await self.event_bus.emit(
                event_type=EventType.MEMORY_WRITE_FAILED,
                trace=trace,
                state=self.state_machine.state,
                payload={"record_type": record_type, "error": f"{type(exc).__name__}: {exc}"},
            )
            return
        await self.event_bus.emit(
            event_type=EventType.MEMORY_WRITTEN,
            trace=trace,
            state=self.state_machine.state,
            payload={"record_type": record_type, "node_id": node_id},
        )

    async def _transition(self, target: RuntimeState, trace: TraceContext) -> None:
        source, dest = self.state_machine.transition(target)
        await self.event_bus.emit(
            event_type=EventType.STATE_TRANSITION,
            trace=trace,
            state=dest,
            payload={"from": source.value, "to": dest.value},
        )

    async def _cancel(self, trace: TraceContext) -> None:
        await self._transition(RuntimeState.CANCELLED, trace)
        await self.event_bus.emit(
            event_type=EventType.TASK_CANCELLED,
            trace=trace,
            state=self.state_machine.state,
            payload={"reason": self.cancellation.reason},
        )

    async def _fail(self, trace: TraceContext, reason: str) -> None:
        await self._transition(RuntimeState.FAILED, trace)
        await self.event_bus.emit(
            event_type=EventType.TASK_FAILED,
            trace=trace,
            state=self.state_machine.state,
            payload={"reason": reason},
        )

    def _result(
        self,
        task: TaskRequest,
        plan: Plan | None,
        skill_results: list[SkillResult],
        main_agent_results: list[SkillResult],
        finish_result: VerificationResult | None,
    ) -> RuntimeResult:
        result = RuntimeResult(
            state=self.state_machine.state,
            task=task,
            plan=plan,
            skill_results=skill_results,
            main_agent_results=main_agent_results,
            finish_verification=finish_result,
            events=self.event_bus.snapshot(),
        )
        if self.skill_evolvement is not None:
            self.skill_evolvement.record_trace(
                task=task,
                state=result.state,
                plan=plan,
                skill_results=main_agent_results,
                finish_verification=finish_result,
                events=result.events,
            )
        return result
