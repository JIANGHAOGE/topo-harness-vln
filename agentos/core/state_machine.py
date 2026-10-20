from __future__ import annotations

from agentos.schemas.enums import RuntimeState


class StateTransitionError(RuntimeError):
    pass


class StateMachine:
    _ALLOWED: dict[RuntimeState, set[RuntimeState]] = {
        RuntimeState.IDLE: {RuntimeState.PLAN, RuntimeState.CANCELLED},
        RuntimeState.PLAN: {RuntimeState.START_SKILL, RuntimeState.FAILED, RuntimeState.CANCELLED},
        RuntimeState.START_SKILL: {RuntimeState.EXECUTE_SKILL, RuntimeState.FAILED, RuntimeState.CANCELLED},
        RuntimeState.EXECUTE_SKILL: {
            RuntimeState.VERIFY_RUNTIME,
            RuntimeState.VERIFY_SKILL,
            RuntimeState.FAILED,
            RuntimeState.CANCELLED,
        },
        RuntimeState.VERIFY_RUNTIME: {
            RuntimeState.START_SKILL,
            RuntimeState.VERIFY_SKILL,
            RuntimeState.PLAN,
            RuntimeState.FAILED,
            RuntimeState.CANCELLED,
        },
        RuntimeState.VERIFY_SKILL: {
            RuntimeState.START_SKILL,
            RuntimeState.EXECUTE_SKILL,
            RuntimeState.VERIFY_FINISH,
            RuntimeState.PLAN,
            RuntimeState.FAILED,
            RuntimeState.CANCELLED,
        },
        RuntimeState.VERIFY_FINISH: {RuntimeState.DONE, RuntimeState.PLAN, RuntimeState.FAILED, RuntimeState.CANCELLED},
        RuntimeState.DONE: {RuntimeState.IDLE},
        RuntimeState.FAILED: {RuntimeState.IDLE},
        RuntimeState.CANCELLED: {RuntimeState.IDLE},
    }

    def __init__(self) -> None:
        self.state = RuntimeState.IDLE

    def transition(self, target: RuntimeState) -> tuple[RuntimeState, RuntimeState]:
        source = self.state
        if target not in self._ALLOWED[source]:
            raise StateTransitionError(f"Invalid transition: {source.value} -> {target.value}")
        self.state = target
        return source, target

    def reset(self) -> None:
        self.state = RuntimeState.IDLE
