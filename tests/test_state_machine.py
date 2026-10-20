from __future__ import annotations

import pytest

from agentos.core.state_machine import StateMachine, StateTransitionError
from agentos.schemas.enums import RuntimeState


def test_state_machine_success_path():
    machine = StateMachine()
    for state in [
        RuntimeState.PLAN,
        RuntimeState.START_SKILL,
        RuntimeState.EXECUTE_SKILL,
        RuntimeState.VERIFY_SKILL,
        RuntimeState.VERIFY_FINISH,
        RuntimeState.DONE,
    ]:
        machine.transition(state)
    assert machine.state == RuntimeState.DONE


def test_state_machine_rejects_invalid_transition():
    machine = StateMachine()
    with pytest.raises(StateTransitionError):
        machine.transition(RuntimeState.DONE)

