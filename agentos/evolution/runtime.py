from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Protocol

from agentos.schemas.enums import RuntimeState
from agentos.schemas.events import Event
from agentos.schemas.planning import Plan
from agentos.schemas.skill import SkillResult
from agentos.schemas.task import TaskRequest
from agentos.schemas.verification import VerificationResult


class SkillEvolvement(Protocol):
    """Capture online traces for later diagnose/compile/gate/promotion cycles."""

    def record_trace(
        self,
        *,
        task: TaskRequest,
        state: RuntimeState,
        plan: Plan | None,
        skill_results: list[SkillResult],
        finish_verification: VerificationResult | None,
        events: list[Event],
    ) -> Path | None: ...


class TraceDrivenSkillEvolvement:
    """Persist bounded traces; asset generation and promotion remain offline and gated."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_trace(
        self,
        *,
        task: TaskRequest,
        state: RuntimeState,
        plan: Plan | None,
        skill_results: list[SkillResult],
        finish_verification: VerificationResult | None,
        events: list[Event],
    ) -> Path:
        payload = {
            "schema_version": 1,
            "task": task.model_dump(mode="json"),
            "terminal_state": state.value,
            "plan": plan.model_dump(mode="json") if plan is not None else None,
            "skill_results": [
                {
                    "skill_name": result.skill_name,
                    "status": result.status.value,
                    "summary": result.compressed_history or result.summary,
                    "evidence": [item.model_dump(mode="json") for item in result.evidence],
                    "stop_reason": result.stop_reason,
                    "error": result.error,
                }
                for result in skill_results
            ],
            "finish_verification": (
                finish_verification.model_dump(mode="json")
                if finish_verification is not None
                else None
            ),
            "event_trace": [
                {
                    "event_type": event.event_type.value,
                    "state": event.state.value,
                    "created_at": event.created_at.isoformat(),
                }
                for event in events
            ],
        }
        path = self.root / f"{task.trace.episode_id}__{task.trace.task_id}.json"
        temporary = path.with_suffix(".tmp")
        with self._lock:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(path)
        return path
