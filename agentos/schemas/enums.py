from __future__ import annotations

from enum import Enum


class RuntimeState(str, Enum):
    IDLE = "IDLE"
    PLAN = "PLAN"
    START_SKILL = "START_SKILL"
    EXECUTE_SKILL = "EXECUTE_SKILL"
    VERIFY_RUNTIME = "VERIFY_RUNTIME"
    VERIFY_SKILL = "VERIFY_SKILL"
    VERIFY_FINISH = "VERIFY_FINISH"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SkillStatus(str, Enum):
    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    STALE = "stale"


class VerificationLevel(str, Enum):
    RUNTIME = "runtime"
    SKILL = "skill"
    FINISH = "finish"


class VerificationStatus(str, Enum):
    SUCCESS = "success"
    RETRY = "retry"
    FAILURE = "failure"


class VerificationAction(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    REPLAN = "replan"
    FAIL = "fail"
    FINISH = "finish"


class EventType(str, Enum):
    TASK_RECEIVED = "task_received"
    PLAN_CREATED = "plan_created"
    STATE_TRANSITION = "state_transition"
    SKILL_STARTED = "skill_started"
    SKILL_CONTEXT_OPENED = "skill_context_opened"
    SKILL_PROGRESS = "skill_progress"
    SKILL_CONTEXT_CLOSED = "skill_context_closed"
    SKILL_RESULT = "skill_result"
    RUNTIME_VERIFIED = "runtime_verified"
    SKILL_VERIFIED = "skill_verified"
    FINISH_VERIFIED = "finish_verified"
    TASK_DONE = "task_done"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    STALE_RESULT_DROPPED = "stale_result_dropped"
    CONTEXT_BUILT = "context_built"
    MEMORY_WRITTEN = "memory_written"
    MEMORY_WRITE_FAILED = "memory_write_failed"
    PLAN_REVISED = "plan_revised"
