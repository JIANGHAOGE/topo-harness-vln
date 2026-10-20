# State Machine

## States

```text
IDLE
  -> PLAN
  -> START_SKILL
  -> EXECUTE_SKILL
  -> VERIFY_RUNTIME
      SUCCESS -> VERIFY_SKILL
      RETRY   -> EXECUTE_SKILL
      FAILURE -> PLAN or FAILED
  -> VERIFY_SKILL
      SUCCESS -> NEXT_STEP / VERIFY_FINISH
      RETRY   -> EXECUTE_SKILL
      FAILURE -> PLAN or FAILED
  -> VERIFY_FINISH
      SUCCESS -> DONE
      FAILURE -> PLAN or FAILED
```

`CANCELLED` can be reached from active states. `FAILED` and `DONE` are terminal
for one task run.

## Runtime Behavior

- Planning is represented by `StaticPlanner` for deterministic tests and a
  `Replanner` extension point for production scene-conditioned planning.
- Skill execution goes through `SkillRegistry` and `SkillContext`.
- Each skill receives a bounded `EmbodiedContext`; raw frame history stays outside prompts.
- Skill timeout is enforced with `asyncio.wait_for`.
- Cancellation is carried by `CancellationToken`.
- Every transition publishes an event.
- Stale results are dropped before verification.
- The single Verifier applies runtime, skill-level, and finish-time checks and
  recommends continue, retry, replan, fail, or finish at the relevant stage.
- Source-grounded task, observation, result, and verification summaries can be
  persisted through `MemoryWriter` without making memory failure fatal to control.

## STOP Handling

OmniNav STOP is not a global terminal transition. It maps to:

```text
EXECUTE_SKILL -> SkillResult(status=success, stop_reason=...)
VERIFY_SKILL  -> success or retry/failure
VERIFY_FINISH -> done only if finish verifier passes
```

This prevents a local navigation model from ending the global task without
independent task-level evidence.
