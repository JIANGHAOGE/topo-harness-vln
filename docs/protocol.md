# Protocol

## Trace Invariant

Every async request and result carries a `TraceContext`:

- `episode_id`
- `task_id`
- `plan_revision`
- `skill_run_id`
- `observation_id`
- `frame_timestamp`
- `pose_timestamp`
- `deadline`

The runtime accepts a skill or verifier result only when the async source fields
match the currently active trace and the deadline has not expired. Late results
are published as `stale_result_dropped` events and do not mutate runtime state.

## Core Messages

### TaskRequest

Global user task. It contains natural-language instruction text and metadata,
but no low-level action.

### Observation

Compressed environment snapshot. It may reference RGB/depth/topdown media paths,
pose, detected objects, and text notes. Raw frame history belongs inside skills
or embodiment adapters, not in the Main Agent context.

### SkillRequest

Bounded execution request:

- `skill_name`
- `command`
- `success_conditions`
- `context_summary`
- `inputs`
- `timeout_s`
- `retry_policy`
- full `TraceContext`

For `NavigationSkill`, `command` must use
`Navigate towards <currently visible anchor>`. AgentOS also records the anchor
and its `left`, `front`, or `right` source view. After each bounded OmniNav run,
the Agent grounds another anchor from the new observation; unseen rooms and
objects are not valid immediate commands.

### SkillResult

Compressed execution outcome:

- `status`
- `summary`
- `evidence`
- `outputs`
- `stop_reason`
- `compressed_history`

For OmniNav, STOP or `arrive_pred` may set `stop_reason` and evidence such as
`omninav_stop=true`. That ends the skill run only.

### VerificationResult

One Verifier emits a decision tagged with one of three stages:

- `runtime`
- `skill`
- `finish`

Only a finish-level success can end the global task as `DONE`.

## Main-Agent Context Rule

The Main Agent receives `SkillResult.summary`, compressed history, selected
evidence references, and Context Management retrievals. `SkillResult.outputs`
is stripped at the Skill Runner boundary. Complete per-frame execution history,
raw image buffers, controller tick logs, and unselected historical frames remain
in audit/memory storage and never enter Main Agent prompts.
