# ABot-N0/N1 + AgentOS Runtime Architecture

## Scope

CEE-AgentOS maps the non-training functional roles and observable runtime contracts of ABot-N0 and ABot-N1.
Learned weights, data construction, imitation/RL training, and the original learned action expert are intentionally
excluded. External Qwen is the Slow System. A* plus the embodiment's step controller is the training-free Fast
System. AgentOS provides protocol enforcement, execution safety, verification, tracing, and embodiment portability.
This is functional emulation rather than model/runtime equivalence: the current Fast System is not N1's learned
visual action expert, and Slow/Fast execution is low-frequency sequential rather than truly asynchronous.

## Component Mapping

| Reference role | CEE-AgentOS implementation | Responsibility |
|---|---|---|
| ABot-N0 Agentic Planner | `AgenticPlannerOutput`, `AgenticPlan`, Hermes/Qwen | Read current observation and both memories, then produce typed coarse-to-fine subtasks |
| ABot-N0 Actor | `advance_navigation` and `SelectedNavigationAction` | Select one macro Skill for the active subtask |
| ABot-N0 Episodic Visual Memory | task-scoped trajectory keyframes in `AgentNavigationMemory` | Retain bounded visual evidence from macro executions |
| ABot-N0 Topo-Memory | `TopoMemoryView` | Supply Block, Road, Function, and Object/POI knowledge |
| ABot-N0 Self-Reflector | `SubtaskReflection` | Judge subtask completion and request replan at macro boundaries |
| ABot-N1 Slow System | external Qwen, `VisualGroundingSkill`, and `SlowPixelGoalAssembler` | Produce task-specific CoT and coordinate-free target/affordance queries; ground each query in one bound camera frame and assemble the audited bbox bottom center into a pixel goal |
| ABot-N1 Fast System | `PointNavigationSkill`, A*, Habitat greedy follower | Convert the goal into a path and execute control steps |
| AgentOS Harness | schemas, service runtime, event bus, memory, verifiers, adapters | Enforce IDs, timeouts, cancellation, stale-result rejection, and global completion |

## N0 Agentic Planner

The Planner receives the original instruction, four-layer Topo-Memory, short-term Episodic Visual Memory, and the
current observation. Its output is an ordered `AgenticPlannerOutput`, not free-form prose. The protocol deliberately
uses two runtime boundaries: `begin_navigation_task(global_goal)` first returns `ABotSlowContext`; the first
`advance_navigation` then carries both `planner_output` and the first Actor decision. This prevents the destination
node from being selected before the Planner has seen Topo-Memory while avoiding a separate extra planning turn.

The current navigation decomposition uses ABot-N0's typed roles:

1. **Approaching / Point-Goal**: reach the connected topology region associated with the destination.
2. **Reaching / Object-Goal or POI-Goal**: locate the requested object or POI in that regional context.
3. **Interaction / Instruction-Following or Person-Following**: used when the task requires a task-specific final
   behavior rather than simple object reach.

Each subtask has a stable ID, task type, stage, target, optional topology node, success condition, and status. The
active subtask type controls which N1 CoT schema is legal. A mismatched CoT type is rejected before movement.
Approaching accepts only Point-Goal, Reaching accepts Object-Goal or POI-Goal, and Interaction accepts
Instruction-Following or Person-Following. Role/type mismatches fail validation, but the roles may interleave or
repeat in the order required by the instruction. For example, `Interaction(exit/turn) -> Approaching(region) ->
Reaching(robot)` is valid. When the Self-Reflector sets `should_replan=true`, the same Slow invocation must supply a
replacement `planner_output`.

## N0 Four-Layer Topo-Memory

`AgentNavigationMemory.topo_memory_view()` projects the fixed, portable topology and semantic JSON into four layers:

1. **Block layer**: room-like or regional nodes, place type, summary, position, and task visit state.
2. **Road layer**: traversable connectivity, route distance, junction/open-space transition character.
3. **Function layer**: place function and task affordances such as food preparation or waiting.
4. **Object/POI layer**: objects and landmarks attached to stable map nodes.

The geometric topology is generated from the environment-neutral `NavigabilityGrid`. Habitat Navmesh and ROS2
OccupancyGrid therefore share the same downstream memory contract. Topology IDs are stable `backbone-XXX` IDs.
Task-scoped visual evidence uses `location-XXX` IDs and is never translated into a backbone ID.

The current project policy keeps the prebuilt backbone and semantic map frozen for a service lifetime. Runtime
observations update task visit/evidence state but do not create dense topology nodes. This preserves portability and
prevents simulator truth from leaking into the Agent protocol.

## N1 Slow-System Input

At invocation `n`, `ABotSlowContext` contains only:

1. Up to four front-view keyframes spanning the immediately previous macro execution.
2. A task-neutral audit of the current synchronized left/front/right observation. Each
   audit is produced from exactly one physical camera image and cached by observation
   identity; the Slow model never infers camera ownership from three images in one call.
3. The original task specification.
4. The previous Slow decision `(C_(n-1), p_(n-1))`.
5. A typed `ABotPlannerContext` containing N0 Topo-Memory, the active Agentic Plan, and the active typed subtask.

Older images stay in server-side memory and enter the model only through an explicit bounded recall. Hermes image
pruning retains the latest visual packet and removes superseded tool images, so prompt growth is bounded by the
current decision window rather than task duration.

## N1 Task-Specific CoT

The Slow System does not emit one universal reasoning template:

- **Instruction-Following**: `finished_sub_instruction`, `ongoing_sub_instruction`.
- **Object-Goal**: `target_localization_and_key_context`, `path_planning`.
- **Point-Goal, POI-Goal, Person-Following**: no linguistic CoT (`cot=null`).

The fields are strict Pydantic and Hermes JSON schemas. The model may not add a generic decision trace or combine all
four text fields. Compatibility fields such as route hypothesis and candidate scoring are synthesized internally and
are no longer exposed in the Hermes tool schema. CoT remains short and is stored once as part of `ABotSlowOutput`.

## N1 Dual Pixel Goal

`DualPixelGoal` uses the N1 tri-view form: each of `left`, `front`, and `right` is either `[]` or `[x, y]`, and only
one view may be populated for each pixel type.

- **Target pixel**: bottom center of the independent Grounding Skill's detection bounding box. It is populated only when the active visual
  anchor is the directly visible task target.
- **Affordance pixel**: bottom center of the Grounding bbox for the next visible, safe, semantically named traversable anchor selected by
  the current CoT. The assembler validates its left/front/right ownership and normalized 0-1000 coordinates before
  converting it to the native RGB resolution.

The grounding pixel can still be backprojected through synchronized depth to define the requested spatial target.
It is not mislabeled as the affordance pixel. Pixel depth is capped at 5 m and may be further capped by the active
connected route segment. Once A* supplies the traversable lookahead, `PointNavigationSkill` emits a
`pixel_guidance_ready` event. The Runtime persists the complete `ABotSlowOutput` during that event, before the first
controller step. `PointNavigationReport` and `ABotFastExecution` carry the same `slow_step_id`, binding the cached
decision to its path and control result. A regression test inspects Memory from the controller callback and enforces
this ordering.

## Training-Free Fast System

The Fast System receives the Slow goal and runs without Qwen calls:

1. Snap the spatial goal to the navigability map.
2. Generate the shortest A* or Habitat Navmesh path.
3. Cache the Slow affordance pixel and derive the A* path to its depth-projected point.
4. Execute one embodiment step per control cycle using that cached decision.
5. Record pose-aware waypoints and at most four macro trajectory keyframes.
6. Stop on arrival, sustained stall, invalid path, cancellation, timeout, or the configured control budget.

This substitutes a deterministic planner/controller for N1's learned action expert while preserving the same
Slow-goal-to-Fast-trajectory responsibility and low-frequency Slow invocation policy.

## Reflection And Verification

At the next macro boundary, `SubtaskReflection` evaluates the previous active subtask using its execution outcome,
Episodic Visual Memory, current observation, and Topo-Memory. Completion activates the next typed subtask. Failure can
request replanning. Reflection and the next Actor decision may share one external-Qwen invocation while remaining
separate structured outputs.

AgentOS verification remains distinct:

- Runtime verification rejects stale, timed-out, invalid, or stalled execution.
- Skill/Self-Reflector verification evaluates one macro subtask.
- The independent Finish Verifier is the sole authority for global `done` or terminal `failed`.

A local anchor reach, A* arrival, controller STOP, or successful Skill result cannot finish the global task. A rejected
finish leaves task memory intact and returns `replan` or `fail` guidance.

## Invocation Schedule

Slow is invoked before the first macro Skill and after a macro Skill finishes. Additional wakeups are restricted to
node arrival, sustained stall, path invalidation, cancellation/timeout, or high-confidence target discovery. A* path
tracking and individual controller steps never invoke Slow.

The normal loop is:

```text
Task -> initial observation + Episodic Memory + four-layer Topo-Memory
     -> N0 Planner -> active typed subtask
     -> mapped regional node -> direct PointGoal, or N1-style Slow(Context, previous C/p)
     -> Slow semantic target/affordance query -> camera-bound VisualGrounding
     -> visible affordance pixel -> depth projection + A* path (cached before motion)
        + optional final target pixel retained for verification
     -> Fast(cached A* path + controller) -> trajectory keyframes + execution result
     -> N0 Self-Reflector -> next subtask / replan
     -> AgentOS Finish Verifier -> DONE / continue
```

## Portability Boundary

The Slow System, Agentic Plan, memory schemas, PointGoal/DualPixel protocol, reflection, and verification are embodiment
independent. Habitat and ROS2 adapters only implement observation, pixel/depth projection, map path planning, and
step execution. Replacing Habitat with a robot dog therefore does not change Planner, Slow Context, CoT, Topo-Memory,
or verifier contracts.

## References

- ABot-N0 paper and project architecture: <https://arxiv.org/html/2602.11598>
- ABot-N1 slow-fast system and task-specific CoT/pixel supervision: <https://arxiv.org/html/2607.10383>
