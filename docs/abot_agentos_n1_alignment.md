# ABot-AgentOS + ABot-N1 Alignment Contract

This document is the implementation contract for the AgentOS replica on the
`experiment/agentos-replica` branch. It is derived from the official
[ABot-AgentOS report](https://arxiv.org/abs/2607.10350) and
[ABot-N1 report](https://arxiv.org/abs/2607.10383). The downloaded report sources are
kept under `artifacts/abot-agentos-official/` and `artifacts/abot-n1-official/`.

The two reports define different layers of one system:

```text
ABot-AgentOS Agent Harness
  Main LLM -> isolated Skill Runner -> Verifier -> Main LLM
                         |
                         +-- Navigation Skill: ABot-N1 Slow/Fast contract
```

ABot-AgentOS owns task lifecycle, context boundaries, memory, delegation, and
verification. ABot-N1 owns the internal navigation policy interface. ABot-N0 topology
is an explicitly named planning-memory extension. The current engineering profile keeps
the task-specific N1 CoT and dual-pixel interface, but replaces the learned Fast Expert
with topology PointGoals or affordance-pixel depth projection, A*, and embodiment steps.

## Task Decomposition Boundary

Both layers decompose work, at different scopes:

- The AgentOS Main LLM forms a revisable high-level plan and delegates a sustained
  procedural subtask to a Skill Runner. This is the global task/Skill decomposition.
- The N1 Slow System advances one Navigation Skill over time. For
  Instruction-Following, its CoT records the finished and ongoing sub-instruction. The
  N1 data pipeline explicitly decomposes long instructions into short sub-instructions
  and aligns them with trajectory ranges; inference emits the current done/current
  state rather than an unbounded full plan.
- For Object-Goal, decomposition appears as target localization/key context plus path
  planning. Point-Goal, POI-Goal, and Person-Following do not emit linguistic CoT.

The stable language goal `g` is classified once per Navigation Skill as one of the five
official N1 task types: Point-Goal, Instruction-Following, Object-Goal, POI-Goal, or
Person-Following. Internal AgentOS plan steps must not overwrite or shorten `g`.

## AgentOS Alignment

| Official report requirement | CEE-AgentOS implementation | Status and acceptance check |
| --- | --- | --- |
| Main LLM performs scene-conditioned semantic planning with explicit completion conditions | `AgentOSHierarchicalPlanner` returns one typed high-level plan and Skill delegation | Aligned for the navigation profile; Main output contains no pixel, view, controller, or verification decision |
| Main LLM may use observation, map memory, robot state, recent history, and available Skills | Main planning receives the current observation, bounded context, fixed map-memory snapshot, compressed Skill results, and Verifier feedback | Aligned; N0 topology is marked as an optional map-memory extension |
| Procedural subtasks are delegated to a Skill Runner with isolated local context | `IsolatedSkillRunner` opens a private session; `N1NavigationSkillSession` exclusively owns local keyframes, previous N1 decision, and rollout state | Aligned; Main and Skill no longer share a mutable navigation session |
| Skill Runner handles repeated observations, actions, failed attempts, and recovery | `AgentOSHierarchicalNavigationSkill` performs bounded repeated Slow/Fast cycles before returning control | Aligned with a bounded engineering rollout |
| Main receives compressed progress/outcome instead of full local action history | `IsolatedSkillRunner` strips raw `outputs` from `main_agent_result` and returns `compressed_history` | Aligned and covered by runtime tests |
| Verifier supervises Runtime, Skill-level, and Finish-time stages | `Verifier` owns `ProgressRuntimeVerifier`, `LLMSkillVerifier`, and `LLMFinishVerifier` | Aligned with the report's explicit three stages |
| Runtime verification detects stagnation, blocked paths, timeouts, or ineffective progress | Runtime Verifier checks execution status, `stalled`, termination reason, steps, displacement, and remaining distance | Aligned for available controller telemetry; mechanical checks are deterministic |
| Skill-level verification checks semantic subgoal evidence independently of normal return | Qwen `LLMSkillVerifier` receives fresh post-observation plus bounded execution evidence | Aligned; local STOP or RPC success cannot prove semantic success |
| Finish-time verification alone authorizes global completion | Qwen `LLMFinishVerifier` checks the original instruction against fresh evidence and executed Skill summaries | Aligned; Skill STOP and target pixels cannot directly end the task |
| Context Management selects, compresses, and retrieves prompt information | `BoundedContextManager` bounds events, results, and graph-memory evidence | Structurally aligned; dense embedding retrieval remains an implementation substitution |
| Long-term Memory is external to the prompt and source-grounded | SQLite typed graph stores tasks, observations, evidence, Skill runs, verification traces, provenance, and source references | Structurally aligned; the current robot-navigation writer is narrower than the report's full person/event/dialogue adapters |
| Failure-driven evolution emits versioned, auditable, rollbackable JSON DSL assets after evaluation | Trace recorder plus `JsonEvoAssetStore` with target/regression gates, promotion, and rollback | Structurally aligned; automated offline diagnosis/proposal remains optional and cannot modify a live split |
| Edge model handles each turn and escalates complex cases to a cloud model through learned routing | Provider/router interfaces exist | Interface aligned; current routing policy is deterministic and is not claimed to reproduce the report's learned router |

## N1 Slow-System Alignment

Every invocation retains the four official N1 categories represented by
`NavigationSlowBrainInput`:

1. `reference_memory`: at most four historical front-view frames.
2. `reference_observation`: one synchronized left/front/right capture, represented to
   semantic consumers by a shared task-neutral audit produced one physical camera at a time.
3. `task_specification`: task type and goal `g`.
4. `previous_decision`: complete immediately preceding CoT and dual-pixel decision.

This project additionally supplies an optional `topology_context`, explicitly identified
as an ABot-N0 extension. It contains at most eight task-relevant semantic nodes and 16
connecting edges, including the current, target, next, and route nodes when known. It is
a semantic/connectivity prior, not visual evidence. The existing N1 CoT fields must
absorb the relevant node hypothesis and connected-route constraint; no extra free-form
CoT fields are introduced, and target pixels still require current visual evidence.

The active public Slow output contains:

```text
cot
affordance_pixel
target_pixel
```

CEE-AgentOS also exposes `affordance_grounding` and `target_grounding` bbox records produced
by an independent, camera-bound `VisualGroundingSkill`. The bottom center of each bbox deterministically produces its
pixel. This sidecar is a user-approved implementation extension for inspectability; it is
not a fourth N1 Slow output field. `target_pixel` is reserved for a final physical object
explicitly named by `g` and currently visible. Semantic intermediate anchors populate
`affordance_pixel`; tasks without an explicit final object use an affordance chain.

Task-specific response schemas enforce the report contract:

| Task | CoT | Execution goal |
| --- | --- | --- |
| Instruction-Following | finished sub-instruction + ongoing sub-instruction | connected topology PointGoal, then continuously refreshed affordance pixels |
| Object-Goal | Target Localization and Key Context + Path Planning | connected topology PointGoal, affordance pixels for local search, optional final target pixel |
| Point-Goal | `null` | direct metric/topology PointGoal; no pixel |
| POI-Goal | `null` | affordance pixel plus optional visible POI target pixel after regional PointGoal travel |
| Person-Following | `null` | affordance pixel plus visible person target pixel |

The N1 temporal state is scoped by the complete task specification `g`. If AgentOS
delegates a different task type or goal, prior CoT/pixels and the old reference-memory
buffer are not fed into the new task. Repeated cycles of the same delegated task retain
the immediately previous decision.

## N1 Fast-System Alignment

`NavigationFastSystemInput` records the four official Fast input streams:

1. current front-view memory and current tri-view observation;
2. latest Slow CoT and dual affordance/target pixels;
3. the Slow reference tri-view on which the pixels were emitted;
4. task specification `g`.

The official learned Fast Expert predicts five ego-frame SE(2) waypoints containing
`(x, y, sin(yaw), cos(yaw), completion)`. Those weights and inference code are not
public. This project uses the accepted functional substitute:

```text
mapped topology node -> metric PointGoal -> A* path
local visual search -> affordance pixel -> depth projection -> A* path
visible POI/object above confidence gate -> target pixel -> depth projection -> A* path
                                      -> Habitat/ROS2 bounded step controller
```

The two pixel roles are alternatives for the next Fast execution. A trustworthy visible
final target preempts the intermediate affordance. If the target is missing or below the
configured confidence threshold, Fast consumes the affordance. Reaching an affordance
does not count as reaching a simultaneously emitted target.

This substitute is deliberately recorded in every result as
`implementation=topology_point_goal_astar_habitat_steps` or
`affordance_pixel_depth_astar_habitat_steps`. It is not described as model-identical.
The bottom-center affordance grounding used for depth projection is also an execution
adaptation: it supplies a supporting-ground endpoint that a traditional planner can
consume, whereas the learned N1 Fast Expert fuses the dual pixels directly.

The Skill now reproduces the N1 asynchronous behavior at the engineering Fast cadence.
The first Slow decision is synchronous; later Slow calls run as background tasks while
8-step A* Fast chunks continue from the last committed decision. The Slow reference
observation remains attached to its pixel, and a result is committed only when source-to-
current pose drift is at most 1.5 m and 90 degrees. Task, observation, and Skill IDs remain
part of the trace contract.

The scheduler also implements the user-requested latency guard. A pending Qwen call has a
3.0 s wall-clock deadline. When it expires, the result is cancelled, the cached Fast goal
finishes, and the Skill returns to synchronous `Slow -> Fast` boundaries using a fresh
observation. This `sequential_fallback` is intentionally stricter than the N1 report and is
recorded as a deployment-safety adaptation for slow or unstable external model endpoints.

## Forbidden Couplings

The strict baseline rejects the following architectural shortcuts:

- Hermes acting as Main Agent, Skill Runner, and Verifier in one unbounded chat.
- Main Agent selecting camera views, pixels, A* targets, or controller actions.
- Navigation Skill authorizing global task completion.
- Verifier trusting normal tool return or local STOP as semantic evidence.
- Reusing previous N1 CoT after task specification `g` changes.
- Appending every historical image to the Main Agent context.
- Adding topology-specific fields, known-location actions, or finish decisions to N1
  Slow output. Relevant topology IDs may occur only inside the report-compatible CoT
  strings as reasoning context.
- Treating the optional N0 topology input extension as part of the N1 report contract.

## Verification Targets

1. Main Agent and Navigation Skill have separate session objects.
2. A changed task specification clears prior N1 decision and reference memory.
3. One Skill Run may execute multiple Slow/Fast cycles while Main receives one compact result.
4. Task-specific Slow schemas reject the wrong CoT shape.
5. Point-Goal rejects a target pixel; visible Person-Following pixels coincide.
6. No executable pixel produces a recoverable Skill failure instead of a fabricated goal.
7. Every Slow invocation writes an audit record containing the four official inputs,
   optional bounded topology extension, and exact N1 output.
8. Every Fast rollout records the official four-stream input contract and the engineering substitute name.
9. Runtime, Skill-level, and Finish-time verification remain separate trace events.
10. Full unit tests and one bounded Habitat task pass before this baseline is considered usable.

## 2026-08-21 Habitat Acceptance Run

Episode `hierarchical-20260821T103005901340` completed
`go to bedroom to find a bed` in 91.75 s and reached `DONE`. One isolated Navigation
Skill traversed connected topology nodes `backbone-004 -> backbone-003` before local
pixel execution; intermediate route nodes did not return procedural history to Main.
The five accepted Slow calls averaged 1.668 s (1.457-1.802 s). Their audits retain the
four N1 input categories, bounded topology extension, exact CoT/dual-pixel output, and
public bbox sidecar. The independent Finish Verifier authorized completion from the
latest synchronized observation. Inspection of the saved camera files places the bed in
the front view, extending into the edge of the right view. The model's evidence text
incorrectly called that image the left view, so view-label fidelity remains a measured
Verifier quality issue even though target presence and task completion were correct.

The first long Skill switched to `sequential_fallback` after a background Slow call
exceeded the 3.0 s deadline while Fast was still following the cached affordance. Later
short local Skills remained in asynchronous mode. This demonstrates both branches of the
adaptive policy. The remaining quality issue is conservative semantic verification: the
task required five Skill outcomes before the final evidence was accepted, although the
bed grounding appeared earlier.
