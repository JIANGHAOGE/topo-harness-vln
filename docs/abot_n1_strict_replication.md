# ABot-N1 Strict Replication Contract

This document separates the architecture reported by ABot-N1 from the engineering
substitutions used by CEE-AgentOS. The authoritative sources are the
[ABot-N1 technical report](https://arxiv.org/abs/2607.10383), the
[official project page](https://amap-cvlab.github.io/ABot-Navigation/ABot-N1/), and the
[official ABot-Navigation repository](https://github.com/amap-cvlab/ABot-Navigation).

## Reported ABot-N1 Architecture

ABot-N1 is a navigation model architecture, not the AgentOS runtime. It consists of
two asynchronously coupled systems joined by a dual language-and-vision interface.

### Slow System

At invocation `n`, the Slow System consumes exactly four input categories:

1. A short reference-memory buffer of historical front-view frames.
2. The synchronized reference tri-view observation: left, front, and right.
3. The task specification `g`.
4. The previous Slow decision: CoT and pixel goals.

Its output is:

```text
(CoT_n, pixel_goals_n)
```

The pixel goal has two independent semantic slots over the three camera views:

- `affordance_pixel`: the next visible, safe, traversable locomotion point, typically
  about 3 m ahead indoors or 5 m outdoors.
- `target_pixel`: the final semantic target, emitted only when visible or during final
  approach.

Both pixels may be present in the same decision. They are not mutually exclusive.
Topo nodes, remembered locations, viewpoint actions, completion decisions, bboxes,
controller actions, and verifier results are not Slow-System output types.

### Task-Specific CoT

Natural-language CoT is emitted only for the two semantically demanding tasks:

- Instruction-Following: `Finished Sub-Instruction` and `Ongoing Sub-Instruction`.
- Object-Goal: `Target Localization and Key Context` and `Path Planning`.

Point-Goal, POI-Goal, and Person-Following use pixel goals without a natural-language
CoT trace. The Slow output is therefore structured by task; it is not a generic free
form action plan.

### Fast System

The reported Fast System consumes:

1. Current visual memory and the current tri-view observation.
2. The latest Slow CoT and dual pixel goals.
3. The Slow reference tri-view on which those pixels were produced.
4. The task specification.

It predicts five continuous ego-frame SE(2) waypoints. Every waypoint contains 2-D
position, heading represented by sine and cosine, and a binary completion flag. Slow
and Fast run at different cadences; Fast continues tracking the cached Slow decision
while a newer Slow decision is being computed.

## Separation From ABot-N0 And AgentOS

Topo-Memory and coarse-to-fine Point-Goal/Object-Goal decomposition belong to the
ABot-N0 Agentic Navigation System. Main Agent, isolated Skill Runner, Verifier, context
management, memory, and skill evolvement belong to ABot-AgentOS. They may wrap N1, but
must not be inserted into the N1 Slow output schema.

The four input categories are represented by `NavigationSlowBrainInput` rather than being
implicitly concatenated into an unbounded chat transcript. `previous_decision` contains
the complete immediately preceding decision; the
reference memory contains at most four front-view frames with their observation IDs.
CEE-AgentOS also carries an optional, separately named ABot-N0 `topology_context` input
extension. It does not replace any of the four N1 categories and is not claimed as a
reported N1 input.

The combined CEE-AgentOS flow is therefore:

```text
AgentOS Main Agent + Topo-Memory
  -> typed navigation SkillRequest
  -> isolated navigation Skill
       mapped node: direct topology PointGoal
       local visual search: task CoT + affordance pixel + optional target pixel
       Fast substitute: A* plus bounded control rollout
  -> compressed SkillResult
  -> independent Skill Verifier
  -> Main Agent continuation/replan
  -> independent Finish Verifier
```

## CEE-AgentOS Engineering Substitutions

The public ABot-Navigation repository currently releases benchmarks and evaluator
interfaces, not the N1 Slow/Fast inference implementation or weights. CEE-AgentOS uses
the following explicit substitutions:

| ABot-N1 component | CEE-AgentOS implementation | Status |
| --- | --- | --- |
| 4B learned Slow VLM | Qwen OpenAI-compatible provider | Functional substitute |
| 2B learned Fast action expert | Node PointGoal or affordance-pixel depth projection + A* + Habitat/ROS2 step controller | Functional substitute |
| Five continuous waypoints per Fast tick | A* path followed through bounded embodiment steps | Not model-identical |
| Asynchronous Slow/Fast scheduling | Background Slow inference while Fast consumes the last committed decision | Functionally aligned, with an explicit latency fallback |
| Learned pixel coordinate tokens | Strict JSON using per-view normalized 0-1000 coordinates | Serialization substitute |
| Fast completion head | Controller result plus independent AgentOS Skill/Finish Verifiers | Runtime substitute |

The topology graph remains available to the Main Agent when constructing task
specification `g`. A bounded task-relevant semantic subgraph is also supplied to the
Slow System as an explicit N0 extension so the existing N1 CoT fields can express the
current-node hypothesis and connected-route constraint. It is not exposed as an
alternative Slow action and cannot independently justify a target pixel. Likewise,
historical keyframes remain a bounded front-view reference-memory buffer and the most
recent Slow decision is fed back explicitly; older multimodal history is not appended
without bound.

The active traditional-planner profile keeps the N1 dual-pixel interface inside a local
visual-search Skill. A mapped topology node bypasses pixel generation and becomes a metric
PointGoal. During local search, Qwen Slow emits the task-specific CoT plus coordinate-free
target and affordance queries. `VisualGroundingSkill` grounds those queries against one
explicitly bound current camera at a time; `SlowPixelGoalAssembler` validates the resulting
view and normalized bbox, then derives each bottom-center pixel. When the final
target confidence reaches the independent target safety threshold (0.90 by default,
overridable with `CEE_TARGET_CONFIDENCE_THRESHOLD`), A* executes `target_pixel`;
otherwise it executes `affordance_pixel` as the next intermediate goal. Merely emitting a
target alongside an executed affordance cannot end the local Skill.

The bbox fields are a user-approved, auditable grounding sidecar. They may be exposed in
the CEE response envelope and Web Relay, but they are not added to the N1 public output
tuple. The N1-compatible portion remains exactly `cot`, `affordance_pixel`, and
`target_pixel`. In the active policy, `target_pixel` is populated only for an explicit
final physical object named by task `g` and visibly grounded in the current tri-view.
Doorways, open floor, route transitions, and other semantic intermediate anchors populate
`affordance_pixel`. A natural-language task with no explicit final object therefore
advances through an affordance chain and leaves `target_pixel` empty.

## Adaptive Slow/Fast Scheduling

The first Slow decision is computed synchronously so Fast never starts without a valid
decision. After each bounded Fast chunk produces a fresh observation, the next Slow call
runs in the background while Fast continues consuming the last committed CoT, pixels, and
their reference observation. A candidate Slow result is committed only if its source
observation remains within 1.5 m and 90 degrees of the current pose.

This is an engineering reproduction of N1's asynchronous responsibility boundary, not a
claim that A* is the learned Fast Expert. Qwen latency is monitored with a 3.0 s deadline.
If the pending decision exceeds the deadline, it is cancelled and the Skill switches to
`sequential_fallback`: the cached affordance finishes as one bounded macro execution, Fast
stops, and Slow reasons again from a fresh synchronized observation. This implements the
requested escape hatch for observation lag instead of applying an arbitrarily stale pixel.

## Current Acceptance Criteria

1. Active public Slow output contains `cot`, `affordance_pixel`, and `target_pixel`.
2. A `route_to_region` topology decision bypasses Slow inference and executes the next
   connected node as a PointGoal.
3. Affordance pixels identify the next visible traversable waypoint; target pixels stay
   empty until the final POI or object is visually grounded.
4. A target at or above the configured confidence threshold preempts the affordance;
   lower-confidence targets retain their audit sidecar but fall back to the affordance.
5. Instruction and object tasks use the exact report-defined CoT blocks.
6. The Fast request preserves the Slow reference observation, task, CoT, and dual pixels.
7. Local controller completion never directly completes the global task.
8. Skill and Finish verification remain independent AgentOS components.
9. Only bounded front-view history and the immediately previous Slow decision recur.
10. Every Slow trace exposes the four official input categories and separately labels
   any bounded N0 topology extension used for that invocation.
11. The A* substitute uses a node position for regional travel, a high-confidence target
    pixel for final approach, and an affordance pixel for intermediate local travel.
12. Fast continues from the last committed Slow decision during an in-budget background
    Slow call; stale or over-deadline results are rejected or trigger sequential fallback.
13. Public bbox groundings remain an auditable sidecar and never change the N1 public
    `cot + dual pixel goal` schema.
