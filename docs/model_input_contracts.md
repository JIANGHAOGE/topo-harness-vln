# Model Input Contracts

This document describes the model-facing contracts used by the current
`experiment/agentos-replica` runtime. It records what each module actually sees,
not what the architecture diagram suggests it sees.

The main debugging rule is:

> The same Qwen model can produce different conclusions because Main Agent,
> Slow Brain, Grounding, and Verifier receive different representations of the
> scene. A downstream model cannot recover information that an upstream audit
> omitted.

## 1. Active Runtime Chain

```text
Hermes
  -> submit global instruction
AgentOS Runtime
  -> Current-view audit: 3 independent Qwen image calls
  -> Main Agent: 1 Qwen structured planning call
     -> optional task decomposition and semantic node ranking Qwen calls
  -> MappingSkill or NavigationSkill
     -> PointGoal: deterministic topology/A*/controller path
     -> ObjectGoal:
        -> Current-view audit, normally cacheable per observation
        -> Slow Brain: 1 Qwen semantic reasoning call
        -> View Binding: 3 independent Qwen image calls
        -> Visual Grounding: 1 call per query on its selected current frame
        -> bottom-center pixel, depth projection, A*, controller
  -> Runtime Verifier: deterministic
  -> Skill Verifier: current-view audit + 1 Qwen decision call
  -> Main Agent receives compressed handoff and advances/replans
  -> Finish Verifier: current-view audit + 1 Qwen decision call
```

The three single-view audit calls are cached by model identity, observation ID,
and image references. Main, Slow, Skill Verifier, and Finish Verifier can reuse
the same audit only when they refer to the same synchronized observation.

## 2. Input Authority

| Information | Intended authority | Important limitation |
|---|---|---|
| Global task | `TaskRequest.instruction` / `stable_global_task` | Summaries and subtasks are derived views, not replacements. |
| Current image | `Observation.rgb_refs` bound to one `observation_id` | Most reasoning modules see single-view audit text rather than the raw current images. |
| Camera identity | `SlowBrainTriViewAudit.left/front/right` | Each entry comes from one physical camera call; image-local direction is not camera identity. |
| Topology prior | Frozen semantic map plus fixed graph edges | Node objects and landmarks are priors, not current visual detections. |
| Current-task search evidence | `goal_search` records | `no_target_observed` is local negative evidence; `blocked` is route failure. |
| Motion result | Embodiment telemetry and execution facts | A reached PointGoal proves arrival, not target visibility. |
| Target localization | `VisualGroundingSkill` on a named current frame | A valid bbox proves only current-frame localization confidence. |
| Skill completion | Skill Verifier | It judges one delegated subgoal, not the global task. |
| Global completion | Finish Verifier | OmniNav STOP or Skill success alone cannot finish the task. |

## 3. Current-View Audit

**Purpose:** extract task-neutral visual facts while preserving physical camera
identity.

**Model calls:** three Qwen calls in parallel, one each for LEFT, FRONT, RIGHT.

**Each call receives:**

```json
{
  "observation_id": "hierarchical-obs-N",
  "camera": "left | front | right",
  "image": "exactly one current RGB frame"
}
```

**Output Schema:**

```json
{
  "view": "left | front | right",
  "visible_objects": ["..."],
  "navigable_semantic_anchors": ["..."],
  "target_relevant_evidence": ["..."],
  "scene_summary": "..."
}
```

**Consumed by:** Main Agent, Slow Brain, Skill Verifier, Finish Verifier.

**Failure signature:** if a bed is visible in a raw frame but absent from the
corresponding single-view audit, every consumer using that audit can miss it.
Agreement among downstream modules then does not provide independent visual
confirmation because they share the same upstream description.

Implementation: `agentos/providers/vision.py`.

## 4. Main Agent

**Purpose:** plan the global task, select a top-level Skill, and delegate one
subtask. It does not select pixels or controller actions.

**Current visual input:** the structured tri-view audit. When all three audits
succeed, the Main Agent does not receive the three raw current images.

**Structured context:**

```json
{
  "stable_global_task": "original user instruction",
  "skill_catalog": ["MappingSkill", "NavigationSkill"],
  "topo_memory": {
    "block_layer": "place type, scene summary, visited, goal_search",
    "road_layer": "fixed connectivity",
    "function_layer": "function and task affordances",
    "object_poi_layer": "objects and landmarks"
  },
  "semantic_route_plan": "previous bounded route state",
  "active_skill_plan": "accepted remaining subtasks",
  "active_delegated_subtask_index": 0,
  "region_search_state": "stable initial candidate queue",
  "region_candidate_progress": "visited and current-task goal_search by candidate",
  "progress_ledger": "last 8 verified Skill outcomes",
  "mapping_evidence": "availability and source observation",
  "previous_skill_results": "last 3 compressed results",
  "verifier_feedback": "latest verification summary",
  "long_term_memory": "up to 6 persistent Entity/Place facts"
}
```

**Output Schema:** `AgentOSSkillDelegation`.

```json
{
  "skill_name": "MappingSkill | NavigationSkill",
  "task_specification": {
    "task_type": "point_goal | object_goal | instruction_following",
    "goal": "currently delegated instruction"
  },
  "planner_output": {
    "global_instruction": "...",
    "destination_query": "place",
    "object_goal": "physical target",
    "node_selection": null,
    "region_search": {
      "candidates": [
        {
          "topology_node_id": "backbone-NNN",
          "confidence": 0.0,
          "place_evidence": "...",
          "object_evidence": "...",
          "relation_reason": "..."
        }
      ],
      "active_topology_node_id": "backbone-NNN",
      "update_reason": "..."
    },
    "subtasks": [
      {
        "skill_name": "NavigationSkill",
        "task_type": "point_goal | object_goal | instruction_following",
        "instruction": "...",
        "target": "...",
        "topology_node_id": null,
        "success_condition": "..."
      }
    ]
  },
  "delegated_subtask_index": 0,
  "rationale_summary": "..."
}
```

**Potential confusion:** current topology appears through `topo_memory`,
`semantic_route_plan`, `region_search_state`, and `region_candidate_progress`.
Execution progress appears through `active_skill_plan`, `progress_ledger`,
`previous_skill_results`, and `verifier_feedback`. These representations have
different authority and can conflict if their observation/revision provenance is
not inspected.

Implementation: `agentos/planning/agentos_hierarchical.py`.

## 5. Semantic Task and Node Ranker

This is an independent model audit around Main Agent planning. It does not see
images.

### Task decomposition

**Input:** only the original instruction.

**Output:**

```json
{
  "global_goal": "go to office to find a white refrigerator",
  "destination_query": "office",
  "object_goal": "white refrigerator",
  "node_local_radius_m": 5.0,
  "anchor_max_depth_m": 5.0
}
```

### Joint region ranking

**Input:** decomposition plus all node place/function/object/landmark fields and
current `visited/goal_search` evidence.

**Output:** one `RegionSearchPlan` with one to four ordered candidates. The same
Qwen call jointly evaluates destination-place compatibility, target-object
evidence, their relation, and current task search evidence. An explicit place is
the primary relation constraint; object evidence breaks ties among compatible
places. There is no second destination-only Qwen call or post-ranking merge.

`confidence` values are Qwen self-reported joint relevance estimates, not a
fixed weighted formula. Runtime validates node IDs and requires the active node
to be the first candidate; it does not reinterpret the semantics or replace the
model's winner.

**Current risk:** Main Agent and the Ranker are two semantic authorities. The
Ranker can reject a valid Main result and trigger up to three Main retries. Qwen
confidence values are not calibrated probabilities.

Implementation: `agentos/planning/semantic_nodes.py`.

## 6. MappingSkill and Context Manager

Neither module calls Qwen.

### MappingSkill output

```json
{
  "map_id": "...",
  "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0, "frame": "...", "source": "..."},
  "current_topology_node_id": "backbone-NNN",
  "nearby_nodes": ["up to 4 node records"],
  "connected_edges": ["up to 12 edges"],
  "topo_memory": "four-layer topology"
}
```

### Skill context

The deterministic Context Manager renders task text, plan summary, textual
observation/modalities, pose, four prior Skill summaries, six recent events, and
up to six retrieved memory facts. Memory retrieval currently uses token overlap,
not an embedding or Qwen semantic ranker.

This rendered string enters `SkillRequest.context_summary` and is included again
inside Slow Brain input, so it can duplicate the global task, plan, history, and
memory already represented by typed fields.

Implementations: `agentos/skills/navigation/agentos_hierarchical.py`,
`agentos/context/manager.py`, and `agentos/memory/store.py`.

## 7. N1 Slow Brain

**Purpose:** decide the next semantic affordance and target query. It does not
produce bboxes, pixels, depth, coordinates, controller actions, or completion.

**Typed input:** `NavigationSlowBrainInput`.

```json
{
  "reference_memory": [
    {
      "slot": 0,
      "front_image_ref": "historical keyframe",
      "observation_id": "historical observation"
    }
  ],
  "reference_observation": {
    "observation_id": "current observation",
    "left_image_ref": "path",
    "front_image_ref": "path",
    "right_image_ref": "path"
  },
  "current_view_audit": {
    "left": "single-view facts",
    "front": "single-view facts",
    "right": "single-view facts"
  },
  "task_specification": {
    "task_type": "object_goal",
    "goal": "delegated subtask"
  },
  "previous_decision": "previous CoT plus previous pixel slots",
  "topology_context": {
    "current_topology_node_id": "...",
    "target_topology_node_id": "...",
    "next_topology_node_id": "...",
    "route_node_ids": ["..."],
    "nodes": "up to 8 bounded semantic nodes",
    "edges": "up to 16 connected edges",
    "target_region_reached": false,
    "guidance_reason": "..."
  }
}
```

**Actual image attachment:** up to four historical FRONT keyframes are attached.
The three current image paths occur in typed JSON, but the current raw images are
not attached to the Slow reasoning call. Slow relies on `current_view_audit` for
current visual facts.

**Additional text input:** stable global task, delegated command and success
condition, plus the rendered `SkillRequest.context_summary`.

**Output Schema for ObjectGoal:**

```json
{
  "cot": {
    "target_localization_and_key_context": "max 512 chars",
    "path_planning": "max 512 chars"
  },
  "affordance_query": {
    "role": "affordance",
    "label": "visible semantic waypoint",
    "preferred_view": "left | front | right",
    "visual_description": "current local appearance",
    "spatial_relation": "why it advances the task",
    "excluded_labels": ["distractors"]
  },
  "target_query": {
    "role": "target",
    "label": "final physical object",
    "preferred_view": "left | front | right | null",
    "visual_description": "...",
    "spatial_relation": "...",
    "excluded_labels": ["..."]
  }
}
```

**Potential confusion:** Slow receives global task, delegated task, plan summary,
previous Skill summaries, previous Slow decision, historical images, current
audit, topology, and retrieved memory in one prompt. Several fields describe the
same intent at different times. `previous_decision` is typed, but its old pixel
slots and wording remain present and can anchor the new answer if current audits
are weak.

Implementation: `agentos/skills/navigation/agentos_hierarchical.py` and
`agentos/schemas/task_reasoning.py`.

## 8. Slow View Binding

**Purpose:** verify which current physical camera supports the exact Slow-selected
semantic query. It cannot choose another route or rename the query.

**Model calls:** three independent calls, one per current camera.

**Each call receives:**

```json
{
  "observation_id": "current observation",
  "image": "one current camera frame",
  "queries": [
    {
      "role": "affordance | target",
      "label": "unchanged Slow label",
      "proposed_view": "Slow hypothesis",
      "visual_description_hypothesis": "Slow description",
      "excluded_labels": ["..."]
    }
  ]
}
```

`spatial_relation`, topology, global task, route reasoning, history, and old
coordinates do not enter this call.

**Output:** supported flag, verified camera, confidence, local visual description,
and evidence summary for each unchanged query.

The highest-confidence supported camera wins. An unsupported affordance rejects
the Slow decision. An unsupported target has its `preferred_view` cleared.

## 9. VisualGroundingSkill

**Purpose:** localize an unchanged semantic query in a current image. It does not
decide which waypoint advances the task.

**Per-query input:**

```json
{
  "observation_id": "current observation",
  "role": "affordance | target",
  "query": "Slow label",
  "candidate_views": ["selected physical camera"],
  "constraints": {
    "preferred_view": "left | front | right",
    "visual_description": "View Binding current-image description",
    "excluded_labels": ["..."]
  },
  "image": "one selected current RGB frame",
  "min_confidence": "0.80 affordance / 0.90 target"
}
```

`spatial_relation`, CoT, topology, memory, route plan, and historical images do
not enter Grounding.

**Output Schema:**

```json
{
  "label": "unchanged query",
  "view": "left | front | right",
  "bbox_normalized_1000": ["xmin", "ymin", "xmax", "ymax"],
  "confidence": 0.0,
  "evidence_summary": "current-image localization evidence"
}
```

The public Pixel Goal is the bbox bottom-center: `[(xmin+xmax)/2, ymax]`.

**Current gate:** a target query reaches Grounding only when Slow/View Binding has
a non-null `preferred_view`. Therefore a target omitted by the single-view audit
cannot be recovered by a broad Grounding search, even if another current frame
contains it. This is a major recall bottleneck and should be treated separately
from bbox precision.

Implementations: `agentos/providers/grounding.py` and
`agentos/skills/perception/visual_grounding.py`.

## 10. Fast System

The current point/pixel Fast path does not call Qwen.

**Input:** current observation ID, selected target or affordance pixel, depth,
camera calibration, semantic map hints, and the delegated task specification.

**Process:** normalized pixel to image pixel, depth projection, maximum 5 m anchor
depth handling, navigability projection, A* path generation, then embodiment
steps/controller execution.

**Output:** deterministic telemetry including reached, progress, remaining
distance, collisions, stalled state, termination reason, and a fresh observation.

Errors here appear as a correct bbox with an incorrect projected point/path, or a
correct point with failed motion. They should not be diagnosed as Slow reasoning
errors.

## 11. Verifiers

### Runtime Verifier

No Qwen call. It consumes Skill status and deterministic execution telemetry. It
checks stale results, blocked/stalled/timeout states, and measurable progress.

### Skill Verifier

**Text input:** delegated command, success conditions, Skill status, compressed
history, deterministic execution facts, semantic completion, and up to six
bounded evidence records.

**Current visual input:** the latest synchronized tri-view audit text. When an
ObjectGoal target was executed, it also receives the Target grounding source
artifact. It receives up to three recent trajectory keyframes from the current
Skill run. Total additional evidence images are capped at four.

**Output:**

```json
{
  "outcome": "achieved | not_achieved",
  "summary": "...",
  "reasons": ["..."],
  "observed_evidence": ["..."]
}
```

For ObjectGoal, Runtime post-processing still requires structured
`semantic_completion.satisfied=true`, meaning a high-confidence Target was
grounded and its target-role execution reached the goal.

### Finish Verifier

**Input:** original task, explicit success conditions, current plan revision,
plan intent, executed Skill summaries, and latest tri-view audit text.

**Output:** `finish`, `replan`, or `fail` with observed evidence.

**Important limitation:** the active Finish Verifier receives audit text rather
than the raw current images when the audit succeeds. Skill and Finish Verifier
can therefore agree with an incorrect audit. This agreement is not equivalent to
two independent visual detections.

Implementation: `agentos/verification/llm.py`.

## 12. Handoff and Memory

### NavigationSkill to Main Agent handoff

Only this bounded object crosses the isolated Skill Runner boundary:

```json
{
  "observation": "latest observation metadata and refs",
  "slow_context": "latest bounded topology context",
  "semantic_route_plan": "latest route state",
  "execution_summary": "controller/reached/distance/stall/termination",
  "active_target_evidence": "sticky target evidence",
  "topology_goal_search": "latest current-task node search record",
  "semantic_completion": "ObjectGoal target protocol state",
  "selected_keyframes": "up to 4 front-frame references"
}
```

Raw local progress history does not enter Main Agent context.

### Memory Writer

Memory writing is deterministic, not a Qwen Writer. It stores Task, Observation,
SkillRun, Evidence, and Verifier records. A visual target or affordance becomes a
persistent Entity linked to a Place only when Grounding confidence is at least
0.90 and a valid bbox/current topology node are present.

Main Agent retrieves up to six persistent Entity/Place facts. Skill context also
retrieves memory. Retrieval is lexical token overlap, so synonyms and verbose
metadata can affect ranking.

## 13. Error-to-Module Lookup

| Symptom | Inspect first | Do not blame first |
|---|---|---|
| Wrong LEFT/FRONT/RIGHT description | Single-view audit artifact and observation ID | A* or controller |
| Correct object name, wrong bbox | Grounding request image/view/normalization | Main Agent plan |
| Object visible but no Target query | Current-view audit, Slow target query, View Binding | Pixel projection |
| Correct bbox, wrong world point | depth timestamp, camera calibration, pixel-to-point | Slow CoT |
| Correct world point, wrong route | navigability projection and A* graph | Grounding model |
| Wrong node selected | Main context, semantic map fields, Ranker merge | Visual Grounding |
| Repeated region exploration | `region_search_state`, `goal_search`, handoff, verifier feedback | Raw images alone |
| Skill ends after affordance | `semantic_completion` and Skill Verifier result | Finish Verifier only |
| Global task ends early | Finish Verifier input and accepted plan progress | Local STOP alone |
| Old conclusion copied | `previous_decision`, historical keyframes, current audit strength | Timestamp unless IDs mismatch |

## 14. Current Input-Design Risks

1. **Shared audit bottleneck.** Main, Slow, and both Verifiers rely on the same
   current-view audit representation. A missed object can propagate consistently.
2. **Duplicate planning state.** Main receives several overlapping plan, route,
   candidate, progress, and verifier summaries with different revision scopes.
3. **Duplicate Slow context.** Slow receives typed task/topology/history plus a
   rendered context string containing overlapping task, plan, memory, and results.
4. **Two semantic planning authorities.** Main Agent proposes a region and the
   independent Ranker can override/reject it, which can still produce retries
   when their evidence or task interpretation differs.
5. **Target recall gate.** Grounding searches for Target only after Slow/View
   Binding assigns a current view. This makes upstream audit recall decisive.
6. **Verifier dependence is not independence.** Skill/Finish Verifier are
   separate decision calls, but their current visual facts often share the same
   cached audit.
7. **Lexical memory retrieval.** Long-term memory is structured but retrieval is
   still token-overlap based, so semantic paraphrases can be missed.

These risks explain why many recent failures looked like weak model reasoning but
were actually caused by missing, duplicated, stale, or differently authoritative
inputs.

## 15. Recommended Diagnostic Record

Every model invocation should eventually persist one compact audit record:

```json
{
  "call_id": "...",
  "module": "main_agent | view_audit | semantic_ranker | slow_brain | view_binding | grounding | skill_verifier | finish_verifier",
  "task_id": "...",
  "skill_run_id": "...",
  "observation_id": "...",
  "plan_revision": 0,
  "input_schema": "...",
  "text_field_names": ["..."],
  "image_bindings": [{"view": "front", "ref": "..."}],
  "output_schema": "...",
  "parsed_output": {},
  "latency_s": 0.0,
  "retry_count": 0
}
```

The record should store structured payloads and image references, not duplicate
image bytes or hidden chain-of-thought. This would make input provenance visible
without expanding the Agent context.
