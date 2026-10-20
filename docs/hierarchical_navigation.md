# Hierarchical Navigation Version

This is an independent third profile alongside `omni` and `minimal`. It shares the Habitat adapter but has a
separate service port, Hermes toolset, runtime home, logs, and launch scripts. It does not register or call OmniNav.

## Skill Composition

```text
Task Agent selects a currently visible semantic anchor
  -> AnchorGenerationSkill
       -> AnchorGroundingSkill (one Qwen vision call -> RGB pixel)
       -> PixelToPointSkill (matching Habitat depth -> world point)
  -> PointNavigationSkill
       -> PathPlanningSkill (navmesh snap + shortest path)
       -> PathExecutionSkill (one Habitat action at a time, no per-frame LLM)
  -> fresh three-view observation returned to the Task Agent

Task Agent selects a recorded location for backtracking
  -> saved world point
  -> the same PointNavigationSkill
```

Each stage returns a standard `SkillResult` with the source `observation_id`, stage `skill_run_id`, and timestamps.
Pixels are top-left-origin coordinates in the selected `640x480` RGB view. The camera profile is three views at
`HFOV 120 degrees / VFOV 90 degrees`; each RGB camera has a matching depth camera.

The Qwen-selected pixel is drawn onto a durable annotated RGB image. Habitat's PathFinder uses its loaded navmesh to
run the shortest-path query, then its greedy geodesic follower emits one native `turn_left`, `turn_right`, or
`move_forward` action per control cycle. During execution, Habitat captures one pose per control cycle, but those
dense image-free waypoints do not become topology nodes. The route graph is the prebuilt semantic backbone loaded
once when the service starts. Task execution never creates a route node or writes observations back into that map;
the task start remains a separate revisitable visual-memory location.
The Agent-facing result
contains at most four evenly spaced front-view keyframes. Each keyframe retains its exact position, yaw, timestamp,
and portable navigation point as an independently revisitable evidence location. Known-location navigation restores the
recorded yaw with bounded native Habitat turns after reaching the position, without inference inside the control loop.

Known-location and fixed-node execution treats the controller's `max_control_steps=64` as an internal safety chunk,
not an Agent decision boundary. Runtime automatically replans and continues toward the same point after a healthy
chunk exhausts its budget. It returns to the task Agent only after reaching the point or on a terminal interruption:
persistent stall, path planning/execution failure, cancellation, the 300-second execution timeout, the 1000-action
overall safety limit, or a future high-confidence target event from the perception layer. The current Habitat profile
does not run target detection inside this loop. Keyframes are sampled across the complete multi-chunk trajectory.

Qwen grounding confidence is checked before PointNavigation. A confidence below `0.80` rejects the generated anchor,
so the pixel never enters path planning or control. The task Agent preselects either a bounded viewpoint adjustment or
return to the previous sparse-route location. If no previous location exists, Runtime safely falls back to viewpoint
adjustment. The Relay exposes the measured confidence, threshold, rejection state, and executed recovery.
(start, roughly one-third, roughly two-thirds, and finish). The next task-level Agent decision receives those four
images, the cumulative directional semantic map, and a fresh three-view observation. When the next multimodal tool
result arrives, older-run images are removed from the outgoing LLM request and only their compact structured result
remains. Task-start views are used only for the first decision and are discarded from the active context after the
next visual packet. The original instruction remains in the structured route plan, and the frozen semantic backbone
provides the persistent spatial reference. Durable trajectory images and Web Relay artifacts are not deleted.

Each navigation run also receives a unique subtask ID and writes its selected keyframes into the task-scoped visual
memory index. The latest run remains automatic short-term visual memory. Older runs stay on disk with their pose,
location, target, and frame metadata and can be recalled in a bounded set of at most four images. A task overview
samples across the complete trajectory instead of returning only the newest frames. Recalled images leave the active
LLM window after the next navigation packet; the server-side archive remains intact.

Every navigation or viewpoint request also binds a concise scene summary and LEFT/FRONT/RIGHT semantic labels to the
current Habitat world point and robot yaw. These are task-scoped observation memories and do not mutate the frozen
semantic backbone.
Recorded locations can be passed back into the same point-navigation skill for deliberate backtracking.
At task start, one bounded Qwen text call ranks the fixed nodes. `destination_query` and `object_goal` are evaluated
as a linked relation: the object is located in or functionally associated with the requested place. The selected node
is a regional goal, not a mandatory point-navigation target. Similar objects in unrelated scenes therefore do not
outrank the requested scene.
The Runtime computes an ordered shortest node route from the robot's nearest region to that goal and retains a route
progress index. Only the active adjacent edge supplies the lookahead point, which is transformed through the current
pose/yaw into a robot-relative bearing and `LEFT / FRONT / RIGHT` view gate. Exploratory grounding receives only the
allowed camera views, and its depth is capped by the remaining distance to the next connected node. Route progress
advances by one node only after metric pose enters that node's map-derived arrival radius. When the route is outside
all three views, Runtime performs a bounded turn before asking for another anchor. Direct visible target evidence
bypasses the route gate.
The task Agent follows an anchor-chain policy. After each macro-run it reviews the complete subtask keyframes and
fresh three-view observation, then chooses the most causally useful next visible anchor. Reaching one anchor updates
the regional bearing but does not end the chain. Geometric proximity to a later non-adjacent node cannot skip the
active route segment. Once the robot is inside the target node's arrival radius and all preceding connected segments
are complete, the direction gate opens and all views are searched for the local object.
When no anchor is directly related to the global goal, the Agent next prefers a concrete visible traversable route
transition: continuous corridor floor, an open doorway, or an opening into a wider/deeper unexplored view. Existing
route memory supplies a revisit penalty. Viewpoint adjustment is the fallback when no reliable transition is visible
or one bounded turn can resolve ambiguity.
Each action declares one intent: direct `goal_pursuit`, target-absent `explore_route`, or `backtrack`. Exploration
continues along visible traversable space without requiring the target to be visible. A blocked, repeated, or
no-progress branch triggers return to a useful junction/open-space decision node before another branch is selected.
After `no_progress`, `blocked`, or `hypothesis_rejected`, the task model may select a useful historical backtrack,
viewpoint adjustment, or a new anchor. Runtime validates the selected target and path but does not reinterpret the
semantic failure or replace the model-selected recovery action. Backbone navigation remains available for explicit
recovery or a deliberate region switch. The 5 m camera-ray depth limit remains and is tightened to the next-node
distance during regional travel.

The hierarchical Hermes profile explicitly declares `supports_vision: true`. Its compact task-level toolset exposes
`begin_navigation_task`, `advance_navigation`, `finish_navigation_task`, and the read-only
`recall_navigation_memory`. Semantic-map search remains an internal diagnostic API; it is not a task-level Tool, so
deterministic token matching cannot bypass the Qwen Planner/Actor's semantic node choice. Low-level stage skills are
also hidden from the task Agent so it cannot guess observation IDs or manually replay partial stages. Multimodal
fallbacks always include the exact observation ID and compressed execution report in structured text.

## Start

Use three terminals:

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_hierarchical_habitat.sh
bash scripts/start_hierarchical_relay.sh
bash scripts/start_hierarchical_agent.sh
```

The service is `http://127.0.0.1:8766`, and the read-only relay is `http://127.0.0.1:5916`.
Starting hierarchical Hermes keeps the current Habitat episode by default. Set
`CEE_HIERARCHICAL_RESET_ON_START=1` only when an explicit episode reset is required.
Resetting an episode does not rebuild or reread the backbone. Edit the semantic-map JSON and restart the service when
an intentional map change is required.

The Relay overlays the task's sparse route graph on the Habitat topdown map. It draws traversed edges, location IDs,
fixed backbone nodes, recorded headings, and the current location. A local auto-scaled inset keeps short routes
readable when the full-floor map covers a much larger area. The graph refreshes whenever Agent Memory revision changes.

For one direct atomic check after obtaining an observation ID:

```bash
curl -s http://127.0.0.1:8766/observe -X POST -H 'Content-Type: application/json' -d '{}'
curl -s http://127.0.0.1:8766/skills/hierarchical-navigate -X POST \
  -H 'Content-Type: application/json' \
  -d '{"anchor":"the visible doorway","view":"front","observation_id":"hierarchical-obs-0","semantic_map":{"scene_summary":"corridor facing a doorway","left":["white cabinets"],"front":["doorway"],"right":[]}}'
```
