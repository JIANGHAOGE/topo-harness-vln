# Live Notes

## 2026-08-13 Hierarchical navigation decision context

- Clarified that the four trajectory keyframes are short-term episodic navigation memory, not a hard constraint.
- The next visible anchor should be inferred jointly from the previous subtask, anchor annotation, trajectory keyframes,
  structured execution outcome, and the fresh left/front/right observation.
- Updated `/home/jianghaoge/CEE-AgentOS/configs/hierarchical_SOUL.md` to encode this behavior.
- Refined the policy so each round first classifies the previous subtask as completed, partially completed, or failed,
  then selects the next visible anchor using the original global task, short-term navigation memory, and fresh views.
- A controller `reached` result remains only one piece of evidence and cannot complete the global task by itself.
- Reduced hierarchical Hermes output growth: `model.max_tokens` is now 256 and Qwen thinking is disabled through
  `chat_template_kwargs.enable_thinking=false`, matching the proven minimal configuration.
- The task Agent now emits at most four short decision lines: previous-subtask assessment, next visible anchor and
  global-task reason, global progress, and an optional final result.
- Replaced hierarchical Pure Pursuit execution with Habitat's native greedy geodesic follower. The A*/Detour navmesh
  planning stage remains; execution now emits one native `turn_left`, `turn_right`, or `move_forward` action per cycle.
- Real Habitat validation: 2.83 m geodesic route, 2 waypoints, 12 native actions, 2.40 m traveled, zero collisions,
  reached with 0.43 m remaining under a 0.5 m tolerance. Keyframes were 0/4/8/12.
- Preserved the anchor annotation, four front keyframes, and start topdown map under
  `visuals/habitat_native_step_20260813/`.
- Updated Web Relay layout: left/front/right remain in the first row; independent Top-down and persistent anchor
  annotation panels now sit side by side directly below. Verified at 1600x1000, 1366x768, and narrow layouts.
- Saved the verified 1366x768 page screenshot as `visuals/web_relay_layout_20260813.png`.
- Enlarged the in-page Top-down/anchor row to 280-340 px and gave the anchor panel 58% of the row width. Per user
  preference, no click-to-fullscreen behavior was retained.
- Recovered timing for hierarchical Hermes session `20260813_175958_96c11f` using its original instruction and
  start pose. Time to first observation was 2.956 s and time to first actionable subtask was 3.960 s.
- Agent inference totaled 14.500 s over 13 calls; five anchor-grounding calls were approximately 4.115 s total.
  One Hermes context compression took 24.999 s and reduced rough context from 71,048 to 22,919 tokens.
- Excluded all 144 Habitat native single-step actions from the requested timing summary. Full details are in
  `artifacts/last_hierarchical_task_timing_20260814.json`.
- Updated the hierarchical version so `navigate_to_visible_anchor` always captures a fresh synchronized three-view
  observation after native-step execution. Its result now includes `post_observation` and `next_observation_id`.
- Updated the Hermes tool adapter to return the marked anchor, up to four trajectory keyframes, and the fresh
  left/front/right views in one multimodal tool result. Hermes is instructed not to call `hierarchical_observe` between
  subtasks, eliminating the redundant Agent turn used only to request observation.
- Targeted tests passed: 5 tests in `test_hierarchical_navigation.py` and `test_hierarchical_hermes_plugin.py`.
- Restarted the hierarchical Habitat service and Hermes. A real one-action smoke run returned matching post-execution
  RGB/depth views and `next_observation_id`, then Habitat was reset to the original start.
- Re-ran the prior bed-finding instruction from the identical start pose in Hermes session
  `20260814_112753_e54aaa`. The Agent successfully found the bed after 22 navigation runs.
- Measured user-visible time to first action at 3.940 s. Across 19 non-compression transitions, execution-end to
  next-action latency averaged 3.333 s (median 3.296 s), versus the prior 4.166 s average.
- Confirmed only one explicit initial `hierarchical_observe` call and zero explicit observation calls between navigation
  runs. Automatic observation plus result packaging averaged 0.310 s per transition.
- Three context-compression attempts totaled 37.596 s; two material pauses produced 24.820 s and 18.071 s transition
  outliers. Full benchmark details are in `artifacts/auto_observe_task_benchmark_20260814.json`.

## 2026-08-14 Scene-memory v1 experiment

- Added an opt-in task-scoped structured scene memory for the hierarchical profile. It groups observations using pose
  proximity and semantic overlap, records visit count and attempted anchors, and clears memory on reset or task finish.
- Added separate `start_hierarchical_memory_habitat.sh` and `start_hierarchical_memory_agent.sh` launchers. The stable
  launchers remain available, and an exact pre-experiment source snapshot is preserved at
  `artifacts/hierarchical_auto_observe_before_scene_memory_20260814.tar.gz`.
- Targeted schema, memory, service, and Hermes plugin tests passed: 12 tests in 2.03 seconds.
- Replayed the same bed-finding instruction from the same start pose. Scene matching found 10 unique scenes across
  21 visits; 6 scenes were revisited and 11 visits were repeats. This confirms that the memory can detect prior places.
- The experiment did not complete the task. It stopped after 19 successful navigation runs and 2 route failures when
  Hermes exceeded context capacity (`max_tokens=-923`). Its normal subtask-switch latency was 4.741 seconds, worse than
  the stable auto-observe baseline of 3.333 seconds, which completed the task in 22 runs.
- Root cause: each navigation result retained eight images, while verbose scene descriptions and the complete memory
  index were repeated in tool history. Hermes only strips historical images near compression/failure, too late here.
- The failed task was explicitly finalized as cancelled. Memory cleared from 21 visits to zero and a fresh task ID was
  issued, verifying task-lifecycle isolation.
- Decision: retain v1 as an experimental profile but restore the stable auto-observe profile for normal runtime. A next
  version should keep only the latest multimodal observation, emit compact scene references, and enforce revisit costs
  at runtime rather than relying solely on model instructions.
- Full metrics are in `artifacts/scene_memory_v1_experiment_20260814.json`; final visual evidence is under
  `visuals/scene_memory_v1_20260814/`.

## 2026-08-14 Local Git baseline and experiment branch

- Initialized `/home/jianghaoge/CEE-AgentOS` as a local Git repository. Runtime outputs, logs, virtual environments,
  third-party Hermes files, generated eval artifacts, package metadata, and local environment files are ignored.
- Preserved the currently running hierarchical auto-observe source as commit `2ce1efa` on branch
  `stable/hierarchical-auto-observe`.
- Added annotated tag `stable-hierarchical-auto-observe-20260814` at the same commit.
- Created and checked out `experiment/bounded-visual-memory` for the scene-ID/tag and latest-round-only visual-context
  experiment. The working tree was clean at branch creation.
- The exact pre-scene-memory tar snapshot remains available as a second independent rollback path.
- Branch purpose and launcher commands are documented in `/home/jianghaoge/CEE-AgentOS/docs/versioning.md`.

## 2026-08-14 Scene ID plus semantic tags branch change

- On `experiment/bounded-visual-memory`, replaced the model-visible scene description with exactly `scene_id` and
  `semantic_tags`. Agent input supplies 1-6 tags; Runtime normalizes them and remains authoritative for scene IDs.
- Pose, visit count, attempted anchors, and last execution status remain internal matching metadata. The exposed scene
  index no longer contains coordinates, prose, per-view lists, attempts, or status.
- Historical image behavior was intentionally left unchanged. Each navigation result still returns one marked-anchor
  image, up to four trajectory keyframes, and three fresh camera views; the existing test continues to assert 8 images.
- Added rejection coverage for the old room/left/front/right/landmark schema. Targeted tests passed 13/13, and all tests
  runnable in the Hermes environment passed 57/57. The untouched Habitat adapter test could not be collected there
  because that lightweight environment does not include numpy.
- Committed as `ee381c7` (`feat: compact scene memory to IDs and semantic tags`). Stable branch and tag remain at
  `2ce1efa`.
- Restarted the live test stack from `experiment/bounded-visual-memory`: Habitat PID `2259083` is listening on `8766`
  with scene memory enabled, and Hermes PID `2262566` was opened in a new terminal using the memory Agent launcher.
  The launcher reset the task to `hierarchical-task-20260814T065357695086`; active scene memory started empty. Web Relay
  on `5916` remained running and reported the same task ID.

## 2026-08-14 Scene memory moved out of navigation backend

- Removed `TaskSceneMemory`, `/task/finish`, Memory health fields, and the `--scene-memory` CLI option from
  `hierarchical_navigation_server`. Port `8766` is now a memory-agnostic navigation/embodiment skill backend.
- Added `hierarchical_agent_runtime_server` on port `8767`. It owns scene IDs, semantic tags, revisit metadata, task
  finish/reset lifecycle, observation references, and forwards memory-free navigation requests to `8766`.
- Added Runtime and Relay launchers. Memory Hermes now connects to `8767`; the Memory Relay on `5916` also reads
  `8767`, while Runtime forwards navigation to `8766`.
- Real process validation confirmed that backend health contains no Memory fields, backend `/task/finish` returns 404,
  Runtime health reports `memory_owner=agent_runtime`, Runtime reset clears memory, and Runtime observe returns all three
  camera views at the configured start pose.
- The new tool schema requires `scene_memory.semantic_tags` for Agent navigation, but Runtime strips that field before
  forwarding to the navigation backend. Existing anchor/keyframe/current-view image packaging remains unchanged.
- All tests runnable in the Hermes environment passed: 59 tests. Shell syntax and Python compile checks passed.
- Committed on `experiment/bounded-visual-memory` as `2938998` (`refactor: move scene memory into agent runtime`). The
  stable rollback branch/tag remains unchanged at `2ce1efa`.

## 2026-08-14 Memory Runtime promoted to stable

- Fast-forwarded `stable/hierarchical-auto-observe` from `2ce1efa` through `ee381c7` to `2938998`, incorporating the
  compact scene-ID/tag protocol and isolated Agent Runtime into the stable code line.
- Updated `docs/versioning.md` in commit `2b69252` and tagged the resulting stable baseline as
  `stable-hierarchical-memory-runtime-20260814`.
- Preserved `stable-hierarchical-auto-observe-20260814` at `2ce1efa` as the previous stable rollback point.
- Verified the promoted implementation with the project Hermes Python environment: 12 targeted Runtime, scene-memory,
  navigation, and Hermes plugin tests passed. The first test attempt was invalidated by an unrelated auto-loaded ROS
  pytest plugin; disabling external plugin autoload produced the clean test result.

## 2026-08-14 Rollback and bounded visual context

- Rolled `stable/hierarchical-auto-observe` back to `2ce1efa`, deleted the `experiment/bounded-visual-memory` branch,
  and deleted the temporary `stable-hierarchical-memory-runtime-20260814` tag.
- Added only a Hermes context fix on top of `2ce1efa`: the original user task and latest assistant tool-call/result pair
  are protected, while images in older tool results are deterministically stripped after every tool round without an
  additional LLM call.
- A four-round simulation confirmed that active visual history remains fixed at the latest eight navigation images;
  all older visual results retain text but zero images. Relevant tests passed 8/8.
- Committed as `8425d81` and tagged `stable-hierarchical-bounded-visual-context-20260814`. The original
  `stable-hierarchical-auto-observe-20260814` tag remains at `2ce1efa`.
- Stopped the Memory Runtime and old Hermes process. The active Hermes PID is `2445579`, connects directly to Habitat
  on port `8766`, and loaded the bounded visual-context configuration from `runtime/hierarchical_hermes_home/config.yaml`.
- Restarted the hierarchical Web Relay on `127.0.0.1:5916`, now pointing directly to the stable Habitat service on
  `127.0.0.1:8766`. `/api/state` returned `ok=true` for the hierarchical navigation service.
- Diagnosed repeated navigation HTTP 400 errors as a request-field mismatch: Qwen sometimes returned
  `next_observation_id` as a request argument, while `HierarchicalNavigateInput` only accepts `observation_id`.
- Added a tool-boundary normalization from `next_observation_id` to `observation_id`, clarified the SOUL instruction,
  and added regression coverage. Six focused tests passed; committed as `afee77b`.
- Restarted Hermes after the prior task had ended. Active Hermes PID `2489885` loaded the fix and reset Habitat to task
  `hierarchical-task-20260814T080154252178`.

## 2026-08-14 Bounded visual-context token benchmark

- Analyzed interactive session `20260814_162113_db47b6` for the same bedroom/bed instruction. It made 20 logged API
  calls and 14 successful navigation runs, then stopped incomplete in a laundry/utility room.
- The active transcript retained exactly one visual tool result with eight images. The 13 older navigation rounds were
  represented by 104 text placeholders, confirming that historical images no longer remain in active model context.
- Current input usage was 239,515 tokens, averaging 11,975.75 per API call with a 24,470 peak. The successful baseline
  `20260814_112753_e54aaa` used 589,771 input tokens over 25 calls, averaging 23,590.84 with a 49,972 peak.
- At equal first-20 API-call count, input usage fell 35.81%. At the fairer 14-navigation progress point, usage fell
  15.29%. Mean and peak reductions against the full baseline were 49.24% and 51.03%, but the current run did not finish,
  so raw end-to-end totals are not directly comparable.
- Full LLM context compression remained enabled at the effective 64,000-token threshold but triggered zero times;
  the baseline triggered three attempts totaling 37.596 seconds. Deterministic image pruning remained automatic and
  requires no LLM call.
- Found one startup 400 because the rollback schema still advertised optional `scene_memory`. Removed scene-memory
  fields from the active Hermes schema/result summary, added regression coverage (6/6 passed), and committed as
  `98ec8ee`. The currently open Hermes process needs a normal restart to load this final schema change.
- Full benchmark data is in `artifacts/bounded_visual_context_benchmark_20260814.json`.

## 2026-08-14 premature hierarchical-task exit

- Session: `20260814_162113_db47b6`, instruction: `go to find the bed in bedroom, bedroom is in front of the sofa.`
- The task did not hit the configured turn limit (`22/40`) and did not hit the 65,536-token context limit or invoke full compression.
- Hermes ended the turn because the final Qwen API response contained ordinary assistant text with API metadata `finish_reason=stop` and no tool call. No literal `STOP` was printed in the terminal, and this was not a verified global-task completion signal.
- The final text said it would navigate toward the far opening, but the model failed to emit `navigate_to_visible_anchor`; Hermes therefore treated the plain-text response as a completed chat turn.
- Two preceding responses reached the configured `max_tokens: 256` output limit (`finish_reason=length`). The first occurred when the model had visually identified a likely bed, fragmenting the intended navigation decision across auto-continuation turns and contributing to route drift.
- Root architectural gap: task completion is prompt-controlled only. The runtime does not currently reject plain-text termination while a navigation task is active, and no independent finish verifier gates the end of the Hermes turn.
- Recommended fix: while a navigation task is active, accept termination only through `finish_navigation_task`; convert any premature plain-text response into a short continuation instruction requiring either another anchor tool call or finish verification.

### Implemented active-task tool gate

- Added project-owned `.hermes/plugins/cee-hierarchical/stop_guard.py`; Habitat and the independent Hermes source checkout are unchanged.
- A successful `hierarchical_observe` or `navigate_to_visible_anchor` marks the Hermes session active. A successful `finish_navigation_task` response with `ok: true` clears it; failed finish calls do not.
- While active, the plugin's `llm_request` middleware replaces deferred-search tools with exactly `navigate_to_visible_anchor` and `finish_navigation_task`, sets `tool_choice=required`, and disables parallel tool calls. Qwen therefore cannot end an unfinished task with narration-only text.
- An initial version that set only `tool_choice=required` was rejected during live testing because it forced repeated `tool_search/tool_describe` calls. The final version narrows the tool list and avoids that loop.
- Verification: 8 focused tests passed; a live PluginManager middleware probe returned `changed=True`, `tool_choice=required`, `parallel_tool_calls=False`, and the expected two tools.
- Restarted hierarchical Hermes after resetting Habitat. New process PID at launch: `2701417` (2026-08-14 17:02 CST).

### Tool-search compatibility correction

- Live test exposed `Unknown tool 'navigate_to_visible_anchor'` followed by three invalid-tool retries. Cause: request middleware advertised the direct navigation schema while Hermes deferred-tool search had not placed that tool in the executor's active valid-name set.
- Disabled deferred tool search for the hierarchical profile (`tools.tool_search.enabled: off`). This profile has only five tools, so direct registration is both simpler and lower latency than repeated `tool_describe` calls.
- Raised per-response `max_tokens` from 256 to 384. The prior `finish_reason=length` warning meant only that one Qwen response reached the 256-token generation cap; it was separate from the 65,536-token context limit and from image-history compression.
- Kept the active-task middleware restriction: after a successful observation it exposes only `navigate_to_visible_anchor` and `finish_navigation_task`, requires one call, and disables parallel calls.
- Eight focused tests passed. Restarted Hermes with Habitat reset; effective config confirms `max_tokens: 384` and tool search off. New PID at launch: `2715785` (2026-08-14 17:06 CST).

### Concise visible decision trace and tool-call truncation

- Session `20260814_170640_fb308b` still produced a `finish_reason=length` entry after the 384-token increase. The persisted assistant row had zero visible content and no tool call, so Qwen exhausted the generation budget while forming the required tool response; Hermes auto-continuation recovered on the next call.
- The missing terminal reasoning text was expected from the combination of `tool_choice=required` and the prior SOUL rule to avoid narration: Qwen emitted bare function calls, represented as `...` by Hermes.
- Raised the per-response ceiling to 1024 to give structured tool generation enough headroom. This is a ceiling, not a fixed per-call consumption; successful calls in the observed run used roughly 26-65 output tokens.
- Added required bounded navigation fields `previous`, `reason`, and `progress`; the plugin prints exactly `Previous`, `Next`, and `Progress` lines before executing the atomic navigation skill, then removes these display-only fields before sending the payload to Habitat.
- Added required bounded finish `evidence`, printed as the `Final` line and removed before the finish endpoint call.
- Nine focused tests passed. Restarted Hermes with Habitat reset; PID at launch: `2742066` (2026-08-14 17:14 CST).

### Stable rollback correction

- User identified `2ce1efa` (`stable-hierarchical-auto-observe-20260814`) as the authoritative stable version.
- Discarded all uncommitted stop-guard, forced-tool, schema-summary, output-limit, SOUL, and test changes from the failed experiment.
- Reverted the ignored nested Hermes checkout changes made for the experimental post-response hook.
- Reset branch `stable/hierarchical-auto-observe` from `98ec8ee` to `2ce1efa`; worktree and nested Hermes checkout are clean.
- The stable commit's own hierarchical test set passed: 7 tests.
- Restarted the exact `2ce1efa` hierarchical Hermes version with Habitat reset; PID at launch: `2782677` (2026-08-14 17:26 CST).
- Treat all preceding active-task gate implementation notes as historical failed experiments, not current behavior.

### Local branch cleanup

- Confirmed the CEE-AgentOS repository has exactly one local branch: `stable/hierarchical-auto-observe` at `2ce1efa`.
- No additional CEE-AgentOS local branches remained to delete. Tags were retained because the request targeted branches.
- The `main` branch shown under `third_party/hermes-agent` belongs to the independent ignored Hermes dependency repository and was intentionally left intact.

### Active viewpoint adjustment experiment

- Preserved `2ce1efa` on `stable/hierarchical-auto-observe` and created the isolated branch
  `feature/active-viewpoint-adjustment`.
- Confirmed from both source configuration and a live Habitat observation that left/front/right use the same output
  resolution and FOV: 640x480, HFOV 120 degrees, effective VFOV 90.03 degrees. Their sensor yaws are +90, 0, and -90
  degrees respectively.
- Added `ViewpointAdjustmentSkill` and Hermes tool `adjust_viewpoint`. It accepts exactly one bounded action from
  `turn_left`, `turn_right`, `move_forward`, or `move_backward`, rejects stale observation IDs, and returns collision and
  pose details plus a fresh synchronized three-view observation.
- Updated the hierarchical SOUL so viewpoint actions are used to reveal or disambiguate visible anchors, not as a
  replacement for anchor-based route navigation. The returned views are consumed directly without another observe call.
- Full regression suite passed: 64 tests. A live Habitat `turn_left` test changed yaw by 15 degrees without translation
  or collision and returned all three views; Habitat was reset afterward.
- Committed as `dee4b99` (`feat: add active viewpoint adjustment skill`). Restarted Habitat on port 8766 and Hermes with
  the new branch; Web Relay remains available on port 5916.

### Semantic localization map memory

- Added the read-only `LocalizationMapSkill` and Hermes tool `inspect_localization_map` on
  `feature/active-viewpoint-adjustment`.
- The tool binds a concise scene summary and up to six distinctive semantic labels to the current Habitat world pose,
  returns the marked top-down navmesh together with fresh left/front/right views, and reports nearby known locations.
- Task-local locations within 0.75 m are merged. The result includes a stable location ID, visit count, revisit flag,
  merged labels, and at most 24 recent location records to bound context size.
- Every completed or incomplete anchor navigation automatically records its anchor and stopping pose without an extra
  LLM call. The explicit map tool is reserved for route uncertainty, suspected revisits, or adding richer scene labels.
- Exploration memory is owned by AgentOS rather than Habitat and is cleared by task finish or reset, preserving the
  future ROS/SLAM embodiment boundary.
- Full regression suite passed: 66 tests. A live Habitat test recorded `location-001`, then recognized the same pose at
  0.001 m as a revisit and merged `doorway` into the existing corridor/cabinet labels.
- Committed as `6431b36` (`feat: add semantic localization map memory`). Habitat and Hermes were restarted; ports remain
  8766 and 5916. Preserved live map preview at `visuals/semantic_map_memory/live_topdown.png`.

### Dual visible-anchor and known-location navigation

- Preserved `navigate_to_visible_anchor` for forward exploration through objects, openings, and landmarks visible in the
  current three camera views.
- Added `navigate_to_known_location` as a complementary tool. It accepts a valid task-memory `location_id`, resolves the
  saved Habitat world XYZ point, and runs the existing navmesh `PathPlanningSkill` and native-step `PathExecutionSkill`
  without Qwen pixel grounding.
- Known locations can now be selected for backtracking, repositioning, returning to a semantic doorway, or switching to
  another exploration branch. The tool returns the target semantic record, trajectory keyframes, post-execution
  top-down map, fresh three views, and updated exploration memory.
- Saved navigation coordinates remain fixed when revisiting within the 0.75 m merge radius, preventing repeated returns
  from gradually drifting the callable point.
- Full regression suite passed: 68 tests. A live Habitat loop registered `location-001`, moved forward 1.0 m to create
  `location-002`, then navigated back to `location-001` using the greedy follower: reached=true, 16 control actions,
  0.8 m executed, 0.20 m remaining within tolerance.
- Committed as `712c64a` (`feat: navigate to recorded semantic locations`). Restarted Habitat and Hermes with empty task
  memory after verification; Web Relay remains on port 5916.

### Shared point navigation and bounded visual memory

- Refactored visible-anchor navigation into `AnchorGenerationSkill -> PointNavigationSkill`. Known-location backtracking
  resolves its saved world point and enters the same `PointNavigationSkill`, which internally owns navmesh planning and
  native Habitat path execution.
- Added directional semantic-map updates to navigation and viewpoint requests. Each task-local location stores its fixed
  world point plus up to four yaw-aligned LEFT/FRONT/RIGHT semantic observations.
- Semantic topdown rendering now draws location/direction markers and a non-overlapping recent-location sidebar. A live
  smoke preview is preserved at `visuals/semantic_map_memory/shared_point_skill_smoke_20260814.png`.
- Agent visual context uses a two-part rolling window: up to four keyframes from the latest movement Skill Run plus the
  freshest semantic map and left/front/right views. A newer movement run replaces the previous run's images; compact
  structured results and semantic location memory remain, and disk/Web Relay artifacts are not deleted.
- Anchor annotation images remain durable and visible through Web Relay but no longer enter the task Agent context.
- Full suite passed: 70 tests. Live Habitat smoke test created `location-001` and returned a single top-level
  `PointNavigationSkill` stage for known-location navigation, using `habitat_greedy_follower` with trajectory keyframes.
- Committed on `feature/active-viewpoint-adjustment` as `9cd3997` (`refactor: share point navigation and bound visual
  memory`). Restarted Habitat with the new code on port 8766; Web Relay remains healthy on port 5916.

### Anchor-first target policy

- Strengthened both hierarchical SOUL profiles so a useful currently visible anchor must take priority over
  `adjust_viewpoint`. Viewpoint adjustment is allowed only when no useful anchor is visible, the needed contact area is
  occluded, or visible candidates cannot be distinguished.
- Added a prompt-policy regression test; full suite passed with 71 tests.
- Committed as `0b21071` (`refine: prefer visible anchors over viewpoint adjustment`). Restarted the active Hermes
  process without resetting Habitat; the runtime SOUL contains the new anchor-first rules.

### Anchor-first policy correction

- Corrected the policy from a hard prohibition to a soft priority: visible anchors are preferred by default, but one
  purposeful `adjust_viewpoint` action may search for a better anchor when current candidates are weak, unsafe,
  irrelevant to the global route, or likely to repeat an explored branch.
- Repeated, purposeless, colliding, or oscillating viewpoint adjustments remain discouraged.
- Full suite passed with 71 tests. Committed as `a308d8f` (`fix: allow purposeful viewpoint search for better anchors`).

### Hermes to robot dog migration packaging

- Hermes personal home alone is not enough for physical robot migration. The navigation terminal depends on the
  project plugin `.hermes/plugins/cee-hierarchical`, the hierarchical SOUL/config files, launch scripts, the Hermes
  source/venv bootstrap, and an AgentOS tool service behind `CEE_HIERARCHICAL_TOOL_URL`.
- Recommended code bundle: `agentos/`, `.hermes/plugins/cee-hierarchical/`, `configs/hierarchical_hermes.yaml`,
  `configs/hierarchical_SOUL.md`, hierarchical launch scripts, `pyproject.toml`, `README.md`, selected docs/tests, and
  `third_party/hermes-agent` if the target machine cannot pull the same Hermes source.
- Do not bundle generated or machine-local directories by default: `runtime/`, `logs/`, `results/`, `__pycache__/`,
  `.pytest_cache/`, `.venv/`, and `runtime/hermes_venv/`. Recreate Hermes with `scripts/bootstrap_hermes.sh`.
- Secrets and machine endpoints should be transferred separately or recreated as env vars: `OPENAI_BASE_URL`,
  `OPENAI_API_KEY`, `QWEN_API_KEY`, `CEE_HIERARCHICAL_TOOL_URL`, plus future ROS2 robot topic/domain/safety settings.
- Physical robot migration still requires implementing/replacing the service side currently served by Habitat. The
  plugin can stay stable if the robot service exposes the same tool contract and returns the same observation/result
  schema.

### ROS2 colleague handoff bundle

- Added `docs/robot_ros2_handoff_zh.md` with the stable HTTP boundary, ROS2 capability mapping, current Habitat-specific
  assumptions, safety requirements, startup flow, and first-round acceptance checklist.
- Added the secret-free template `configs/robot_ros2.env.example`; physical robot startup disables Habitat-style reset
  with `CEE_HIERARCHICAL_RESET_ON_START=0`.
- Built an offline bundle containing CEE-AgentOS revision `a308d8f` plus Hermes Agent revision `8f27127` (version
  `0.20.0`). Git metadata, virtual environments, runtime state, logs, Habitat data, model weights, and local API keys
  are excluded.
- Source verification passed: 71 tests. The bundle contains an internal per-file `FILES.sha256` manifest and passed all
  checks.
- Final artifact: `artifacts/CEE-AgentOS-robot-handoff-20260817-a308d8f.tar.gz` (62 MB), SHA256
  `c8bf7ce61cbd9b7690d6dffa38a9bc0844acc7ff5562973e426d9ff8dbb8e940`.

### Latest hierarchical session token diagnosis

- Diagnosed session `20260817_101609_997a91`: 40 task-model API calls, 33 tool turns, 1,192,270 cumulative input
  tokens, 6,877 output tokens, and a peak actual request of 55,303 input tokens. The turn ended at the 40-iteration
  limit.
- Automatic compression ran three times: rough 69,916 -> 28,771 tokens (28.5 s), 68,676 -> 21,714 (14.4 s), and
  65,041 -> 31,117 (0.13 s). After the configured three attempts, context continued growing.
- The dominant retained text was 15 active `navigate_to_visible_anchor` messages totaling 124,578 characters (roughly
  31k text tokens). Their size grew from 4,427 to 14,116 characters because every result repeated the full cumulative
  `exploration_memory` snapshot, including an increasing number of location IDs and scene summaries.
- Historical image pruning was active: the database retained only one compacted eight-image packet and no active image
  blobs at task end. However, 13 successful navigation calls produced 4.93 MB of raw multimodal output over time, so
  the newest map/keyframes/views still contributed image tokens on each round.
- Invalid tool-call loops amplified billing and context churn: repeated missing `name`, `view`, and `semantic_map`
  errors consumed 40 API calls. Active assistant/tool-call text was about 32,493 characters and active tool-error text
  about 13,589 characters.
- Recommended first fix: keep full exploration memory server-side, return only a map revision plus current/relevant
  location delta, and retain only the latest memory snapshot in Agent context. Recommended second fix: route one
  structured navigation decision deterministically instead of letting Hermes repeatedly repair raw tool calls.

### Portable Agent Memory and structured task reasoning

- Created isolated branch `feature/portable-agent-memory-reasoner` from the retained hierarchical baseline.
- Added embodiment-independent navigation-memory schemas for pose evidence, directional semantic observations, place
  nodes, task working state, per-step memory deltas, and bounded memory views.
- Added `AgentNavigationMemory` over the existing graph-memory interface. Places merge within 0.75 m in the same map
  frame, retain at most 12 semantic labels and four directional observations, and persist as SQLite graph nodes and
  traversed-to edges. Habitat now publishes generic `metric_position_xyz`, `map_id`, and `pose_confidence` fields;
  memory code has no Habitat import or Habitat-only field dependency.
- Replaced cumulative `exploration_memory` tool results with one `memory_delta` and a task-relevant `MemoryView`
  containing the current location plus at most five retrieved places. A 40-place regression remains below 7,000 JSON
  characters and restores from SQLite.
- Added structured causal navigation decisions: previous-subtask assessment, route hypothesis, up to three candidates
  with relevance/information-gain/revisit scores, and one selected action with expected evidence, success condition,
  and fallback. The selected action must match a proposed candidate.
- Added `/decision/begin` and `/decision/advance`. The Runtime atomically validates the latest observation ID before
  mutating memory and deterministically routes `visible_anchor`, `known_location`, or `viewpoint_adjustment` to the
  existing low-level Skills. The relay state receives the full selected decision and compact reasoning record.
- Reduced the Hermes tool surface from eight raw tools to `begin_navigation_task`, `advance_navigation`, and
  `finish_navigation_task`. Historical image pruning keeps the latest movement keyframes and fresh visual packet,
  removes old `MemoryView` copies, and preserves old `MemoryDelta`/action summaries.
- Raised the per-response output ceiling from 256 to 768 tokens so the structured JSON decision can finish without
  `finish_reason=length`; prompt rules still require at most three short user-visible lines and disable extra thinking.
- Added the paper-ready Chinese Method draft at `docs/method_portable_memory_reasoning_zh.md`, including formal memory
  retrieval, Skill routing, bounded-context analysis, implementation details, and explicit current limitations.
- Verification: `compileall`, `git diff --check`, and the full test suite pass with 74 tests.
- Saved rollback point on `feature/portable-agent-memory-reasoner`: commit `4ee593f`
  (`feat: add portable navigation memory and task reasoner`). Prior robot handoff artifacts remain untracked and were
  intentionally excluded.

### Portable-memory Habitat smoke test

- Stopped the stale two-day-old hierarchical Habitat process and old Hermes instance, then started commit `4ee593f`.
- Habitat health is ready on port 8766; Web Relay remains on port 5916. Camera metadata reports three synchronized
  640x480 views, HFOV 120 degrees, and VFOV 90.03 degrees.
- `/decision/begin` returned an empty bounded MemoryView from the reset start pose.
- Sent one structured `viewpoint_adjustment` decision through `/decision/advance`. Runtime routed `turn_left`, changed
  yaw from 0 to 0.261799 rad without collision, created `location-001`, stored generic map/source/confidence evidence,
  and returned revision 2 with one bounded current place.
- Reset Habitat after the smoke action. The handoff state is a fresh task with zero places and memory revision zero.
- New Hermes terminal is open and loaded exactly three deferred tools (about 1,073 schema tokens):
  `begin_navigation_task`, `advance_navigation`, and `finish_navigation_task`. No new startup errors were recorded.
- Web Relay was opened at `http://127.0.0.1:5916`.
- Preserved post-action left/front/right frames and semantic topdown under
  `visuals/portable_memory_habitat_smoke/` in this workframe.

### New structured-session context growth diagnosis

- Diagnosed live session `20260817_110651_f59d09`. By API call 29, the actual request had grown to 46,044 input
  tokens; the session later recorded 30 task calls, 796,016 cumulative input tokens, 12,891 output tokens, 56 active
  messages, and 27 tool calls while still running.
- Historical image pruning is working for model requests: only one latest eight-image packet is retained at a time.
  However each raw `advance_navigation` result is 200k-470k characters because it embeds Base64 images in the
  persistent tool message.
- Hermes performs its pre-API compression estimate before the plugin's `llm_request` image-pruning middleware. This
  caused false early compression: call 8 had only 11,332 actual input tokens but was estimated at 64,568. Similar
  mismatches triggered all three allowed compression attempts at 67,138 and 68,263 estimated tokens.
- Compression itself became a growth source: it retained an 11,170-character compaction summary, and the third pass
  was ineffective (`32 -> 32` messages, rough estimate still 41,996). Actual input jumped from 11k before the first
  compression to 19k, 25k, and then 36k after successive passes.
- Text history still grows linearly. A typical structured `advance_navigation` tool call is 1.8k-2.1k characters;
  active assistant tool-call arguments reached 48,306 characters. Historical compact tool results remain about
  4k-7k characters each in storage; repeated MemoryViews accounted for at least 26,356 characters in the inspected
  active slice.
- Five invalid structured decisions caused extra repair calls: overlong scene summary, invalid task-effect enum,
  known-location field mismatch, malformed nested action, and selected-action/candidate mismatch. Tool search and
  describe also consumed four initial model calls.
- Root-fix order: store image references rather than Base64 and hydrate only the latest packet immediately before the
  model call; make pre-compression token estimation operate on the pruned request; roll old task decisions/results into
  a fixed-size server-side working state; simplify and normalize the high-level decision Schema to prevent repair
  loops. Raising compression thresholds alone would only hide the issue.

### Bounded Hermes context fix

- Changed hierarchical tool results to persist `cee-local-image:` filesystem references instead of Base64 payloads.
  The LLM-request middleware hydrates only the selected current image packet immediately before the provider call.
- Replaced image-only pruning with a bounded navigation message window. A movement round keeps its latest map, up to
  four trajectory keyframes, and three current views. If a later viewpoint adjustment exists, the previous movement
  contributes trajectory keyframes only and the viewpoint packet contributes the current map/views. Older navigation
  tool messages and their paired assistant tool calls are omitted from the provider request.
- Disabled automatic Hermes compression in both hierarchical profiles. The actual provider request is now bounded,
  while pre-compression would otherwise estimate the unfiltered persistent conversation and create redundant summaries.
- Added project-local tool-request normalization for protocol aliases, enum synonyms, overlong semantic text,
  candidate/action mismatch, nested action objects, and omitted finish outcome. Unknown finish fields are discarded.
- Added a 40-round regression: the 5-round and 40-round outbound requests both contain exactly three messages and only
  the latest eight hydrated images, with less than 100 characters of serialized-size difference.
- Full verification passed with 76 tests. A live Habitat observation persisted as 2,149 characters with zero Base64
  images and three local refs; provider preparation produced three Base64 images and no unresolved local refs.
- Restarted hierarchical Hermes with compression disabled, reset Habitat, and left Web Relay on port 5916 and the
  hierarchical service on port 8766 for manual testing.

### Live-session follow-up after bounded-context fix

- Inspected session `20260817_120031_e77629`. The image-reference/context-window fix worked: `begin_navigation_task`
  returned 2,023 characters instead of about 149k, navigation results stayed around 7.8k-13k instead of 200k-470k,
  and actual model input stabilized near 11.8k tokens from calls 9-15 without automatic compression.
- Found two real execution failures before stabilization. Hermes rejected incomplete `advance_navigation` calls before
  tool-request middleware ran: the first omitted protocol metadata and the second omitted only `observation_id`.
- Relaxed the public tool schema to require only the selected action. The plugin now owns and overwrites
  `observation_id` from the latest begin/advance result, so every service request still carries a current observation
  identity without requiring the model to copy it.
- Replayed the exact first failed call from `state.db`. Natural Qwen aliases (`action.type`, candidate `anchor`, and
  `direction=forward`) now normalize to the visible anchor `open hallway/living area visible ahead`, front view, the
  original route hypothesis, expected success condition, and fallback rather than a generic target.
- Detected that a normal termination signal interrupted the old turn but left its CLI process alive. Force-stopped the
  stale process, restarted exactly one Hermes instance, and reset Habitat to an empty-memory start state.
- Verification: 78 tests pass. Commit: `257d87c` (`fix: normalize incomplete navigation decisions`).

### Sparse route graph and first-frame global anchor

- Added pose/yaw/map/confidence/navigation-point metadata to trajectory keyframes. The latest four visual keyframes
  remain short-term evidence for reviewing the previous subtask; each frame now references a task-scoped location.
- Separated route construction from visual memory. Every local controller cycle emits an image-free trajectory
  waypoint. Waypoints in the same map/frame within 0.75 m reuse a location; larger transitions create a route node
  and `traversed_to` edge. This forms a sparse graph without adding LLM calls inside execution.
- Persisted the original task instruction and initial synchronized LEFT/FRONT/RIGHT views on one `global_anchor`
  place. Instruction-matched initial objects are exposed as key landmarks in the bounded MemoryView.
- The LLM request window now keeps the original user instruction, initial three-view global anchor, and only the
  latest bounded visual packet. A 40-round regression contains 11 images total rather than growing per round.
- Known-location navigation uses the shared PointNavigationSkill and restores the location's recorded yaw with
  bounded Habitat `turn_left`/`turn_right` actions after reaching its position, without an Agent call.
- Verification passed with 82 tests in the Python 3.10 `internnav` environment.
- Habitat smoke: 15 raw waypoints across 14 controls became 3 locations; four keyframes mapped to
  `location-001, location-002, location-003, location-003`. `location-001` stored the global instruction and initial
  semantics. Revisit to `location-001` succeeded in 23 path controls plus 12 orientation steps, with final yaw error
  about 5.96e-8 rad.
- Smoke JSON is under `logs/sparse_route_*.json`; representative frames are under
  `visuals/sparse_route_global_anchor_smoke/`. Habitat is live on 8766, Web Relay on 5916, and one fresh Hermes
  hierarchical terminal is running after a reset.

### Sparse route graph Web Relay visualization

- Added a lightweight `route_graph` relay payload sourced from task-scoped PlaceMemory nodes and spatial neighbors.
  It exposes location pose/yaw, current/global-anchor flags, node spacing, and deduplicated traversed edges.
- Added a transparent canvas over the existing topdown image. The full-floor overlay uses Habitat/SLAM map origin and
  meters-per-pixel metadata; a local auto-scaled inset keeps short route topology legible.
- Visual encoding: cyan traversed edges, blue route nodes, yellow initial global anchor, green current location, and
  a heading line for every node. The side panel reports node and edge counts.
- Live smoke produced 3 nodes and 2 edges (`L001 -> L002 -> L003`). Desktop and mobile screenshots are preserved as
  `visuals/sparse_route_global_anchor_smoke/web-relay-route-graph-inset.png` and
  `web-relay-route-graph-mobile.png`; relay JSON is under `logs/route_graph_web_*.json`.
- Full verification remains 82 passing tests. Habitat was reset after the visual check; Web Relay now shows an empty
  route and is ready for the next Hermes task.

### Qwen anchor confidence gate and recovery

- Added a fixed Runtime safety threshold of 0.60 between AnchorGenerationSkill and PointNavigationSkill. A lower
  Qwen grounding confidence rejects the pixel before path planning and control.
- Every visible-anchor decision now carries a preselected `low_confidence_recovery`: either one bounded viewpoint
  adjustment or return to the previous sparse-route location. A missing previous node deterministically falls back to
  viewpoint adjustment. No extra task-level LLM call is inserted during recovery.
- Fixed route working-memory predecessor bookkeeping so revisiting the current node or binding visual keyframes no
  longer overwrites the true previous location; completed paths retain their penultimate route node.
- Tool results and Web Relay now expose grounding confidence, threshold, rejection, and actual recovery. Hermes SOUL
  instructs the Agent to choose recovery according to ambiguity versus branch abandonment.
- Regression tests prove confidence 0.35 adjusts viewpoint with zero PointNavigation calls, while confidence 0.40 can
  return to `location-001` through PointNavigation when a prior route node exists. Full verification: 84 tests pass.
- Restarted hierarchical Habitat and exactly one Hermes process. The episode is reset, threshold appears in `/health`,
  Habitat remains on 8766, and Web Relay remains on 5916.

### Bounded persistent visual memory recall

- Preserved every navigation subtask's complete selected keyframe set (up to four trajectory-spanning front frames)
  in task-scoped Agent Memory. Each image reference now carries a unique subtask ID, location ID, pose/yaw, map,
  navigation point, frame index, timestamp, and source target. Initial LEFT/FRONT/RIGHT global-anchor images are also
  indexed; raw image bytes remain in durable log files rather than SQLite.
- The latest subtask keyframes remain automatic short-term visual evidence for the next decision. Older keyframes no
  longer accumulate in the prompt but remain searchable through the read-only `recall_navigation_memory` tool.
- Recall is capped at four images and supports latest-subtask replay, location or semantic filtering, and a complete
  task overview that samples evenly across all trajectory history. A recalled packet is evicted after the next
  navigation packet while its server-side archive remains intact.
- Added visual archive counts to bounded MemoryView and task summaries, updated the Hermes prompts and paper Method,
  and added regressions for task-wide sampling, task isolation, tool output, and prompt eviction.
- Verification: 87 tests pass. Commit: `dcd8b75` (`feat: add bounded visual memory recall`). Hierarchical Habitat is
  live on 8766, Web Relay on 5916, and exactly one fresh Hermes Qwen terminal is running after reset.

### Live Qwen anchor-format failure diagnosis

- Session `20260817_180453_409ca8` first emitted one incomplete Hermes call without the required `action`; Hermes
  rejected it locally before invoking the service.
- The next two valid `advance_navigation` calls reached AnchorGenerationSkill but failed after about 9.9 seconds with
  `Qwen anchor grounding did not return a JSON object`. The local OpenAI-compatible Qwen endpoint returned content
  that the JSON parser could not decode despite a strict response schema. Raw malformed content is not currently
  logged, so it cannot be distinguished as empty versus prose after the fact.
- A third call completed in 10.05 seconds and navigation continued, followed by another successful advance. Habitat
  remained healthy; the live task reached 12 explored locations and memory revision 17. This was a recoverable Qwen
  output-format fluctuation, not a Habitat crash or visual-memory indexing failure.

### Anchor confidence threshold raised to 0.80

- Raised the Runtime Qwen anchor-grounding safety threshold from 0.60 to 0.80. Scores below 0.80 are rejected before
  path planning and execute the decision's preselected viewpoint-adjustment or previous-location recovery.
- Updated both Hermes prompts, implementation documentation, and paper Method text. The regression now verifies that
  confidence 0.75 is rejected, while the existing 0.91 success path remains valid.
- Full verification: 87 tests pass. Commit: `4b31e40` (`tune: raise anchor confidence threshold`). Restarted Habitat
  and exactly one Hermes process; `/health` reports `anchor_confidence_threshold: 0.8`, with Web Relay still on 5916.

### Traversable-route fallback policy

- Added an explicit three-level Task Reasoner priority: select a directly goal-related visible anchor first; when none
  is visible, select a concrete traversable corridor floor, open doorway, or opening into a wider/deeper unexplored
  view; adjust viewpoint only when no reliable transition is visible or one turn can resolve route ambiguity.
- Fallback transitions are ranked through the existing information-gain and route-memory revisit-penalty fields. The
  prompt forbids using furniture, walls, or an inferred invisible room as substitutes for visible floor/threshold
  anchors.
- Updated both Hermes prompts, the deferred tool schema description, hierarchical documentation, and paper Method.
  Full verification: 87 tests pass. Commit: `a566814` (`feat: prefer traversable exploration routes`).
- Restarted exactly one Hermes process with `CEE_HIERARCHICAL_RESET_ON_START=0`; Habitat was not reset and retained the
  active task with 18 route locations and memory revision 45.

### Human-like exploration with semantic topology

- Split navigation memory into two layers. Every selected trajectory keyframe remains a pose/yaw-bound visual
  evidence location that can be recalled or revisited, while the visualized route graph now contains only the global
  anchor, true multi-continuation junctions, and centers of broad open spaces. Dense controller waypoints and ordinary
  corridor stops no longer become topology nodes.
- Added structured navigation intents: `goal_pursuit`, `explore_route`, and `backtrack`. When direct target evidence is
  absent, the Agent is instructed to follow visible traversable floor, door thresholds, corridors, or wider unexplored
  space. Viewpoint adjustment remains available for bounded disambiguation.
- Added a pose-aware `backtrack_location` to bounded MemoryView. It uses the robot's latest physical pose, so a newly
  completed branch can immediately expose the preceding decision node even before that stop is written as a semantic
  place.
- Added deterministic Runtime recovery: after `no_progress`, `blocked`, or `hypothesis_rejected`, a repeated
  `explore_route` action is replaced with known-location navigation to the nearest semantic route node. New direct
  goal evidence remains eligible through `goal_pursuit`.
- Stabilized topology nodes so nearby ordinary observations cannot move or erase a junction/open-space center's pose,
  route type, summary, or traversable directions.
- Updated the Hermes schemas and normalizer, both hierarchical SOUL prompts, Web Relay health metadata,
  implementation docs, and the paper Method section. Full verification: 91 tests pass. Commit: `bffbe5e`
  (`feat: add semantic topology exploration`).
- Restarted a clean Habitat episode and exactly one Hermes Qwen terminal. Habitat is on 8766, Web Relay is on 5916,
  and `/health` reports zero places, zero topology nodes, and anchor confidence threshold 0.8.

### Known-location ID extraction fix

- Diagnosed session `20260818_092007_989fb6`: the Agent correctly identified sofa evidence at `location-025`, but its
  first `known_location` action omitted the dedicated `location_id` field and embedded the ID in the descriptive
  target `the beige sofa visible in location-025 at z≈18.46`.
- The request normalizer previously copied the complete target sentence into `action.location_id`, causing Pydantic's
  `^location-[0-9]{3,}$` validation failure. The immediate retry supplied `location_id=location-025` and succeeded.
- The normalizer now extracts the first valid `location-NNN` token from explicit IDs, action targets, or candidate
  targets while preserving the human-readable target separately. Added a regression using the exact failure shape.
- Full verification: 92 tests pass. Commit: `4852c17` (`fix: extract known location ids from targets`). Restarted only
  Hermes with reset disabled; Habitat retained task `hierarchical-task-20260818T012006555821`, 42 places, and memory
  revision 40.

### Static backbone topology visual prototype

- Extracted a task-independent topology prototype from the complete Habitat Navmesh for `map_id=1st_floor_exh`
  (`378x512`, `0.0901888 m/px`). This was an offline visual experiment and did not modify the live navigation runtime.
- Retained navigable components, generated a one-pixel medial skeleton with Zhang-Suen thinning, clustered nearby
  branch pixels, and rejected jagged-boundary false junctions unless three branches persist through a 12-pixel
  (about 1.1 m) radial ring.
- Classified broad local-clearance maxima as open-space centers. The final prototype contains 1,891 skeleton pixels,
  3 junctions, and 5 open-space centers, with no equidistant corridor nodes.
- Saved comparison image, full-map overlay, node JSON with pixel/world coordinates and clearance, and the reproducible
  prototype script under `visuals/topology_skeleton_prototype/` and `artifacts/topology_skeleton_prototype/` in this
  workframe. The main comparison image is 1328x1132.

### Living Anchor Agent development report

- Created a living development-report folder at `/home/jianghaoge/CEE-AgentOS/develop report/` rather than a short
  final-state summary. The main report records the current architecture, Tools versus Skills, execution chain, portable
  memory, bounded visual context, semantic topology, performance, failure analysis, rollback points, migration boundary,
  and next steps.
- Added `development_history.md` with the complete progression from the independent P0-P3 AgentOS kernel through
  Habitat smoke tests, Minimal VLN, Gemini/Qwen tests, Agent-supervised OmniNav, hierarchical anchor navigation,
  native-step execution, auto-observe latency work, Scene Memory v1 and rollback, active viewpoint adjustment,
  portable Agent Memory, bounded provider requests, route graph, confidence recovery, semantic topology, and the
  static backbone prototype.
- Added `process_image_index.md`. All 62 archived images are referenced, including every retained Web Relay layout
  iteration, Habitat and Hermes validation sheets, anchor and trajectory keyframes, Scene Memory v1 evidence,
  semantic-map and pose-revisit tests, route-graph views, and topology prototype outputs.
- Recovered all 10 user-provided debugging screenshots from the session JSONL into the report. They document Gemini
  HTTP 429 quota failure, unavailable navigation service, hierarchical tool loops, Web Relay fill feedback, premature
  plain-text task exit, unknown-tool retries, output truncation, Qwen anchor JSON failure, begin-tool errors, and the
  known-location ID validation bug.
- Copied benchmark JSONs, topology node data, the reproducible extraction script, and this live-notes snapshot into
  `develop report/evidence/`. No API key or secret was copied into the report.
- Verified 98 Markdown links with zero missing targets, 62/62 image references, clean `git diff --check`, and the full
  project suite: 92 tests passed in 7.77 seconds.

### Unified Habitat/ROS2 backbone-topology design

- Clarified that the current Habitat prototype operates on a complete Navmesh-derived traversability raster. Habitat
  Navmesh and ROS2 OccupancyGrid require different input adapters, but can share connectivity filtering, distance
  transform, medial skeleton, stable-junction, open-space-center, and edge extraction after normalization.
- Defined a portable `NavigabilityGrid` boundary with map ID, geometry revision, frame, resolution, origin, free mask,
  and optional unknown mask. Backbone caches must also include robot radius and safety margin because embodiment size
  changes traversability.
- Confirmed that every task injects its start pose into a task overlay. A nearby backbone node receives the
  `task_start` role; otherwise a dynamic start node connects to the nearest reachable skeleton edge. The static map
  cache remains unchanged.
- Confirmed sparse online updates: ordinary control points, stops, and visual keyframes do not enter topology. A
  semantic point is admitted only when directly tied to the instruction, its semantic confidence is at least 0.90,
  and it has valid map-frame pose evidence. Duplicate evidence merges into an existing node.
- Kept all trajectory keyframes in Visual Memory for recall and pose revisit even though they do not become topology
  nodes. Separated ROS2 geometry-revision handling from semantic overlays so changed walls/connectivity cannot be
  hidden by semantic updates.
- Added `/home/jianghaoge/CEE-AgentOS/develop report/topology_runtime_design.md` and updated the living report and
  development history. Markdown validation now checks 100 relative links with zero missing targets.

### Correction: report ownership and implemented backbone runtime

- Reverted the unrequested report-only topology additions above. The user decides when `develop report/` is updated;
  this implementation turn did not modify that report and `topology_runtime_design.md` is absent.
- Added an environment-neutral `NavigabilityGrid` contract. Habitat converts its Navmesh to a free-space raster;
  ROS2 OccupancyGrid dictionaries convert through a separate normalizer. Both then use the same sparse-backbone
  extractor.
- Added topology caching keyed by map source, `map_id`, and geometry revision. A repeated task on unchanged geometry
  loads the cached graph instead of extracting it again.
- `begin_navigation_task` now records the task start as `location-001`, materializes cached map junction/open-space
  nodes afterward, connects the start to the backbone, and injects the bounded route graph into Hermes once.
- Ordinary stops, model-declared junctions, control waypoints, and trajectory keyframes no longer enter topology.
  Visual keyframes remain in Visual Memory. A reached anchor becomes a `semantic_landmark` only when both Qwen
  grounding confidence and Agent task relevance are at least 0.90.
- Hermes receives only the initial bounded route graph and later `topology_update` deltas, avoiding repeated full-map
  context growth. Full verification: 97 tests passed in 8.65 seconds.

### Mapping-provider decoupling correction

- Removed topology extraction and cache ownership from `HabitatGSAdapter`. Embodiment adapters remain responsible
  only for observation and motion.
- Added a Mapping-layer `NavigabilityMapProvider` protocol with separate Habitat Navmesh, ROS2 OccupancyGrid, and
  static/test providers. The Agent Runtime depends only on this protocol and the normalized `NavigabilityGrid`.
- Habitat is selected only in the current simulator composition root. A robot deployment injects
  `ROS2OccupancyGridMapProvider` while reusing the same topology extractor, cache, Agent Memory, start-node overlay,
  and high-confidence semantic-node admission.
- Moved topology cache ownership to the map provider and made its default directory stable across Habitat log
  sessions. Added extractor parameters and algorithm version to the cache key.
- Verification after decoupling: 98 tests passed in 8.57 seconds. No development report files were updated.

### Runtime backbone startup validation for 1st_floor_exh

- Started the real hierarchical Habitat backend and called `POST /decision/begin` once for
  `map_id=1st_floor_exh`.
- The first production extractor attempt exposed a quality regression: it saturated the 24-node cap and classified
  all nodes as junctions. Preserved that cache only as a diagnostic and replaced the extractor with the previously
  validated prototype parameters and Zhang-Suen thinning.
- The corrected startup produced exactly 8 static backbone nodes: 3 junctions and 5 open-space centers. The task
  overlay added `location-001` as the global start, resulting in 9 route nodes and 8 edges.
- Active cache: `/home/jianghaoge/CEE-AgentOS/runtime/topology_cache/1st_floor_exh-c6a3eb6201f4051d534e.json`.
  Geometry revision is `8b9721c01633bc3b1e65`; extraction cache key is `c6a3eb6201f4051d534e`.
- Copied the validated JSON to `artifacts/runtime_backbone_startup/` in this workframe. The backend remains online
  at port 8766. Full verification after the runtime extraction change: 98 tests passed in 9.04 seconds.

### Reset-time Web Relay backbone overlay

- Fixed the startup gap caused by `start_hierarchical_agent.sh`: its default `/reset` previously cleared the
  task overlay and deferred backbone loading until `begin_navigation_task`, temporarily leaving Web Relay at zero
  nodes.
- `/reset` now resets Habitat and task memory, captures the new start observation, creates `location-001`, restores
  the cached 8-node backbone, and exposes the resulting route graph immediately. Old task semantics are still cleared.
- Validated without calling `/decision/begin`: reset returned 9 nodes and 8 edges, and Web Relay `/api/state`
  independently returned the same graph with 3 junctions and 5 open-space centers.
- Captured the rendered Web Relay at
  `visuals/reset_backbone_web_relay/web_relay_after_reset.png`; the topdown canvas visibly contains L001-L009 and
  the sidebar reports 9 nodes / 8 edges before any task instruction.
- Full project verification: 98 tests passed in 9.35 seconds. Backend remains online at 8766 with a cache hit.

### Three-view anchor identity and fixed semantic backbone

- Diagnosed the anchor mismatch from the latest Habitat run. Annotation files used the correct source RGB pixels,
  but missing planner `view` fields silently defaulted to `front`, and Qwen could be forced to mark an anchor that
  was absent from that image.
- Changed anchor grounding to inspect one synchronized LEFT/FRONT/RIGHT triptych. The camera names and physical panel
  indices are rendered into the image; Qwen returns `panel_index=0/1/2`, and runtime maps that index to the camera
  view. The planner view is no longer used to bias visual grounding.
- Added `anchor_visible`; absent anchors produce zero confidence and no annotation instead of an arbitrary point.
  Pixel-to-depth projection and point navigation use the Qwen-selected camera, not the planner hint.
- Web Relay now labels the anchor panel with the actual grounded view and source observation ID. A deliberate live
  test with requested `RIGHT` correctly selected `FRONT`, confidence 0.95, and generated a `_front.jpg` annotation.
- Replaced runtime semantic-node creation with a fixed topology policy. Navigable route nodes are limited to the task
  start plus the startup cached backbone. High-confidence task semantics annotate the nearest existing backbone node;
  keyframe evidence keeps exact pose but remains outside the route graph.
- Added bounded `nearby_backbone_nodes` to Agent Memory. When an action has `intent=explore_route`, runtime prefers an
  adjacent unexplored backbone location, observes there, and later writes strong task semantics back to that node.
- Live reset reports 9 route nodes and 8 edges. The node count remained 9 after anchor execution. Full project
  verification: 101 tests passed in 10.30 seconds.
- Preserved visual evidence under `visuals/three_view_anchor_fixed_backbone/`: the grounding triptych, grounded front
  anchor, and Web Relay screenshot. No files under `develop report/` were changed.

### Full fixed-backbone semantic exploration

- Reset Habitat and traversed the complete fixed graph for `map_id=1st_floor_exh`: 9 route nodes, 8 edges, and 11
  graph-connected navigation legs. The route graph remained fixed at 9 nodes throughout the survey.
- At each node, captured synchronized LEFT/FRONT/RIGHT views at the arrival heading, rotated in place approximately
  180 degrees, and captured a second synchronized triplet. Qwen received one labeled six-view mosaic per node.
- Saved directional semantics for both headings, place type, retrieval text, up to 12 retrieval tags, visible search
  cues with orientation/view provenance, model confidence, and uncertainty. No unseen room was labeled as a bedroom;
  doorway/corridor cues remain searchable exploration candidates.
- All 9 nodes were surveyed. Identified place types were 2 laboratory nodes, 2 corridor nodes, and 5 open-hall nodes;
  per-node model confidence ranged from 0.85 to 0.95.
- Durable project output: `/home/jianghaoge/CEE-AgentOS/logs/semantic_backbone/20260818_full_exploration_r2/`.
  Machine-readable outputs are `semantic_backbone.json` and `semantic_index.json` (52 indexed terms).
- Preserved review visuals under `visuals/semantic_backbone_full_exploration/`: a clean semantic backbone overlay and
  a 9-node contact sheet containing all 54 directional images. The one-off runner is preserved under
  `artifacts/semantic_backbone_exploration/`. No files under `develop report/` were changed.

### Orientation-invariant editable node semantics

- Added a canonical `NodeSemantics` model for human-style place knowledge: place type, function, objects, landmarks,
  spatial character, task affordances, aliases, confidence, provenance, and notes. Camera direction, pose, yaw, and
  image refs remain evidence and no longer define the semantic identity of a map node.
- Added stable `topology_node_id` values (`backbone-XXX`) to route nodes. Task-local `location-XXX` IDs remain the
  navigation handles, while persistent environment semantics are keyed by the stable topology IDs.
- Added a human-editable semantic map store under `runtime/semantic_maps/<map_id>.json`. Reset/startup validates and
  applies it; `POST /semantic-map/reload` applies edits without resetting Habitat. Invalid map IDs, geometry revisions,
  node IDs, or field types are rejected by Pydantic.
- Converted the complete six-view survey into `runtime/semantic_maps/1st_floor_exh.json` with 8 stable backbone nodes.
  The task start is intentionally excluded because its pose may differ between runs.
- Added `POST /semantic-map/search` and a Hermes `search_semantic_map` tool. Queries rank direct place-type/alias
  matches above mentions in functions or objects, and return both stable topology IDs and current location IDs.
- Live-tested a temporary human override from corridor to bedroom: reload changed `backbone-005 / location-007` to
  `bedroom`, and the query `find the bedroom` returned that location. The production JSON was then restored to its
  original corridor classification.
- Final verification: 105 tests passed in 10.19 seconds; `git diff --check` is clean. The Habitat backend is running
  on port 8766 with all 8 semantic nodes loaded. No files under `develop report/` were changed.

### Current semantic-node visualization

- Generated a standalone visualization from the live `/relay/state` route graph and the editable
  `runtime/semantic_maps/1st_floor_exh.json`; Web Relay itself remains unchanged and continues to show node IDs only.
- The left panel overlays the task start and eight fixed backbone nodes on the current topdown map. The right panel
  shows each stable topology ID, task-local location ID, place type, confidence, function, spatial character,
  landmarks, and task affordances.
- Preview: `visuals/current_semantic_nodes/current_semantic_nodes.png`. The exact relay-state snapshot and reusable
  renderer are under `artifacts/current_semantic_nodes/`.
- Applied user-confirmed semantic corrections: `backbone-003 / location-005` is now `bedroom`, and
  `backbone-006 / location-008` is now `kitchen`. Their image-derived landmarks were retained, while type, function,
  aliases, task affordances, source, and notes were updated as merged human/model knowledge.
- Validated and hot-reloaded all 8 nodes. Search now returns B003 first for `bedroom` and B006 first for `kitchen`.
  Refreshed the standalone visualization with distinct bedroom and kitchen colors; Web Relay remains unchanged.

### ABot-N0 coarse-to-fine navigation research

- Reviewed the official ABot-N0 technical report, especially Sec. 6.1.1-6.1.3. Its Agentic Planner takes the user
  instruction, current observation, long-term Topo-Memory, and short-term episodic visual memory, then emits a sequence
  of executable sub-tasks rather than one monolithic navigation command.
- Topo-Memory has four layers: Block (rooms/districts), Road (doorways/intersections and reachability), Function
  (kitchens, rest areas, elevator halls), and Object/POI (specific entities used as last-meter visual-semantic anchors).
- The paper's memory-aware example directly matches the requested bedroom/bed behavior: for "find an oven," known
  kitchen coordinates first produce deterministic Point-Goal navigation; only after arrival does execution switch to
  local Object-Goal search for the oven.
- The resulting protocol is Approaching (Point-Goal to a known semantic region) -> Reaching (Object/POI-Goal for the
  precise local target) -> optional Interaction, with a Self-Reflector after each sub-task and replanning on failure.
- For CEE-AgentOS, fixed graph connectivity corresponds to Road; a node's bedroom/kitchen identity is a Block/Function
  annotation; bed is an Object-layer anchor. Thus `go to bedroom to find the bed` should retrieve B003, navigate to
  `location-005`, verify region arrival, then ground and approach a visible bed from fresh local views.
- The official ABot-Navigation repository currently exposes benchmarks/evaluators rather than the Agentic Planner or
  Topo-Memory implementation. Research artifacts are preserved under `artifacts/abot_n0_research/`; paper pages 20-23
  are under `visuals/abot_n0_research/`. No project implementation changes were made during this research step.

### Semantic-node-first local anchor execution

- Added a coarse-to-fine task protocol: Hermes decomposes one instruction into `global_goal`, semantic
  `destination_query`, and local `object_goal`. Runtime resolves the destination against persistent node semantics and
  maintains an explicit phase: `approach_node`, `local_anchor_search`, `verify_goal`, or `explore_nodes`.
- Runtime forces point navigation to the selected fixed semantic node before any visible-object anchor can execute.
  After reaching the node, visible anchors and viewpoint adjustment operate locally. Agent-requested node switching is
  accepted only as an explicit `known_location` / `explore_route` decision to another fixed backbone node.
- Added a 5 m default local node radius. A local anchor outside that radius is clipped onto the node-centered boundary
  before path planning, preventing object grounding from drifting into a different semantic region.
- Added a 5 m default pixel-depth cap. If Habitat depth is farther away, camera, robot, world, and navigation vectors
  are scaled along the original projected direction to a 5 m point. Responses preserve `raw_depth_m`, `depth_m`, and
  `depth_limited` for inspection.
- Updated Hermes begin/advance schemas, request forwarding, compact outputs, and both hierarchical SOUL prompts. The
  Agent now receives the selected topology/location ID, object goal, active phase, and depth/scope limiter status.
- Verification: 108 tests passed in 10.25 seconds; focused hierarchical/Hermes tests were 39/39. Live Habitat resolved
  `bedroom` to `backbone-003 / location-005` with score 0.985, overrode a premature visible-anchor action with node
  navigation, reached it in 81 native steps over 12.82 m with no collision, and transitioned to
  `local_anchor_search`. The environment was reset afterward; backend 8766 and Web Relay 5916 remain running.
- Representative initial and post-node Habitat frames are preserved in
  `visuals/semantic_route_live_20260818/`; original live run artifacts are under
  `/home/jianghaoge/CEE-AgentOS/logs/hierarchical/20260818_073641`.

### Place-type-only semantic corrections

- Audited the earlier B003 bedroom and B006 kitchen edits against the preserved two-heading, six-view survey JSON.
  Those edits had changed more than identity: function, objects, spatial character, task affordances, aliases,
  provenance, and notes were also rewritten.
- Restored every visual/model-injected field exactly to the survey output. The production semantic map now differs
  from the raw survey in exactly three lines: `backbone-003.place_type=office area`,
  `backbone-005.place_type=bedroom`, and `backbone-006.place_type=kitchen`.
- Validated all 8 nodes and hot-reloaded the running Habitat backend. Live search resolves office area to B003 first,
  bedroom to B005/location-007, and kitchen to B006/location-008. A new bedroom task plan selected B005 with score
  0.978. The environment was reset after verification. No development report files were changed.

### Unified navigation identifiers

- Fixed the protocol leak where a stable map node such as B005 was executed and displayed as its task-local materialized
  memory handle L007. Agent-facing map navigation now uses `topology_node_id=backbone-XXX` end to end.
- `location_id=location-XXX` is now reserved for task-scoped historical observations and trajectory-keyframe return.
  Both target types remain supported by the same point-navigation execution path, but the schema requires exactly one.
- Semantic route plans no longer expose `selected_location_id`. Node-first planning, automatic policy overrides, node
  switching, local-radius enforcement, semantic-map search, and point-navigation target IDs all use stable B IDs.
- Hermes semantic search strips internal task-local materialization IDs from fixed nodes. Request normalization extracts
  B IDs before considering L IDs. Historical visual recall and backtracking retain their L IDs.
- Web Relay displays `Bxxx` for fixed backbone nodes and `Lxxx` only for the task start or historical positions; graph
  connectivity may still use internal memory keys without exposing them as node identity.
- Verification: 109 tests passed. A live one-step bedroom task selected `backbone-005`; the decision, navigation target,
  policy success condition, and PointNavigation target all reported `backbone-005`, with `location_id=null`. Habitat was
  reset and Hermes restarted. Live artifacts: `/home/jianghaoge/CEE-AgentOS/logs/hierarchical/20260818_075417`.

### Along-route L evidence positions

- Defined trajectory `location-XXX` IDs as task-scoped evidence positions rather than topology nodes. Each recorded
  keyframe now keeps its exact pose/yaw, navigation point, and all available LEFT/FRONT/RIGHT image references.
- Automatic macro-step context remains bounded: Hermes still receives only the latest four front-view trajectory
  frames plus current three views. The complete three-view archive remains server-side and is returned only by an
  explicit location-filtered `recall_navigation_memory` call.
- When a task-relevant object appears while approaching a B node but disappears at the endpoint, the Agent can inspect
  that frame's three views and revisit its L ID. A successful revisit keeps the selected B node unchanged while making
  the L pose the temporary 5 m local-anchor search center.
- Switching to another B node clears the previous L evidence scope. Missing/stale L evidence safely falls back to the
  selected B node instead of treating a fixed node's internal memory record as historical evidence.
- Tests verify three-view recall at one L position and L-centered anchor clipping while retaining the original B goal.

### Strict Agent-facing B/L protocol

- A live screenshot showed the remaining leak: Hermes correctly selected `backbone-005` for bedroom but also received
  the backend materialization ID `location-007`, then narrated its route using the internal L number.
- Added Hermes-side projection for `memory_view`, `route_graph`, `memory_delta`, and topology metadata. Fixed nodes now
  expose only `topology_node_id=backbone-XXX`; graph edge endpoints are rewritten from internal storage keys to B IDs;
  `node_locations` is removed. Only actual observations and trajectory evidence expose `location-XXX`.
- Live validation resolved bedroom to `backbone-005`, rewrote the incoming graph edge to
  `backbone-004 -> backbone-005`, and confirmed `location-007` was absent from the complete Hermes begin summary.
- Final verification: 109 tests passed in 14.04 seconds and `git diff --check` passed. Habitat is running on 8766,
  Web Relay on 5916, and a fresh Hermes terminal is running with the unified B-node/L-evidence protocol.

### Corrected place-type overrides

- Reconciled `runtime/semantic_maps/1st_floor_exh.json` against the preserved original eight-node survey map in
  `artifacts/semantic_backbone_exploration/1st_floor_exh.json` before applying human place-type corrections.
- Restored B003 from the mistaken `office area` override to its original `open hall`. The only differences from the
  original survey are now exactly three `place_type` values: B001=`office area`, B005=`bedroom`, B006=`kitchen`.
- All functions, objects, landmarks, spatial descriptions, affordances, aliases, confidence, source, and notes remain
  byte-for-byte aligned with the original survey JSON.
- Schema validation passed for all 8 nodes and the running Habitat service hot-reloaded all 8. Live semantic search
  ranks B001 first for office area (0.992), B005 first for bedroom (0.978), and B006 first for kitchen (0.985).
- Reset the task and launched a fresh Hermes terminal so no old B003 office-area context survives. Habitat remains on
  8766 and Web Relay on 5916. No files under `develop report/` were changed.

### Final bedroom-node correction

- Superseded the preceding bedroom assignment per the user's latest map correction: B003 is `bedroom`; B005 was
  restored exactly to its original survey type, `corridor`. B001 remains `office area` and B006 remains `kitchen`.
- The semantic map validated and all 8 nodes hot-reloaded. Live search now has one bedroom result:
  `backbone-003` with score 0.985; B005 no longer matches bedroom.
- Reset the Habitat task and launched a fresh Hermes terminal titled `Hermes Qwen - B003 Bedroom` so no B005 route
  plan remains in active context. No development-report files were changed.

### Bed-anchor semantic target loss

- Investigated Hermes session `20260818_161608_59b3c7` in
  `runtime/hierarchical_hermes_home/state.db`. At `obs-1123` and `obs-1202`, Hermes explicitly recognized the bed and
  emitted the legacy nested fields `selected_anchor.semantic_hint="bed - tan/brown bed cover"` and a bed description.
- The request normalizer recognized `visible_anchor` but did not promote the nested semantic hint into flat
  `action.target`. Its default target became `visible traversable anchor`, so the Qwen grounding Skill faithfully
  selected ordinary traversable floor rather than the visible bed. This was a protocol-normalization bug, not a
  visual-recognition failure.
- Updated `.hermes/plugins/cee-hierarchical/request_normalization.py` to promote legacy selected-anchor action type,
  semantic hint/description, and view. Top-level expected evidence and success condition are also preserved.
- Updated both hierarchical SOUL prompts to require the exact object identity in flat `action.target`; the grounding
  Skill still chooses the safe floor contact point in front of that object. Replaying the recorded call now produces
  `target="bed - tan/brown bed cover"` instead of the generic target.
- Added a regression test for the recorded legacy call shape. The focused Hermes plugin suite passes 24/24 and
  `git diff --check` passes. A full-suite attempt used mismatched lightweight environments, while the previous full
  regression remains 110 passing tests.
- The user accepted continued approach after seeing a `find` target. The temporary rule that forced pure find/locate
  tasks to finish immediately on visibility was removed; only the semantic-target transmission fix remains.
- Restarted Hermes as `Hermes Qwen - Bed Anchor Fix`. Habitat remains on port 8766 and Web Relay on 5916. The final
  runtime prompt contains the exact-anchor rule and no forced visible-find finish rule.
- Preserved the representative erroneous annotation at
  `visuals/bed_anchor_target_loss_20260818/obs-1202-generic-anchor.jpg`; the original is
  `/home/jianghaoge/CEE-AgentOS/logs/hierarchical/20260818_080411/observations/0033/anchor_hierarchical-anchorgenerationskill-65fd25daf9824e2c9ff728ef5e4da6ec_front.jpg`.

### Recent-task context and latency audit

- Audited completed Hermes task `go to find the bed in bedroom` in session `20260818_162840_0dc04f`. The final call
  used 26,248 input tokens; cumulative billing input across ten calls was 159,941 tokens.
- Confirmed navigation packets replace the previous decision/result pair. The retained visual window is three initial
  views plus one topdown, four latest trajectory keyframes, and three current views. All 11 images account for about
  3,369 tokens (12.8%), so repeated structured JSON is the larger issue.
- The begin payload costs about 8,378 tokens including images. Its route graph and memory view alone cost 5,477 tokens.
  The latest navigation pair costs about 9,736 tokens, led by another 2,836-token memory view and a 6,012-token result
  JSON. Tool discovery/setup costs about 2,986 tokens; `tool_describe` alone is 2,019.
- Full task wall time was 71.80 seconds. Physical Habitat control occupied 34.66 seconds; non-control overhead was
  37.14 seconds. Ten Agent model calls consumed 31.50 seconds. Stable post-subtask reassessment averaged 4.55 seconds,
  one anchor-grounding call took about 1.1-1.3 seconds, and post-control observation/packaging averaged 0.95 seconds.
- First motion started 19.84 seconds after the instruction. The main outlier was a 15.2-second initial decision call
  containing three unrepairable tool-call argument fragments. Detailed evidence and calculations are in
  `text/recent_task_context_latency_20260818.md`.

### Agent re-entry during node navigation

- Confirmed that the Agent is not called per Habitat frame. `PathExecutionSkill` performs native navmesh actions and
  records keyframes internally until the target is reached, three consecutive non-turn no-progress steps occur, or
  `max_control_steps` is exhausted.
- The normalized default is 64 control steps. Exhausting that budget without reaching the node returns a retryable
  `incomplete` Skill result to Hermes, which always invokes the Agent again. The semantic route remains in
  `approach_node`, but the Agent still spends another inference turn choosing the continuation.
- In the audited bedroom task, the first node run used 64 actions and stopped 3.19 m short; Agent reassessment took
  4.7 seconds, then a second 17-action run reached B003. The later kitchen task showed the same mid-node boundary on
  consecutive 64-step runs.
- Recommended execution contract: keep a bounded controller chunk for safety, but let Runtime automatically continue
  the same B-node goal across chunks, with local observation, collision checks, and deterministic path refresh only.
  Return to the Agent only on node arrival, persistent stall/collision, invalid/unreachable path, cancellation or
  deadline, or a high-confidence task-relevant semantic trigger. Sample four keyframes over the complete node run,
  rather than treating each controller chunk as a new reasoning subtask.

### Atomic point and node execution

- Implemented automatic continuation inside `navigate_to_known_location`. The existing 64-action budget remains an
  internal controller safety chunk; a healthy budget exhaustion captures an internal observation, replans the same
  target, and continues without returning a tool result to Hermes.
- One point/navigation request now returns after `reached`, persistent `stalled`, path/stage failure, a 300-second
  timeout, or a 1000-action overall safety limit. The result exposes `execution_chunks` and `termination_reason`.
  Existing Skill cancellation checks remain active. High-confidence target interruption is reserved for a portable
  Perception event because the current Habitat control loop has no per-frame semantic detector.
- Aggregated control steps, distance, collisions, route waypoints, and four trajectory-spanning keyframes now cover the
  complete multi-chunk run. Historical L positions use the same atomic point-navigation behavior.
- Added a regression with a four-action chunk and a ten-action destination. Runtime used three internal stages but
  returned one completed result with keyframes `[0, 3, 7, 10]`. Full test suite: 111 passed.
- Live Habitat test navigated from reset start to `backbone-003`: two internal chunks (64 + 17), 81 actions, 12.816 m,
  0.219 m remaining, one final `completed/reached` result, and keyframes `[0, 21, 64, 81]`. Elapsed time was 30.597 s;
  no Agent boundary occurred at action 64. Habitat was reset afterward, and the refreshed Habitat/Hermes terminals
  remain connected to ports 8766/5916.

### RIGHT-view bbox anchor regression

- Reproduced the incorrect `obs-1274` annotation for `white humanoid robot standing near glass doors`. The target was
  visibly near x=250 in the RIGHT 640x480 image, while the generated navigation point was `(457, 240)` on the
  refrigerator.
- Replayed the exact triptych and prompt. Qwen correctly selected `panel_index=2` and confidence 0.95, but returned
  `x=715, y=500, coordinate_space="pixels"`. The old code silently reinterpreted the out-of-range values as local
  normalized coordinates, producing `(457, 240)`. The target recognition was correct; the point-coordinate contract
  was ambiguous.
- Changed grounding to send LEFT/FRONT/RIGHT as three independently labeled images and request a fixed local
  normalized `0..1000` bbox. Runtime now derives a deterministic traversable point at the bbox bottom center plus a
  small floor margin. It no longer asks the model to choose or self-report a coordinate space.
- The same recorded observation now returns bbox `[240, 220, 265, 268]` and ground point `(252, 278)`, correctly on
  the robot's foot/floor region. The annotation displays the detected bbox in yellow and the projected navigation
  pixel in red. Reversed min/max values are normalized; zero-area bboxes are rejected.
- Added RIGHT-view regression coverage for view selection, bbox conversion, three-image input, and generated anchor
  coordinates. Full suite passed: 111 tests. Habitat, Hermes, and Web Relay were restarted; ports remain 8766 and
  5916.
- Preserved diagnostics under `visuals/bbox_anchor_regression_20260818/`: `source_right.jpg`,
  `before_wrong_point.jpg`, `after_bbox_ground_point.jpg`, and `input_triptych.jpg`.

### Development report update for 2026-08-18

- Updated `develop report/development_history.md` with stages 21-26: online cached backbone loading and editable
  semantics, node-first semantic routing, exact visible-target propagation, context/latency audit, atomic node/point
  execution, and the RIGHT-view bbox coordinate regression fix.
- Updated `develop report/README.md` to reflect 111 passing tests, the active 8766/5916 services, fixed eight-node
  topology loading, 5 m local anchor scope, internal 64-step execution chunks, and bbox-based grounding.
- Added current visual evidence to `develop report/assets/workframe/`, including reset topology, semantic backbone
  survey, semantic route observations, the lost-bed-target example, and before/after bbox grounding images. Updated
  `process_image_index.md` and refreshed the live-notes snapshot and detailed context/latency audit.
- Verified all local Markdown links in the three report entry documents; no missing links were found. The report now
  contains 93 files before the latest snapshot refresh, and the semantic map validates with 8 nodes.

### Node-first semantic selection diagnosis

- Diagnosed the Chinese task `到办公室找到白色冰箱`. The decomposition produced
  `destination_query=办公室`, but all node semantics use English place labels. Exact search returned no candidate, so
  Runtime selected the nearest fixed node `backbone-006` via its no-match fallback; place-type scoring was never
  reached.
- The current matcher is also structurally weak: it performs substring matching over all place/function/object/
  landmark/affordance/alias text, and gives the same identity bonus to a place-type match and an alias match. English
  `office` returns B001 (place_type `office area`, 0.992) and B004 (alias `office`, 0.985), while Chinese `办公室`
  returns no results.
- Recommended correction: canonicalize multilingual destination labels; select exact canonical place_type first,
  exact aliases second, semantic function/affordance only as fallback; keep object_goal out of primary place routing;
  and represent a no-match nearest node as `explore_nodes`, not as the selected semantic destination.

### Latest completed Node-first route audit

- Audited completed Hermes session `20260818_171530_0ac6a4` for `go to office to find a white fridge`. At task start,
  Runtime searched only `destination_query=office`, selected B001 at score 0.992, and did not use
  `object_goal=white fridge` in the initial node ranking.
- The semantic map already records `refrigerator`, landmark `tall white refrigerator`, and affordance
  `navigate to refrigerator` at B002. The cached topology directly connects B001 and B002 by a 5.996 m edge.
  Hermes therefore discovered information after reaching B001 that was already available before execution, then
  switched to B002 and finished at the visible refrigerator.
- Revised design conclusion: Node-first should remain hierarchical but perform joint task-conditioned node selection.
  It should retain B001 as the matched place node while selecting adjacent B002 as the initial object-evidence node,
  using destination-place match, object/landmark match, and graph association together. This supersedes the previous
  note that object_goal should be excluded from primary route selection.

### Skeleton-constrained backbone edge fix

- Confirmed the reported false visual edge: the old `sparse-backbone-v2` used a Euclidean minimum spanning tree
  inside each free-space component. On the real map, B002-B003 were only 6.956 m apart in Euclidean space but their
  medial-skeleton path was 21.383 m, so the straight edge crossed walls and skipped the actual route structure.
- Replaced Euclidean MST edge construction with multi-source shortest-path Voronoi adjacency on the medial skeleton
  and bumped the extractor cache version to `sparse-backbone-v3`. Nodes are connected only when their skeleton
  regions meet; edge distance is the skeleton geodesic distance.
- Added a free-space-validated simplified world-coordinate polyline to every topology edge. Web Relay renders this
  polyline instead of a source-target chord. The Hermes projection strips edge polylines so visualization geometry
  does not enter Agent context or increase image/token history.
- Real Habitat regeneration preserved all eight node IDs and positions, generated cache
  `63271126e56dc000b09a`, and removed the reported task-local `location-004 -> location-005` edge (fixed-node
  B002-B003). The new topology has eight skeleton-derived backbone edges plus the task-start attachment.
- Updated only the runtime semantic map cache-key metadata; all eight place/object/landmark semantic records were
  reused unchanged. Full suite: 112 tests passed. Habitat 8766, Web Relay 5916, and a fresh Hermes terminal are live.
- Preserved the Web verification screenshot at
  `visuals/topology_edge_connectivity_fix_20260818/web_relay_backbone_v3.png` and the generated cache JSON under
  `artifacts/topology_edge_connectivity_fix_20260818/`.

### Detailed development-report reconciliation

- Rechecked the complete 2026-08-18 live record against `develop report/development_history.md` and
  `develop report/evidence/daily_development_20260818.md` instead of relying on the final code diff alone.
- Expanded both reports with the previously underrepresented work: Mapping Provider decoupling from Habitat,
  the failed 24-all-junction startup extraction and parameter rollback, reset-time backbone restoration, the 54-image
  fixed-graph semantic survey, the standalone semantic-node review view, strict B/L projection, and server-side
  three-view L evidence recall.
- Kept the two visual-anchor fixes distinct: Qwen-selected physical camera/view identity with invisible-anchor
  rejection, and the later RIGHT-view bbox coordinate-contract repair that moved the point from the refrigerator at
  `(457,240)` back to the robot ground contact near `(252,278)`.
- Refreshed the report's live-notes evidence snapshot after completing the reconciliation. No runtime code was changed
  during this report-only pass.
- Verified 140 local Markdown links across the report entry points and today's detailed evidence with zero missing
  targets. `git diff --check` is clean.
