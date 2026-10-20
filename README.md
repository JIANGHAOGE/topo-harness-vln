# CEE-AgentOS

> Public project snapshot for **Topo-Harness-VLN**. The navigation runtime is
> based on stable commit 2c6d16a; the experiment dashboard is maintained under
> [docs/](docs/). See [PUBLICATION_BASELINE.md](PUBLICATION_BASELINE.md) for the
> exact publication boundary.

Hermes-based robot AgentOS for high-level planning, isolated skill execution,
three-stage verification, source-grounded memory, recovery, traceability, and
edge/cloud routing.

This project is intentionally separate from `CEE-Nav`. The first milestone keeps
OmniNav as a navigation skill boundary rather than importing the old relay state
machine or robot control loop.

## Implemented Kernel

- Typed task, observation, navigation intent, spatial anchor, skill, execution,
  verification, planning, capability, memory, and event protocols.
- Event-driven runtime with isolated skill context, cancellation, timeout,
  stale-result rejection, retry, and bounded replanning.
- Runtime, skill, and global-finish verification as separate stages.
- SQLite-backed source-grounded graph memory and bounded evidence retrieval.
- Capability registry, rule-based edge/cloud model routing, and gated/versioned
  JSON skill-evolution assets with rollback.
- Qwen OpenAI-compatible and Gemini HTTP providers selected through config.
- Real Habitat-GS observations and bounded trajectory execution through the
  existing, proven Habitat bridge.
- Production OmniNav HTTP skill with source-observation binding, official
  left/front/right multipart fields, compact waypoint results, and one bounded
  embodiment action per call.
- Hermes plugin tools for capability discovery, observation, bounded execution,
  evidence memory, stop, and reset.

## Quick Check

```bash
cd /home/jianghaoge/CEE-AgentOS
/home/jianghaoge/miniconda3/bin/python scripts/run_mock_task.py --task "Go straight for 5 m and stop."
PYTHON=/home/jianghaoge/miniconda3/bin/python scripts/run_tests.sh
```

On this workstation the flag avoids loading unrelated ROS pytest plugins from
the global environment.

The mock command runs the full task lifecycle:

```text
IDLE -> PLAN -> START_SKILL -> EXECUTE_SKILL -> VERIFY_RUNTIME
     -> VERIFY_SKILL -> VERIFY_FINISH -> DONE
```

OmniNav-style STOP is represented as skill evidence only. The global task still
must pass the finish verifier.

## Habitat-GS Smoke Test

The first simulator integration intentionally uses a deterministic
`HabitatNavigationSkill` so the test isolates the AgentOS-to-Habitat boundary
from cloud inference. It loads the real `1st_floor_exh_collision_v1` scene,
records front/left/right RGB-D plus a topdown view, executes one bounded Habitat
action sequence inside the AgentOS skill lifecycle, and records a second observation.

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_habitat_smoke.sh
```

Success requires both `runtime_state=DONE` and a changed Habitat pose. Outputs
are written under `logs/habitat_smoke/<timestamp>/`, including `summary.json`,
`sensor_sheet.png`, and `contact_sheet.png`. The core runtime remains independent
of any simulator; the Habitat skill owns this bounded embodiment action.

## OmniNav Skill

Start the persistent Habitat service with OmniNav enabled:

```bash
cd /home/jianghaoge/CEE-AgentOS
OMNINAV_URL=http://127.0.0.1:5802/eval \
  bash scripts/start_habitat_omninav_tools.sh
```

Hermes submits the user goal once with `cee_start_agent_task`. The Agent first
decomposes it from the current left/front/right views and selects one semantic
subtask. OmniNav then runs that subtask locally without per-frame LLM calls. On
local completion, at most four keyframes spanning the run are sent to the Agent
for description, progress assessment, and the next subtask. OmniNav local STOP
ends only that local run; only the boundary Agent can emit global STOP. See
[`docs/agent_supervised_omninav.md`](docs/agent_supervised_omninav.md).

## Hierarchical Version

The independent `hierarchical` profile replaces OmniNav's learned trajectory output with four explicit skills:
Qwen visible-anchor pixel grounding, Habitat RGB-D backprojection, navmesh shortest-path planning, and native Habitat
greedy-follower step execution. It uses separate ports and a separate Hermes toolset, leaving both `omni` and `minimal` unchanged. See
[`docs/hierarchical_navigation.md`](docs/hierarchical_navigation.md).

## Hermes + AgentOS + Qwen

Hermes is the required generic Agent/Harness foundation: it supplies the dialogue
loop, model tool-calling, session context, and plugin system. CEE-AgentOS extends
Hermes according to ABot-AgentOS with embodied planning, isolated skill execution,
verification, evidence memory, and edge-cloud routing. Habitat/ROS2 remains the
physical control and safety boundary. See
[`docs/hermes_integration.md`](docs/hermes_integration.md) for the process layout
and verification command. The Hermes-specific Qwen endpoint uses H800 port
`8011`; the existing service on `8010` is left untouched.

The launcher installs [`configs/navigation_SOUL.md`](configs/navigation_SOUL.md)
as Hermes' identity so the model behaves as a navigation Agent rather than a
general Code Agent. The framework design and extension procedure are documented
in [`docs/agentos_framework_zh.md`](docs/agentos_framework_zh.md).
