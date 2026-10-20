# CEE-AgentOS Architecture

## Positioning

The strict replica places ABot-AgentOS above Hermes and below the user-facing
conversation. Hermes is a thin task client; it does not own embodied planning,
skill execution, memory retrieval, or completion authority.

- Main Agent owns scene-conditioned planning and skill selection.
- Skill Runner owns bounded procedural context and returns compressed results.
- One Verifier supervises runtime, skill-level, and finish-time stages.
- Context Management selects and compresses prompt material.
- Persistent Memory stores source-grounded typed graph evidence outside prompts.
- Skill Evolvement turns later failure analysis into gated, versioned assets.

For the MVP, OmniNav remains one replaceable visual navigation skill. It does not
shape the Agent runtime or own global task completion.

The hierarchical navigation profile now implements the training-free
[ABot-N0/N1 + AgentOS runtime contract](abot_n0_n1_agentos_architecture.md): external Qwen is the low-frequency Slow
System, while A* and the embodiment controller form the Fast System.

## Target Runtime

```text
User / Web UI
      |
Hermes task client
      |
CEE-AgentOS Runtime
  - Edge-cloud model routing
  - Agent Harness
      - Verification-aware ReAct
      - Context Management
      - Skill Evolvement
  - Main Agent / isolated Skill Runner / one multi-stage Verifier
  - Private graph memory / common-memory interface
      |
Skill Registry
  - NavigationSkill: OmniNav
  - PerceptionSkill
  - MappingSkill
  - SearchSkill
  - ControllerSkill
      |
Embodiment Adapter
  - Habitat-GS
  - Robot Dog / ROS2
```

The central rule is: AgentOS hosts the high-level embodied Agent loop; language
handles semantics; structured protocols handle execution. The Agent does not
output velocity commands and does not control the robot frame by frame. It calls
skills with explicit success conditions.

## CEE-Nav Compatibility Notes

The existing `CEE-Nav` stack exposes stable deployment concepts that are useful
as protocol evidence, but its runtime is not migrated:

- Cloud model servers expose `/eval`, `/reset`, and `/health`.
- Habitat and robot bridges send observations through a `/step`-style relay.
- Robot pose usually carries `x`, `y`, `yaw`, `source`, and `frame`.
- OmniNav emits local trajectories in `[forward, left, yaw?]` form after adapter
  conversion, plus fields such as `model_goal_r`, `model_goal_theta`,
  `omninav_arrive_pred`, `planned_actions`, and `trajectory_coord_frame`.

CEE-AgentOS keeps those as adapter-facing payload fields under `SkillResult`
evidence and outputs. It does not import the old relay state machine,
instruction UI state, or controller runtime.

## Current Kernel Boundary

Implemented now:

- Independent package skeleton.
- Typed navigation intent, spatial anchor, observation policy, execution report,
  capability, memory, verification, planning, task, and event protocols.
- Event bus and authoritative embodied state machine.
- Runtime with isolated context, cancellation, timeouts, stale async-result
  rejection, retry, bounded replanning, and one Verifier operating at three stages.
- Persistent source-grounded graph memory with bounded retrieval.
- Skill capability registry, edge/cloud router, bounded evolution traces, and gated JSON assets.
- Mock and Habitat NavigationSkill implementations for lifecycle and simulator tests.
- Production scene-conditioned planner/replanner over live scene, robot state,
  skill contracts, task constraints, and bounded memory.
- Persistent isolated OmniNav execution with local STOP evidence, stagnation and
  collision guards, and compact skill results.
- Fresh post-skill runtime, skill, and global-finish verification; only the
  finish verifier may mark a global task complete.
- Qwen and Gemini provider adapters behind one factory.
- Hermes navigation identity plus one AgentOS task-submission boundary, task
  status, capability discovery, and grounded-memory tools.

Still adapter-level work rather than missing architecture:

- Complete ROS2 command lease, watchdog, collision, and human-takeover adapters.
- Add semantic/vector retrieval behind the existing graph-memory interface.
- Harden the Web Relay and build the production evaluation harness.
