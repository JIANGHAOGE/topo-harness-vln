# Strict ABot-AgentOS Harness Replica

## Version Boundary

`release/cyhy-agentos-version1` and tag `cyhy-agentos-version1` preserve the
Hermes-centered hierarchical navigation prototype at commit `90d3339`.
This document applies only to `experiment/agentos-replica`.

The implementation reproduces the public ABot-AgentOS component roles, runtime
boundaries, and data flow. It does not claim source-code equivalence because the
official implementation is not public. A*, Habitat controllers, or OmniNav may
implement low-level execution without changing these AgentOS boundaries.

## Authority Boundaries

```text
User
  -> Hermes task client
  -> edge/cloud model routing
  -> Agent Harness
       -> Verification-aware ReAct
            -> Main Agent
            -> isolated Skill Runner
            -> one Verifier at runtime, skill-level, and finish-time stages
       -> Context Management
       -> Skill Evolvement
  -> Skills and Tools
       -> registered Hierarchical Navigation Skill
            -> node navigation / visible-anchor navigation
            -> viewpoint adjustment / historical-location revisit
            -> Qwen bbox / depth projection / A* / bounded controller
  -> private graph memory / optional common cloud memory
  -> Habitat or ROS2 adapter
```

Hermes can submit, inspect, or cancel a task. It does not choose navigation
anchors, call procedural tools in a loop, retrieve embodied memory directly, or
declare completion.

The Main Agent owns the original task and revisable high-level plan. It receives
only compact `SkillResult` records, evidence references, the current observation,
and Context Management retrievals. Full trajectory outputs remain available to
the audit and memory-writing path but cannot enter Main Agent or Verifier prompts.

Each delegated subtask opens a `SkillRunnerSession` with its own progress
history, attempt number, source observation, and completion conditions. The
session closes with a compressed outcome. `Verifier` is one AgentOS component;
it applies runtime progress checks, skill-level semantic checks, and finish-time
global checks. Only finish-time success can authorize global completion.

Context Management selects current observations, compact skill summaries, and a
bounded evidence subgraph for each reasoning call. Persistent Memory remains
outside the prompt. Skill Evolvement records bounded execution traces for the
offline diagnose -> propose -> compile -> gate -> promote lifecycle; only
versioned JSON DSL assets that pass target and regression gates may become active.

## Hermes Tools

The strict plugin exposes only:

- `cee_navigation_status`
- `cee_start_agent_task`
- `cee_navigation_task_status`
- `cee_cancel_agent_task`

Skill discovery, memory retrieval, memory writing, planning, execution, and
verification remain inside AgentOS.

## Launch

Start the embodiment and AgentOS Harness:

```bash
bash scripts/start_agentos_replica_habitat.sh
```

Start the thin Hermes task client:

```bash
bash scripts/start_agentos_replica_hermes.sh
```

The current replica intentionally disables OmniNav. `HierarchicalMacroPlanner`
produces one structured Slow decision per macro boundary. The isolated
`HierarchicalNavigationSkill` executes the existing node, anchor, viewpoint,
historical-location, depth, A*, and bounded-controller tools. Raw geometry and
trajectory outputs stay inside the Skill Runner/audit path; Main Agent receives
only a compressed handoff and selected evidence references before the next plan
revision.

## Public References

- ABot-AgentOS report: <https://arxiv.org/abs/2607.10350>
- Official project repository: <https://github.com/amap-cvlab/ABot-AgentOS>
