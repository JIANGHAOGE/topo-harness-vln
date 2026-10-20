# Hermes-Based AgentOS Integration

Hermes is the required Agent/Harness foundation of CEE-AgentOS. The project does
not implement a parallel generic planner, session manager, or tool-calling loop.
Instead, it extends Hermes following ABot-AgentOS with scene-conditioned planning,
an isolated Skill Runner, staged verification, embodied context and memory, and
edge-cloud routing.

The deterministic embodiment runtime remains the navigation and safety boundary.
Hermes can discover capabilities, submit a goal, inspect authoritative task
state, and retrieve or record source-grounded evidence. It cannot call the
controller frame by frame or publish raw robot velocity.

```text
Hermes Core
  -> CEE-AgentOS planner/context/verifier extensions
  -> cee-navigation plugin / isolated Skill Runner
  -> Habitat or ROS2 embodiment runtime
      |
      +---- OpenAI-compatible Qwen request -> localhost:18011
                                                   |
                                                   +-> SSH tunnel -> H800:8011
```

The pinned Hermes source is stored under `third_party/hermes-agent`. Runtime
state and its Python environment stay under the ignored `runtime/` directory.

## Dependency Rule

- Reuse Hermes sessions, providers, tool loop, plugins, context compression, and
  generic tracing directly.
- Implement robot-specific planning policy, typed observations, skill lifecycle,
  verification, embodied memory, and stale-result rejection as Hermes extensions.
- Keep controllers, collision protection, watchdogs, emergency stop, and command
  leases in the Habitat/ROS2 embodiment runtime.
- Do not add a second generic LLM loop under `agentos/`; every high-level request
  must enter through Hermes.

## Navigation Identity And Tools

`scripts/start_hermes_qwen.sh` copies `configs/navigation_SOUL.md` into the
isolated Hermes home before startup. This changes the generic Hermes behavior
into a navigation-specific loop without patching Hermes itself.

The `cee-navigation` plugin currently exposes:

| Tool | Responsibility |
| --- | --- |
| `cee_navigation_status` | Readiness and embodiment health |
| `cee_navigation_capabilities` | Machine-readable registered skill contracts |
| `cee_start_agent_task` | Submit one global goal to the AgentOS Main Agent |
| `cee_navigation_task_status` | Read authoritative plan, skill, and verification state without moving the robot |
| `cee_navigation_memory_search` | Compact prior evidence retrieval |
| `cee_navigation_remember` | Store a fact tied to an actual observation ID |

New atomic capabilities are added to `SkillRegistry` with a `SkillDescriptor`.
The scene-conditioned planner selects them internally; individual low-level
skills are deliberately not exposed as competing Hermes control loops.

## Start

Run these in separate terminals:

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_h800_hermes_qwen.sh
```

This starts a Hermes-only Qwen endpoint with vLLM tool calling enabled. It does
not modify or restart the existing Qwen service on port `8010`. Port `8011` is
bound to H800 loopback only and is reached through the SSH tunnel below.
The first model start compiles H800 kernels and can take several minutes;
subsequent starts reuse the persistent cache under
`/user-docker-data/data/jhg/cache/cee-hermes-qwen-vllm`.

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_qwen_tunnel.sh
```

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_habitat_omninav_tools.sh
```

The production default is the dedicated local tunnel endpoint
`http://127.0.0.1:5808/eval`. Start the read-only visualization relay with:

```bash
bash scripts/start_web_relay.sh
```

Open `http://127.0.0.1:5914`. Navigation instructions still enter through the
Hermes terminal; the Web Relay displays the latest three-view observation,
smaller topdown map, current semantic subtask, Agent call count, OmniNav
trajectory, and bounded execution result without advancing the simulator.

`cee_start_agent_task` uses the Agent-supervised OmniNav boundary loop documented
in [`agent_supervised_omninav.md`](agent_supervised_omninav.md). The Agent is
called once at initialization and once after each complete local OmniNav run,
not once per navigation frame.

By default the Hermes tool waits for the submitted task and prints a compact
live trace in the terminal. It reports the initial plan, one progress line per
16 navigation frames, each Post-Agent assessment, independent Finish Verifier
evidence, and the terminal result. These are explicit structured summaries;
hidden model chain-of-thought and per-frame sensor history are not exposed.

The AgentOS runtime captures a fresh observation after every skill execution and
passes that exact observation to runtime, skill, and global finish verification.
Stale asynchronous results are rejected by task and revision identifiers.

Install Hermes once, then start it:

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/bootstrap_hermes.sh
bash scripts/start_hermes_qwen.sh
```

The OmniNav Hermes launcher resets Habitat through the AgentOS `/reset`
endpoint before opening the interaction window. Starting a new window therefore
cancels any active navigation task and restores the configured start pose. For
diagnostic sessions that must preserve the current pose, explicitly opt out:

```bash
CEE_HERMES_RESET_ON_START=0 bash scripts/start_hermes_qwen.sh
```

The default ports are `8011` on H800, `18011` for the local SSH tunnel, and
`8765` for the local CEE-AgentOS Habitat tool service. The Qwen endpoint declares
a 65,536-token context to match Hermes session requirements.

For an automated check:

```bash
bash scripts/start_hermes_qwen.sh -Q -q \
  "Call cee_navigation_status and cee_navigation_capabilities, then submit: find the bedroom."
```

Habitat observations and topdown images are preserved under
`logs/hermes_habitat/<timestamp>/`.
# Historical Hermes-Centered Integration

This document is retained for the `release/cyhy-agentos-version1` behavior. In
`experiment/agentos-replica`, Hermes only submits, inspects, and cancels AgentOS
tasks; Main Agent, Skill Runner, Verifier, Context Management, and Memory live
inside the server-side Agent Harness.
