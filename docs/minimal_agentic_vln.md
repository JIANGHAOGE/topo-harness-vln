# Minimal-Interface Agentic VLN

This is an isolated implementation of the minimal observe-reason-act interface described in arXiv:2607.26148. It does not replace or modify the existing OmniNav service, skill, relay, ports, or runtime directories.

## Model-facing contract

- `observe()` returns one front RGB image (`640 x 480`, `HFOV 120 deg`, `VFOV 90 deg`, camera height `0.45 m`) to limit visual-token growth. Habitat renders an approximately `832 x 480` projection before resizing to the model-facing output because Habitat-Sim exposes only horizontal FOV directly.
- This experimental variant restricts `step(actions)` to exactly one primitive per call: `0=STOP`, `1=forward 0.25m`, `2=left 15deg`, `3=right 15deg`. It therefore obtains a fresh observation after every primitive instead of using the paper's multi-action bursts.
- The movement budget defaults to 500 actions. `STOP` permanently terminates the episode.
- The Hermes run is limited to 200 model/tool turns and 2400 seconds by default.
- The agent receives no pose, odometry, depth, map, waypoint, collision feedback, external memory, planner, trained navigation policy, or verifier.
- The isolated Hermes session keeps only its normal conversation/tool history and starts fresh for each invocation.

The read-only relay may display pose and topdown diagnostics, but those fields are served through `/relay/live` and never returned by the `observe` or `step` tools.

## Isolated resources

| Component | Default | Existing Omni counterpart |
| --- | --- | --- |
| Habitat API | `127.0.0.1:8770` | unchanged on `8765` |
| Web relay | `127.0.0.1:5915` | unchanged on `5914` |
| Hermes home | `runtime/minimal_vln_hermes_home` | existing home unchanged |
| Artifacts | `logs/minimal_vln/<timestamp>` | existing logs unchanged |
| Plugin | `.hermes/plugins/minimal-vln` | `cee-navigation` unchanged |

## Start

Run each command in a separate terminal:

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_minimal_vln_habitat.sh
```

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_minimal_vln_relay.sh
```

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_minimal_vln_agent.sh "Find the bedroom and stop at the endpoint."
```

To open an interactive Hermes window and enter the task there, omit the instruction:

```bash
bash scripts/start_minimal_vln_agent.sh
```

Open `http://127.0.0.1:5915` for read-only visualization. The model is instructed to begin with `observe()` and only the `minimal-vln` toolset is enabled.

Scene and start pose can be overridden only on the isolated Habitat command, for example:

```bash
bash scripts/start_minimal_vln_habitat.sh \
  --dataset-root /path/to/dataset \
  --scene-id scene_name \
  --start-position X Y Z \
  --start-yaw-deg YAW
```
