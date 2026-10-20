# Agent-Supervised OmniNav

The OmniNav task endpoint follows this boundary loop:

```text
global task + current left/front/right views
  -> Agent: choose one currently visible anchor
       -> emit only: Navigate towards <visible anchor>
  -> OmniNav atomic subtask run
       -> repeated local visual inference and bounded actions
       -> no LLM calls inside the local run
       -> stop on OmniNav local STOP or local step limit
  -> select at most four front-view keyframes spanning the complete local run
  -> Agent: describe keyframes, assess the previous anchor, ground the next visible anchor or propose STOP
  -> independent Finish Verifier accepts STOP or returns another visible anchor
  -> repeat
```

Long-range navigation is therefore an observation-conditioned chain of local
visual anchors. An anchor must be identifiable in one current camera view, such
as a distinct object, doorway, corridor opening, or room boundary. The Agent may
not send an unseen room or distant semantic destination directly to OmniNav.

`OmniNav STOP` is local evidence. It ends the current atomic subtask run but does
not complete the global task. Only `decision=stop` from the boundary Agent ends
the global task.

The Agent is called once at initialization and once after each complete OmniNav
subtask run. The active context contains the current three views, the previous
subtask and compressed run report, and at most four descriptions selected across
the global execution trajectory. SQLite stores those descriptions in four stable
semantic-event slots; raw image bytes stay in observation artifacts.

Hermes starts the loop with `cee_start_agent_task`. Useful request limits are:

- `max_steps`: total low-level OmniNav frame budget for the global task.
- `max_omninav_steps_per_skill`: local frame budget before control returns to the Agent; default `64`.
- `timeout_s`: timeout for one OmniNav HTTP inference; default `30` seconds.

Start the three-view service and relay:

```bash
cd /home/jianghaoge/CEE-AgentOS
bash scripts/start_habitat_omninav_tools.sh
bash scripts/start_web_relay.sh
```

The relay at `http://127.0.0.1:5914` shows all three RGB views, the smaller
topdown map, active subtask, Agent phase and call count, local OmniNav state, and
the bounded trajectory-memory count.
