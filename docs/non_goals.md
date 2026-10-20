# Current Non-Goals

- No migration of the old CEE-Nav relay state machine.
- No Agent-generated velocity commands.
- No unbounded frame history in Main Agent context.
- No claim that the unpublished ABot-AgentOS implementation has been reproduced
  line by line; CEE-AgentOS follows the public architecture and technical report.
- No direct modification of Hermes third-party source.
- No generated Python in skill evolution; only gated, versioned JSON DSL assets.
- No claim that the unfinished ROS2 adapter is production-ready until its
  command-lease, watchdog, collision, and human-takeover tests pass. The
  Habitat+OmniNav integration is the currently verified embodiment path.
