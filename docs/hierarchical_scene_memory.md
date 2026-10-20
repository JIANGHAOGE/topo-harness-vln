# Hierarchical Scene Memory Experiment

This version adds task-scoped structured scene memory to the hierarchical navigation service. The existing automatic
observation version remains available and does not enable memory unless the Habitat service is started with
`--scene-memory`.

## Start Memory Version

```bash
bash scripts/start_hierarchical_memory_habitat.sh
bash scripts/start_hierarchical_relay.sh
bash scripts/start_hierarchical_memory_agent.sh
```

The Agent records compact left/front/right semantics in the same inference that selects an anchor. No additional model
call is used solely for memory. `finish_navigation_task` clears active memory and creates a new task ID. The memory Agent
launcher also clears the active task if Hermes exits unexpectedly.

## Return To Automatic Observation Version

Stop the process listening on port `8766`, then run:

```bash
bash scripts/start_hierarchical_habitat.sh
bash scripts/start_hierarchical_agent.sh
```

The base Habitat launcher omits `--scene-memory`, so scene recording and matching are disabled.

## Exact File Rollback

The pre-experiment source snapshot and SHA256 are stored in the active Codex workframe artifact:

```text
hierarchical_auto_observe_before_scene_memory_20260814.tar.gz
3e5235583245fd99617ce82d74abcbbe7ab3d9bb1c8fe019dddd6d2e79d3373b
```
