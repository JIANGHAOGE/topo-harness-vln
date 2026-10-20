# Local Version Baselines

## Stable hierarchical baseline

- Branch: `stable/hierarchical-auto-observe`
- Tag: `stable-hierarchical-auto-observe-20260814`
- Habitat launcher: `bash scripts/start_hierarchical_habitat.sh`
- Agent launcher: `bash scripts/start_hierarchical_agent.sh`
- Expected health: `scene_memory_enabled=false`

This branch preserves the currently running auto-observe behavior. The optional
scene-memory implementation is present in the source tree but is not enabled by
the stable launchers.

An additional exact pre-scene-memory source snapshot is archived outside Git at:

```text
/home/jianghaoge/.codex/workframes/2026-08-06_07-48-03_先明确定位：Agent_框架应参考_ABot-AgentOS，而不是_ABot-N1。ABot-N1_更适合作为未来替…_6dcb9600b124/artifacts/hierarchical_auto_observe_before_scene_memory_20260814.tar.gz
```

SHA256:

```text
3e5235583245fd99617ce82d74abcbbe7ab3d9bb1c8fe019dddd6d2e79d3373b
```

## Bounded visual-memory experiment

- Branch: `experiment/bounded-visual-memory`

New scene-ID/tag memory and latest-round-only visual context work belongs on
this branch. Do not merge it into the stable branch until the same-start A/B
navigation test passes.
