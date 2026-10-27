# Experiment media

The three current publication visuals are from the frozen
50-case, five-policy evaluation on the first-floor exhibition scene:

- B06: graph-distance G wrong-place kitchen-cabinet stop vs. place-guided P valid bedroom-cabinet completion;
- O09: object-only O wrong-room robot vs. joint J valid office-robot completion;
- L10: structural J-V false stop vs. hierarchical J valid sofa-side-cabinet completion.

`case-l01.png` is retained as an earlier qualitative example, but is no longer
one of the three main-paper comparisons.

Each visual is generated from the frozen eight-node topology, recorded
top-down observation-pose markers, and retained terminal RGB frames by
`paper materials/experiments/make_task_level_qualitative_figure.py` in the
private experiment workspace. The colored lines connect sparse saved poses;
they are not dense controller trajectories. The images illustrate individual
mechanisms and do not replace the full 50-case result table.

Future compressed browser-compatible MP4 clips can also be placed here and
referenced from `docs/data/results.json`, once their source episodes and
timing are verified.

Keep individual videos small enough for ordinary GitHub delivery. Preserve
full-resolution raw recordings outside the repository and publish only the
curated experiment clips required by the project page.
