# Experiment media

The three `case-*.png` files are publication visuals from the frozen
50-case, five-policy evaluation on the first-floor exhibition scene:

- B06: geometry-order G false stop vs. joint J valid bedroom-cabinet completion;
- O09: object-only O wrong-room robot vs. joint J valid office-robot completion;
- L10: structural J-V false stop vs. hierarchical J valid sofa-side-cabinet completion.

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
