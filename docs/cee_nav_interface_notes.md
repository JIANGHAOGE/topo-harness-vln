# CEE-Nav Interface Notes

Checked local project: `/home/jianghaoge/CEE-Nav`.

Stable interface facts to preserve:

- H800 model servers use HTTP health/reset/eval endpoints.
- Habitat and robot bridges submit multipart images plus JSON payloads.
- Payloads commonly include `instruction`, `reset`, `idx`, `robot_pose`,
  `robot_metrics`, `robot_control`, and camera metadata.
- `robot_pose` carries planar pose as `x`, `y`, `yaw`, plus `source` and `frame`.
- OmniNav adapter output includes `trajectory`, `model_trajectory`,
  `trajectory_coord_frame`, `model_goal_r`, `model_goal_theta`,
  `omninav_arrive_pred`, `planned_actions`, and latency/debug fields.
- Converted OmniNav trajectory convention is `[forward, left, yaw?]`.

Boundaries not to inherit:

- Relay instruction/control UI state.
- Legacy `/step` runtime state machine.
- Controller implementation details.
- Raw frame ring buffers in Main Agent context.

