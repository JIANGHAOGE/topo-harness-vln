# Hermes 分层导航到 ROS2 本体的迁移说明

本交接包保留当前 CEE-AgentOS 分层导航版本。Hermes、Agent 提示词、工具名和多模态上下文管理可以继续使用；迁移工作的边界是把当前 Habitat 后端替换为 ROS2 本体后端，并保持现有 HTTP 工具协议。

## 包内内容

- `agentos/`：Agent Runtime、Schema、分层 Skill 和后端服务参考实现。
- `.hermes/plugins/cee-hierarchical/`：Hermes 工具插件及历史图片裁剪逻辑。
- `configs/hierarchical_hermes.yaml`：Hermes Qwen 配置。
- `configs/hierarchical_SOUL.md`：任务 Agent 行为约束。
- `configs/robot_ros2.env.example`：本体侧环境变量模板，不包含密钥。
- `scripts/start_hierarchical_agent.sh`：Hermes 启动参考。
- `scripts/start_hierarchical_relay.sh`：Web Relay 启动参考。
- `third_party/hermes-agent/`：可离线安装的 Hermes 源码快照。
- `tests/`：协议与当前行为的回归测试。

包内不包含模型权重、Habitat 场景、运行日志、数据库、Python 虚拟环境或 API Key。

## 保持不变的边界

Hermes 插件通过 `CEE_HIERARCHICAL_TOOL_URL` 访问本体服务，默认地址为 `http://127.0.0.1:8766`。建议 ROS2 版本继续提供以下接口，使 Agent 和插件无需改名：

| 方法 | 路径 | 本体职责 |
| --- | --- | --- |
| `GET` | `/health` | 返回服务、传感器、定位和控制状态 |
| `GET` | `/capabilities` | 返回可用 Skill Schema |
| `POST` | `/observe` | 返回同步三视图、深度、位姿和时间戳 |
| `POST` | `/skills/hierarchical-navigate` | 锚点生成、像素投影、点位规划与执行 |
| `POST` | `/skills/known-location-navigate` | 导航到语义地图中保存的世界点 |
| `POST` | `/skills/viewpoint-adjustment` | 执行一次左转、右转、前移或后退 |
| `POST` | `/skills/localization-map` | 返回定位、地图和已记录语义点位 |
| `POST` | `/task/finish` | 结束全局任务，不触发运动 |

实体机不应把 `/reset` 映射成运动命令。Hermes 启动时设置 `CEE_HIERARCHICAL_RESET_ON_START=0`。

## ROS2 能力映射

当前分层 Skill 依赖 `HierarchicalEmbodiment` 协议。ROS2 Adapter 至少需要实现这些能力：

| CEE-AgentOS 方法 | 推荐 ROS2 数据或接口 |
| --- | --- |
| `observe` / `observe_keyframe` | 三路 RGB、匹配深度、`camera_info`、TF 和 odom；保存为稳定文件路径后构造 `Observation` |
| `pixel_to_world` | 深度像素 + 相机内参 + `tf2`，投影到 `map` 或 `odom` 坐标系 |
| `plan_path` | Nav2 `ComputePathToPose`，或同事现有移动接口的路径规划方法 |
| `execute_world_path` / 单步跟随 | Nav2 `FollowPath`，或现有移动接口；持续返回进度、碰撞、剩余距离和最终位姿 |
| `step` | 有界的视角调整动作；必须经过速度、距离、超时和急停安全门 |
| 定位地图 | occupancy grid 或已有地图渲染；位姿与保存的语义点必须使用同一固定坐标系 |

三路视图的逻辑顺序固定为 `left/front/right`。当前 Agent 按 `640x480`、HFOV 120 度、VFOV 90 度理解图像。若实体相机参数不同，应在观测 metadata 中返回真实参数，并同步修改锚点投影配置，不能只拉伸图片。

每个异步结果必须保留 `episode_id`、`task_id`、`plan_revision`、`skill_run_id`、`observation_id`、图像时间戳、位姿时间戳和 deadline。过期结果不得覆盖新观测。

## 当前 Habitat 专用点

当前 `agentos/embodiments/robot_ros2.py` 只是占位实现。第一轮迁移还需要处理以下 Habitat 专用假设：

1. `agentos/api/hierarchical_navigation_server.py` 直接创建 `HabitatGSAdapter` 和 `ThreadBoundHabitatAdapter`；ROS2 版本应通过独立入口注入 `RobotROS2Adapter`。
2. `agentos/skills/navigation/hierarchical.py` 的 Skill 描述、`supported_embodiments` 和进度文本仍写着 Habitat。
3. `PathExecutionReport.controller` 当前只允许 `habitat_greedy_follower`，需要扩展为 ROS2 控制器标识。
4. `PathExecutionSkill` 当前逐步调用 `follow_navmesh_path_step`。若现有移动接口接收完整路径，应由 ROS2 Adapter 负责异步执行、反馈采样和最多四帧关键帧采集。
5. Habitat topdown navmesh 应替换为实体机 occupancy map；没有地图时可先返回位姿和轨迹图，但 `inspect_localization_map` 的输出 Schema 仍需满足。

建议保留 Habitat 服务作为协议参考，不直接在原文件中混入 ROS2 分支。新增 `robot_navigation_server.py`，让两个后端共享 Schema、Skill 和 Hermes 插件。

## 安装与启动

在目标机器解压后：

```bash
cd CEE-AgentOS-robot-handoff
bash scripts/bootstrap_hermes.sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
cp configs/robot_ros2.env.example configs/robot_ros2.env
```

填写本机模型地址和 ROS2 参数后，在终端加载环境变量。先启动 ROS2 本体服务，再启动 Web Relay 和 Hermes：

```bash
set -a
source configs/robot_ros2.env
set +a
bash scripts/start_hierarchical_relay.sh
bash scripts/start_hierarchical_agent.sh
```

ROS2 本体服务启动命令需要在 Adapter 完成后补入。启动 Hermes 前先确认：

```bash
curl -s http://127.0.0.1:8766/health
curl -s http://127.0.0.1:8766/observe -X POST -H 'Content-Type: application/json' -d '{}'
```

## 第一轮验收

1. 静止状态调用 `/observe`，三路图、深度、位姿和时间戳一致且不会驱动机器人。
2. 图像锚点可通过深度和 TF 投影到固定世界坐标系。
3. 点位导航调用现有 ROS2 移动接口，不直接由 Agent 输出速度。
4. 执行过程中采集轨迹，结束后只向 Agent 返回最多四张均匀关键帧。
5. 旧 `observation_id` 的导航请求被拒绝。
6. 急停、超时、失去定位、碰撞或通信中断时停止运动并返回失败原因。
7. Hermes 仍只看到压缩结果、最新关键帧、语义地图和最新三视图。
