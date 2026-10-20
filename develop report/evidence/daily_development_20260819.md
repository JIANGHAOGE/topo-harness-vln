# Anchor Agent 详细开发记录：2026-08-19

> 状态：稳定基线归档 + ABot 架构实验版本
> 稳定分支：`feature/portable-agent-memory-reasoner`，稳定提交 `5ebae16`
> 当前实验分支：`experiment/anchor-chain-node-cognition`，当前提交 `c3e49ad`
> 验证：134 项测试通过，`git diff --check` 通过
> 服务：Habitat `8766`，Web Relay `5916`，Hermes Qwen hierarchical，Qwen tunnel `18011`

本日工作的主线是把 Node-first 实验收敛为一个可复现、可回滚的稳定版本。重点不只是新增语义节点选择，
还包括修复 Agent 文本判断与实际 Skill 不一致、重新定义节点执行边界、冻结环境级语义骨干，并隔离 Reset。

## 1. Agent 判断与实际动作不一致

### 1.1 复现

在 `go to office to find a white fridge` 中，Hermes 的文本推理明确提出前往 B002，因为 B002 的语义中已有
`refrigerator` 证据；但提交给 Runtime 的结构化动作仍是 `visible_anchor + explore_route`，目标为当前开放区域。
B002 只出现在自然语言 fallback 中，没有出现在 `action.topology_node_id`。

Runtime 只按结构化协议执行，不会从解释文本中推断运动命令，因此实际只沿局部锚点移动约 0.2 m。该问题不是
Habitat 路径规划失败，而是“语言结论、结构化动作、最终执行”三者没有闭合。

### 1.2 修复

- 任务内的 `explored` 状态只由当前任务访问和观测产生，不再因环境语义已预载而把全部 B 节点标为已探索；
- Runtime 在结构化动作进入执行前进行混合 Skill 仲裁；
- 与任务目标强相关、尚未访问的 B 节点可以替换低相关的普通探索锚点；
- 可信目标锚点仍可直接执行，不要求所有局部任务都先走节点；
- 低置信或无进展时，历史 L 点回访和视角调整保持同级，由证据强弱决定。

## 2. 90 度主动视角调整

早期视角调整沿用 Habitat 单步 15 度转向，单次改变太小，Agent 常在近似相同画面上重复判断。当前默认目标角度
调整为 90 度，但执行仍由六次 15 度原生动作组成，不直接改 pose，也不瞬移。

Skill 持续读取 yaw、累计包裹后的角度增量，达到目标角度后采集新的三视图。该动作适合以下情况：

1. 目标很可能在机器人侧后方，一次转向即可获得新证据；
2. 当前锚点置信度不足 0.80，需要主动消除视角歧义；
3. 当前地点值得继续观察，但没有足够证据触发 L 点回访。

## 3. 两次局部锚点预算实验及撤回

曾实现“到达一个语义节点后，最多执行两次局部锚点探索；仍未发现目标则返回节点重新观察”的预算。它可以
抑制局部漫游，但会破坏强语义长链任务，例如：

```text
沿走廊前进 -> 右转 -> 找到沙发 -> 前往沙发对面的木门 -> 进入卧室 -> 找床
```

这类任务天然需要超过两个连续锚点。固定次数不是可靠的语义边界，因此该实现连同 Schema 字段、Runtime 覆盖、
Prompt 规则和测试全部撤回。保留无限制的宏观锚点链，由进展、路径状态和完成证据决定何时重新规划。

## 4. 显式报告最终执行 Skill

此前 Hermes 显示的是 Agent 提议，Runtime 可能因为节点优先级或锚点低置信安全门改写动作，用户无法确认机器人
最终执行了什么。现在每次 `advance_navigation` 结果都包含紧凑的 `executed_action`：

| 模式 | 含义 |
| --- | --- |
| `NODE_NAV` | 前往固定语义骨干节点 |
| `VISUAL_ANCHOR_NAV` | Qwen 锚点生成、深度投影和点位导航 |
| `L_REVISIT` | 返回历史观察位置并恢复朝向 |
| `VIEWPOINT_ADJUST` | 原地或小范围调整视角 |

结果同时标记 `target`、`intent` 和 `selection_source`。其中 source 区分 Agent 原始选择、Runtime 仲裁覆盖和
低置信锚点恢复。Hermes 的任务后输出压缩为三行，并首先显示实际动作，避免冗长复述扩大上下文。

## 5. 语义节点选择从词项匹配升级为关系推理

### 5.1 旧方法的问题

旧检索依赖字符串和少量同义词规则。它对 `fridge/refrigerator`、中英文表达和同类物体跨场景重复都不稳健。
更关键的是，只匹配 `destination_query` 会忽略任务中的关系。例如“find the robot in the kitchen”不能只找
robot，也不能只找 kitchen，而应判断：

```text
object_goal located_in_or_associated_with destination_query
```

### 5.2 当前实现

任务开始时增加一次有界 Qwen 文本调用。输入只包含全局任务、拆出的 destination/object、固定 8 节点的紧凑
语义文档以及 JSON Schema；不传图像、不传执行历史。模型必须从已有节点中选一个 ID，并提供结构化置信度、
命中概念和理由，禁止生成新节点或虚构语义。

选择时联合考虑：

- `place_type` 与场所功能；
- objects 和 landmarks；
- affordances；
- destination 与 object 的所属/功能关联；
- 同名目标在不同场景出现时的场所一致性。

若模型调用失败或置信度不足，保留确定性的本地检索降级。自动切换 object-evidence node 时，还要求它与当前
场所节点在骨干图上邻接，避免远处相似物体把任务拉到不相关区域。

真实 `1st_floor_exh` 验证中，任务 `find the robot in the kitchen` 选择 `backbone-006`，置信度 0.90；证据
联合了 kitchen、robotic equipment、robotics 和 robot demonstration，而不是只靠 `robot` 字符串。

## 6. 固定语义骨干改为服务生命周期只读

### 6.1 为什么冻结

环境级语义骨干是地图知识，不应随一次任务 reset、一次临时观测或 Hermes 的语言输出变化。旧流程会在 task
begin/reset 时重新物化并加入启动点，还允许执行结果提升语义，导致同一张地图在不同任务中节点数量、ID 和属性
不稳定，也不利于迁移到 ROS2 静态地图。

### 6.2 当前生命周期

```text
服务启动
  -> Mapping Provider 读取可导航地图
  -> 按 map_id + geometry hash 读取/构建 skeleton backbone
  -> 读取一次可编辑语义 JSON
  -> 必要时仅在内存中兼容等价平面拓扑 revision
  -> 冻结 topology + semantic model

每次 begin/reset/finish
  -> 把同一个冻结模型物化到新的任务 Memory
  -> 不重新提图、不重新读 JSON、不写 JSON、不增加启动节点
```

上午稳定基线中，首帧三视图仍被保存为任务级全局视觉锚点，并具有可回访 pose/yaw，但明确标记
`is_route_node=false`，不属于固定 B 图。下午实验进一步移除了首帧常驻机制，详见第 11 节。无论采用哪一版，
`advance_navigation` 都不再接收或写入可复用节点语义；旧请求即使携带 `node_semantics` 也只为兼容解析，
不会修改环境地图。

`reload_semantic_map` 现在返回 no-op 说明：手工修改 JSON 后必须重启服务才能装载，防止任务中途地图语义突然
变化。对等价的 Habitat 平面 Navmesh revision 只在内存中重绑定，不再为了兼容而改写磁盘 JSON。

### 6.3 验证

- Reset 前后都只有固定 8 个 B 节点；
- route graph 中没有 global/start node；
- 语义 JSON 的 mtime 在 reset、begin 和任务执行后保持不变；
- 既有 node ID、语义和 skeleton 折线边保持稳定；
- 完整回归测试 122 项通过。

## 7. Reset 与 Hermes 隔离

`/reset` 继续保留为 Habitat 服务的运维 API，但没有注册为 Hermes Tool。Agent 可见工具只有：

1. `begin_navigation_task`；
2. `search_semantic_map`；
3. `recall_navigation_memory`；
4. `advance_navigation`；
5. `finish_navigation_task`。

启动脚本过去默认先请求 `/reset`，容易让用户误以为 Agent 获得了 reset 能力，也会在重开交互终端时意外清除
当前 episode。现在 `scripts/start_hierarchical_agent.sh` 默认保持现场；只有显式设置
`CEE_HIERARCHICAL_RESET_ON_START=1` 才在 Hermes 启动前 reset。实测重启 Hermes 后服务 `task_id` 完全不变。

该段记录的是上午稳定基线。当日 `c3e49ad` 进一步把独立 semantic search 合并进 Planner 对四层
Topo-Memory 的读取，当前 Hermes 对外只保留 begin、visual recall、advance 和 finish 四个任务级 Tool。

## 8. 当前稳定版本边界

本次稳定版本保留 Node-first 混合导航：固定节点用于地点和目标关系推理，Runtime 可以在节点导航、视觉锚点、
历史回访和视角调整之间仲裁。它同时保留：

- 三路 640 x 480、HFOV 120 度、VFOV 90 度同步观测；
- 固定 `0..1000` bbox、真实 View 归属、bbox 底边地面点和 0.80 置信度门；
- 最远 5 m 的锚点深度/局部范围截断；
- Skeleton geodesic 骨干边及 Web 折线可视化；
- 64 步内部控制分块、300 秒和 1000 动作总安全界限；
- 稳定基线保留首帧；当前实验版只保留最近完整子任务四关键帧、当前三视图及历史图按需召回；
- B 节点与 L 历史观察位置严格分离；
- 独立 Finish Tool，不把局部导航结束等价为全局任务完成。

## 9. 下午实验方向

当前版本将作为回滚基线。下一条实验分支不删除节点图，但把它从“强制执行优先级”降为“地域认知上下文”：
Agent 主要根据全局任务、当前三视图、上一锚点执行结果和有界轨迹关键帧，连续生成最合适的下一可见锚点；
多次锚点子任务共同完成全局目标。节点只负责场所消歧、区域语义、可选回退和跨区域提示，不再默认强制先到节点。

该方向已在 `experiment/anchor-chain-node-cognition` 上实施，并继续演化为第 10 至 14 节所述的
ABot-N0/N1 + AgentOS 功能架构。稳定 Node-first 分支保留为独立回滚对象。

## 10. 节点引导的连续视觉锚点链

### 10.1 从 Node-first 强制执行到地域认知

上午稳定版会在目标节点得分较高时直接以 `NODE_NAV` 覆盖普通探索锚点。该策略对明确地点有效，但容易把
拓扑节点当成最终动作目标，削弱连续视觉语义导航。实验版保留节点图，但把它改成视觉锚点生成的地域坐标系：

- 节点提供当前区域、目标区域和下一相邻区域；
- Road 连通关系约束允许探索的方向，不允许穿过未连接节点；
- 当前三视图内仍由 Agent 选择最能推进任务的可见物、门洞、走廊或可通行区域；
- 只有显式跨区或恢复时才直接调用 B 节点导航；
- 到达目标区域后，解除方向限制并在节点附近搜索最终对象。

提交 `574763e` 将这一策略写入 Runtime、Hermes Schema、Prompt 和测试。它不是删除拓扑，而是把拓扑从
“强制动作优先级”变成“连续锚点的方向与地域约束”。

### 10.2 连通节点逐段推进

早期实验每轮按机器人几何最近节点重新计算方向。如果机器人偶然靠近后续节点，可能跳过中间连通段并误判已经
到达目标区域。提交 `6027a27` 增加持久 `RegionalAnchorGuidance`：

- `route_node_ids` 保存本轮完整骨干路径；
- `route_progress_index` 只在进入下一相邻节点半径时增加一；
- `next_topology_node_id`、下一节点距离和目标节点距离分别报告；
- `target_region_reached` 只有在所有前序边完成且进入目标节点半径后才为真；
- 探索锚点的允许 View 与当前连通边绑定；
- 锚点深度同时受 5 m 和当前连通段剩余距离约束。

三节点对抗测试把机器人直接放到最终节点附近，Runtime 仍保持 `target_region_reached=false`，必须按
`0 -> 1 -> 2` 完成连通段。真实 bedroom 检查得到
`backbone-006 -> backbone-004 -> backbone-003`，初始活动边为 `006 -> 004`。阶段回归为 126 项通过。

## 11. 移除首帧常驻视觉锚点

上下文审计发现，旧 Hermes 图像中间件会固定保留会话中第一次 `begin_navigation_task` 的三张图。跨任务时它
可能保留旧任务首帧并裁掉新任务证据；即使修正为“最新 begin”，首帧也会在每轮形成固定视觉成本。当前已经有
冻结语义骨干、当前 pose 和原始任务作为稳定坐标系，因此不再需要让首帧永久驻留模型上下文。

提交 `6a07b48` 完成以下修改：

- 首帧只参与第一次决策，下一视觉包到达后退出活跃上下文；
- 原始任务仍保存在计划和 Task Working Memory；
- 最近完整宏任务最多四张前视关键帧继续自动进入下一轮；
- 更早关键帧继续保存在服务端，可按地点或语义有界召回；
- 普通任务观察不再与同位的固定 `map_backbone` 记录合并，避免覆盖骨干语义；
- reset/start 后仍保持固定 8 个 B 节点，不产生第 9 个任务起点节点。

实验分支回归 125 项通过；稳定 Node-first 分支同步形成 `83ef261` 和回滚标签
`stable-frozen-semantic-backbone-no-firstframe-20260819`。

## 12. ABot-N1 CoT 与 Slow/Fast 接口调研

本阶段重新核对 [ABot-N1 技术报告](https://arxiv.org/html/2607.10383)，重点不是复制模型训练，而是复刻
训练以外的输入、输出和调度职责。确认的关键设计如下：

1. Slow 输入由历史前视图、当前三视图、任务和上一轮 CoT/像素决策组成；
2. Instruction-Following 使用 `Finished Sub-instruction` 与 `Ongoing Sub-instruction`；
3. Object-Goal 使用 `Target Localization and Key Context` 与 `Path Planning`；
4. Point/POI/Person-Following 不强制输出语言 CoT；
5. `target_pixel` 表示可见目标定位，`affordance_pixel` 表示短程可通行落点；
6. Slow 低频更新，Fast 持续消费最近一次可用 Slow 结果。

当前工程不复刻 N1 的训练数据、权重和学习型动作专家。用户确认可以用现有 A* 与 Habitat/ROS2 单步控制器作为
功能等价 Fast System：Slow 负责产生结构化语义目标，Fast 把目标转换成实际轨迹并执行。

## 13. ABot-N0 Planner、Topo-Memory 与 AgentOS 组合

结合 [ABot-N0 技术报告](https://arxiv.org/html/2602.11598)，当前架构不再只保留 ABot 名称，而是实现以下
可执行角色：

| ABot-N0 角色 | 当前实现 | 运行职责 |
| --- | --- | --- |
| Agentic Planner | `AgenticPlannerOutput` / `AgenticPlan` | 读取观测与两类 Memory，产生类型化粗到细子任务 |
| Actor | `advance_navigation` / `SelectedNavigationAction` | 为活动子任务选择一个宏观 Skill |
| Episodic Memory | 最近宏任务关键帧与服务端历史视觉索引 | 提供短期执行证据与按需回忆 |
| Topo-Memory | `TopoMemoryView` | 提供 Block、Road、Function、Object/POI 四层长期空间知识 |
| Self-Reflector | `SubtaskReflection` | 判断上一子任务完成、部分完成、失败或是否重规划 |
| AgentOS Verifier | Runtime / Skill / Finish Verifier | 分离局部执行成功与全局任务完成 |

### 13.1 真正的 Planner 边界

旧流程可能在 Agent 看到当前三视图和拓扑前就按字符串选定节点。现在 `begin_navigation_task` 只接收
`global_goal`，先返回当前同步三视图和四层 Topo-Memory，并设置 `planner_required=true`。第一轮
`advance_navigation` 必须同时携带 Planner 计划与第一个 Actor 动作。后续只有 Self-Reflector 明确
`should_replan=true` 时才能替换计划。

Planner 子任务采用严格粗到细合同：

- `Approaching` 只能是 Point-Goal；
- `Reaching` 只能是 Object-Goal 或 POI-Goal；
- `Interaction` 只能是 Instruction-Following 或 Person-Following；
- 每个子任务包含稳定 ID、目标、成功条件、状态和可选骨干节点；
- 顺序、活动索引、节点合法性以及当前 Task Type 一致性均由 Pydantic 验证。

### 13.2 四层 Topo-Memory

`AgentNavigationMemory.topo_memory_view()` 将冻结骨干和可编辑语义 JSON 投影为：

- **Block**：房间/区域、place type、场景摘要、位置和当前任务访问状态；
- **Road**：真实可通行连接、skeleton geodesic 距离和岔路/开放空间过渡；
- **Function**：区域功能与任务 affordance；
- **Object/POI**：附着在稳定节点上的对象与地标。

Habitat Navmesh 与未来 ROS2 OccupancyGrid 都先转换成环境无关 `NavigabilityGrid`，下游 Planner 和 Memory
只读取统一四层合同。B 节点与任务级 `location-NNN` 视觉证据仍严格分离。

## 14. 任务专用 CoT、双像素与 Slow/Fast 时序

### 14.1 任务专用 CoT

Hermes 对外 Schema 已删除通用长篇 `route_hypothesis`、候选评分和旧 assessment 字段。运行时只允许：

- Instruction-Following：`finished_sub_instruction`、`ongoing_sub_instruction`；
- Object-Goal：`target_localization_and_key_context`、`path_planning`；
- Point/POI/Person-Following：`cot=null`。

错误 Task Type、混合字段、额外字段或不匹配活动子任务的 CoT 会在运动前被拒绝。兼容所需的旧字段只在
Normalizer 内部生成，不再消耗公开 Agent 输出。

### 14.2 双像素接口

- Qwen 检测 bbox 的底边中心作为可见最终目标的 `target_pixel`；
- A* 路径通常取约 3 m 前视可通行点，投影回同步三视图得到 `affordance_pixel`；
- 三视图各自使用 `[]` 或 `[x, y]`，每类像素最多填充一个 View；
- 投影检查当前深度可见性，并继续执行 5 m 和连通路段限制；
- 检测像素可以用于深度反投影，但不会再错误冒充 affordance pixel。

### 14.3 Slow 必须先于 Fast

Point Navigation 在 A* 规划和双像素生成后、第一步控制前发布 `pixel_guidance_ready`。Runtime 在该事件中
写入完整 `ABotSlowOutput`，随后 Fast 才能执行。Slow 输出、路径报告和 Fast 执行共享同一个
`slow_step_id` 与源 `observation_id`，可审计旧决策是否错误驱动了新轨迹。

Fast 过程不调用 Qwen：它只执行 Navmesh/A*、路径缓存、Habitat 原生单步动作、停滞/取消/超时检测和最多四张
轨迹关键帧记录。下一宏观边界再由 Self-Reflector 联合当前三视图、上一宏任务关键帧和 Topo-Memory 判断。

提交 `c3e49ad` 共修改 21 个文件，增加 2465 行、删除 227 行；完整回归为 **134 项通过**，compileall 与
`git diff --check` 均通过。详细协议见 `docs/abot_n0_n1_agentos_architecture.md`。

## 15. 当日启动与最终状态

从 `c3e49ad` 重启 Habitat、Web Relay 和 Hermes。Habitat 健康检查确认：

- `map_id=1st_floor_exh`；
- 固定语义骨干 8 个节点、8 条边；
- 语义图已载入且冻结；
- 锚点置信度阈值 0.80；
- 执行策略为 `anchor_chain`。

首次启动时 H800 SSH 在密钥交换前被远端关闭，本地 `18011` 暂不可用；稍后重试恢复，SSH 隧道成功监听
`127.0.0.1:18011`，模型列表返回 `qwen3.6-vl-35b`，关闭 thinking 的最小生成请求准确返回 `READY`。
现有 Hermes 窗口无需重启即可继续使用，且没有为恢复模型额外 reset Habitat。

当前测试入口：

- Habitat API：`http://127.0.0.1:8766`；
- Web Relay：`http://127.0.0.1:5916`；
- Qwen OpenAI-compatible API：`http://127.0.0.1:18011/v1`；
- Hermes：`bash scripts/start_hierarchical_agent.sh`。
