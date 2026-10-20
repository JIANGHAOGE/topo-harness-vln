# 2026-08-18 Anchor Agent 详细开发记录

> 范围：CEE-AgentOS 分层 Anchor Agent、可迁移 Memory、语义骨干地图、Node-first 路由、原子点位执行、
> 三视图锚点和 Web Relay。
> 记录原则：同时保留最终实现、诊断过程、被否定方案、回滚和待完成项。

## 1. 今日目标

今天的工作从“已有三视图锚点导航如何减少无效探索”继续向上推进，主要解决八类问题：

1. 固定地图节点、历史观察位置和视觉锚点的身份混用；
2. 任务先去哪个语义区域、再找哪个物体的层级不清楚；
3. 长点位执行中 64 步边界导致 Agent 被无意义重复调用；
4. 锚点语义和坐标在 Hermes -> Runtime -> Qwen grounding 链路中丢失或误解释；
5. planner 声明的视图与锚点实际所在的 LEFT/FRONT/RIGHT 视图不一致；
6. 骨干提取、缓存和语义记忆曾错误靠近 Habitat Adapter，不利于 ROS2 实机迁移；
7. reset、固定骨干加载、语义调查和 Web Relay 之间缺少一致的启动行为；
8. 骨干节点虽然来自 Navmesh，但边仍可能跨墙，导致拓扑关系和 Web 可视化不可信。

## 2. 工作阶段总览

| 阶段 | 主要工作 | 最终状态 |
| --- | --- | --- |
| 19 | 修复历史地点 ID 从自然语言描述中提取失败 | 已实现，Commit `4852c17` |
| 20 | 从完整 Navmesh 提取 medial skeleton、岔路和开阔中心原型 | 已完成离线原型 |
| 21 | 骨干图按 map_id 在线缓存、启动载入、可编辑节点语义 | 已实现并接入 Habitat |
| 21-A | Mapping Provider 从 Habitat Adapter 解耦、reset 即恢复骨干和 Web 显示 | 已实现并实测 |
| 21-B | 固定骨干全图巡检、54 张多方向图像和 8 节点语义注入 | 已完成并持久化 |
| 22 | destination node -> local object anchor 的 Node-first 层级 | 已实现第一版 |
| 22-A | 三视图锚点真实 View 归属、不可见拒绝和固定节点策略 | 已实现并实测 |
| 23 | bed 已识别但 `action.target` 丢失为泛化锚点 | 已修复 |
| 24 | 最近完成任务的上下文和延迟逐消息审计 | 已完成，确认 JSON 大于图片占用 |
| 25 | 64 步从 Agent 边界改为 Runtime 内部安全分块 | 已实现并实测 |
| 26 | RIGHT 视图坐标空间误判，改为 bbox 合同 | 已修复并重放验证 |
| 27 | office/white fridge 首轮节点选择审计 | 设计已确认，联合评分待实现 |
| 28 | 欧氏 MST 跨墙边改为 skeleton 邻接与折线路径 | 已实现并实测 |

## 3. 历史地点 ID 与固定节点 ID 分离

### 3.1 原错误

模型知道 sofa 出现在 `location-025`，但曾把“location-025 where the sofa was seen”之类完整描述写入
`location_id`，触发 Pydantic 正则错误。更深一层的问题是固定地图节点也曾通过任务内 `location-XXX` 暴露，
Agent 无法区分“永久地图节点”和“本轮关键帧观察位”。

### 3.2 最终协议

- `topology_node_id=backbone-XXX`：固定地图骨干节点，跨任务稳定；
- `location_id=location-XXX`：本轮任务产生的关键帧或历史观察位置；
- 两者复用同一个 Point Navigation 执行 Skill，但请求必须且只能携带其中一个 ID；
- normalizer 可以从自然语言 target 中提取合法 ID，但不会把描述本身写进 ID 字段；
- Hermes 输出隐藏固定节点的内部数据库 location handle，Web 只用 B/L 前缀展示对应身份。

### 3.3 L 证据位置与历史图像

每个轨迹关键帧都保存精确 pose、yaw、导航坐标和可用的 LEFT/FRONT/RIGHT 图片引用。它们不会作为普通路网
节点，也不会全部自动塞进 Hermes：自动宏观决策只带最近完整子任务的最多 4 张前视轨迹帧，完整三视图档案
留在机器人端 Memory 中，只有 `recall_navigation_memory(location_id=...)` 才按位置取回。Agent 可据此回到途中
看见目标时的 L 点并恢复朝向；切换到另一个 B 节点后，旧 L 点不再充当当前局部搜索中心。

Hermes 投影还会把 route graph 的内部 edge endpoint 重写为 B ID，并删除固定节点的内部 location handle。
因此 B 是环境级语义路由身份，L 只表示任务级视觉证据，不再因存储实现而混用。

## 4. 静态骨干拓扑：从原型到在线地图

### 4.1 原型提取

从 Habitat 完整 Navmesh 的 topdown free mask 开始：

1. 删除过小自由空间连通分量；
2. Zhang-Suen thinning 得到单像素 medial skeleton；
3. 根据骨架分支数聚类稳定岔路；
4. 根据 distance transform 局部极大值提取开阔区域中心；
5. 用最小净空、节点间距、聚类半径和最大节点数抑制密集普通路点。

`1st_floor_exh` 使用当前参数得到 8 个固定节点：5 个 `open_space_center` 和 3 个 `junction`。节点坐标采用
环境无关 `NavigabilityGrid`，Habitat 使用 world XZ，ROS2 OccupancyGrid 使用 world XY；核心提取器不依赖
Habitat API。ROS2 归一化和测试已经存在，真实 ROS2 地图订阅与生命周期接入仍待同事侧完成。

### 4.2 在线缓存与启动行为

- 缓存键包含 map_id、geometry revision、提取算法版本和参数；
- 第一次启动提取并保存，之后 reset 或新任务直接命中缓存；
- 每轮任务把全局起点作为临时节点附着到最近骨干节点；
- reset 清除任务记忆，不删除环境级固定骨干和语义 JSON；
- 运行中禁止任意增加固定骨干节点，任务产生的关键帧仍按 L ID 单独保存。

接入真实 `1st_floor_exh` 的第一次在线提取曾暴露参数回归：节点达到 24 个上限且全部被判为 junction。该结果
只保留为诊断，随后恢复离线原型验证过的 Zhang-Suen、12 px 稳定分支环和 3+5 节点参数，启动结果重新稳定为
8 个固定节点。之后又修复启动脚本先 `/reset` 导致 Web Relay 暂时显示 0 节点的问题：现在无需先发任务，reset
返回后就能看到固定骨干、任务起点及其附着边。

### 4.3 Mapping Provider 与实机边界

骨干提取和缓存最初短暂放在 `HabitatGSAdapter` 中，随后从 Embodiment 层移除。当前 Runtime 只依赖
`NavigabilityMapProvider -> NavigabilityGrid`：

- Habitat Provider 把 Navmesh 归一化为 free mask；
- ROS2 Provider 接收 OccupancyGrid/SLAM 栅格并转换到同一结构；
- extractor、cache、task-start overlay、Agent Memory 和 semantic overlay 在两端复用；
- cache key 同时考虑 map source、geometry revision、算法版本、参数，以及后续实机需要纳入的 robot radius 和
  safety margin；
- 地图几何变化使骨干缓存失效，语义增量则单独保存，不能掩盖墙体或连通性变化。

运行时普通停点、控制路点和视觉关键帧均不创建固定节点。曾实现的在线语义节点准入要求是：到达锚点、Qwen
grounding 置信度和 Agent 任务相关性都不低于 0.90；后续按用户要求进一步收紧为“当前禁止新建固定节点，
只向既有节点补语义”。

## 5. 节点语义地图与多轮人工校正

### 5.1 从方向标签改为地点认知

最初尝试按 LEFT/FRONT/RIGHT 固化每个节点语义，但这会把语义身份绑定到机器人某次 yaw。最终改为地点中心
描述：`place_type`、`function`、`objects`、`landmarks`、`spatial_character`、`task_affordances`、`aliases`、
`confidence`、`source` 和 `notes`。视图方向、位姿和图片继续作为证据，不再定义节点身份。

### 5.2 尝试与纠正过程

1. 临时把 B005 改为 bedroom，验证 JSON 热加载和检索链路后恢复；
2. 曾同时改写 bedroom/kitchen 节点的 function、objects、aliases 和 affordances；审计发现这会把人工地点判断
   扩散成未经视觉证明的内容，因此全部恢复到六视图调查原值；
3. 后续只允许人工修正 `place_type`，视觉 objects/landmarks 保持调查结果；
4. 中间经历过 B003 office、B005 bedroom 的错误对应，并通过原始 survey、实际导航和用户确认多次校正；
5. 当前最终值为 B001=`office area`、B003=`bedroom`、B005=`corridor`、B006=`kitchen`，其他字段保持原始
   多视图语义调查内容。

语义 JSON 可人工直接编辑，并可通过 `/semantic-map/reload` 热更新，不需要重启 Habitat。

### 5.3 固定骨干全图语义调查

为了避免只凭单一朝向给节点命名，实际遍历了 9 个路由点（任务起点 + 8 个固定骨干节点）和 11 段图连通路径。
每个节点在到达朝向采集一组三视图，再原地旋转约 180 度采集第二组三视图，共保留 54 张方向图像，并给 Qwen
输入带节点编号的六视图拼图。调查期间路网始终保持 9 节点，没有把控制轨迹和观察停点写回固定拓扑。

模型输出包括地点类型、检索文本、标签、搜索线索、置信度和不确定性；随后再转换为不依赖朝向的
`NodeSemantics`。完整结果保存在
`logs/semantic_backbone/20260818_full_exploration_r2/semantic_backbone.json` 和 `semantic_index.json`，后者包含
52 个检索词。过程图保留了语义骨干图和 9 节点六视图 contact sheet。

### 5.4 独立语义节点可视化

Web Relay 按用户要求只显示 B/L 编号；另行生成一张“地图节点 + 对应语义”审阅图。左侧是 topdown 节点位置，
右侧列出 topology ID、place type、confidence、function、spatial character、landmarks 和 task affordances。
这张图用于人工发现 place type 错配并修改 JSON，不把大段语义常驻 Web 页面。

## 6. ABot-N0 调研与 Node-first 层级

### 6.1 ABot-N0 对应关系

查阅 ABot-N0 技术报告后，将其 Block/Road/Function/Object 层次映射到当前系统：

- Road：固定骨干节点与连通边；
- Block/Function：bedroom、kitchen、office 等节点地点语义；
- Object/POI：bed、refrigerator、sofa 等最后一段视觉锚点；
- Self-Reflector：每个宏观 Skill 前后由 Agent 复盘，而不是逐帧调用 Agent。

第一版 Runtime 将任务拆成：

```text
global_goal + destination_query + object_goal
approach_node -> local_anchor_search -> verify_goal
```

并实现：

- 先到固定 B 节点，再在节点附近生成物体锚点；
- 节点局部搜索默认限制在 5 m；
- 像素深度超过 5 m 时沿原相机射线截断到 5 m；
- 去节点途中看见目标但终点丢失时，召回对应 L 关键帧，回到该 pose 并恢复 yaw；
- 切换 B 节点时清除旧 L 证据中心，防止跨语义区域继续使用旧局部范围。

### 6.2 三视图锚点身份与固定节点策略

在 bbox 坐标回归之前，还先修复过另一类独立问题：planner 缺失 `view` 时默认 FRONT，即使目标实际在 LEFT 或
RIGHT，也可能强迫 Qwen 在错误图片里找锚点。改动后 grounding 同时检查同一时刻的三路图像，由 Qwen 返回
真实 `panel_index`，Runtime 再映射到物理相机。`anchor_visible=false` 时返回零置信度且不生成标注，深度投影、
点位导航和 Web Relay 标题全部使用 Qwen 实际选择的视图，而非 planner hint。

同轮还把探索节点限制为固定骨干：`nearby_backbone_nodes` 给 Agent 提供邻近候选，`explore_route` 优先选择相邻
未探索 B 节点；到达后只把高置信任务语义写回既有节点。实测中 planner 请求 RIGHT，但三图 grounding 正确选择
FRONT、置信度 0.95，标注文件和 Web Relay 都显示 FRONT，且锚点执行后路网节点数没有增长。

## 7. Bed 可见但锚点仍错误

### 7.1 日志复现

在 `obs-1123` 和 `obs-1202`，Hermes 文本已经识别 bed，历史关键帧也包含 bed；但请求使用旧嵌套字段
`selected_anchor.semantic_hint`。normalizer 只提取 action type，没有把 bed 语义提升到平坦
`action.target`，最终默认成 `visible traversable anchor`。Qwen grounding 并非没有识别床，而是在正确执行
一个已经被协议改坏的泛化请求。

### 7.2 修复与一次撤回

- 提升 legacy semantic hint、description、view、expected evidence 和 success condition；
- Prompt 明确要求 `action.target` 保留精确物体身份；
- Runtime grounding 继续选择目标前方安全地面点，而不是物体表面。

期间尝试过“纯 find/locate 看见目标就强制结束”的规则。用户确认继续靠近是合理行为后，该规则被删除；
最终只保留语义目标传递修复，没有改变 find 任务的完成语义。

## 8. 上下文与延迟审计

最近 bedroom 找床任务最终一次模型输入 26,248 tokens，10 次调用累计输入 159,941 tokens。当前活动视觉窗口
是首帧三视图和 topdown、最近完整子任务 4 帧、当前三视图，共 11 张图，约占 3,369 tokens（12.8%）。
真正更大的来源是重复结构化文本：

- begin payload：约 8,378 tokens；
- route graph + MemoryView：约 5,477 tokens；
- 最近 decision/result：约 9,736 tokens；
- 完整 navigation result：约 6,012 tokens；
- Tool discovery/setup：约 2,986 tokens。

结论不是删除轨迹关键帧，而是继续保持“首帧 + 最近完整子任务 + 当前观测 + 按需召回”，并压缩重复
MemoryView、route graph、执行结果和工具 schema。

任务总耗时 71.80 s，其中 Habitat 控制 34.66 s，非控制 37.14 s；10 次 Agent 调用共 31.50 s，稳定复盘
平均 4.55 s。首次动作延迟 19.84 s 的主要异常是一轮 15.2 s 无效工具参数生成。

## 9. 64 步边界与原子 Point Navigation

原 `max_control_steps=64` 同时承担控制安全预算和 Agent 调用边界。长节点路径走满 64 步但没有故障时也会返回
Hermes，增加一次约 4-5 s 推理，并重复上下文。

最终将 64 步改为 Skill 内部安全分块：

- 正常耗尽：自动观测、刷新同一目标路径并继续，不返回 Agent；
- 返回 Agent：仅到达、持续卡住、路径失效、取消、总超时或未来高置信语义事件；
- 总安全限制：300 s、1000 动作和 Cancellation Token；
- 四张轨迹关键帧覆盖完整宏观任务，而不是每个 64 步分块。

真实 reset 起点到 B003 使用 `64 + 17` 两个内部块，共 81 动作、12.816 m、剩余 0.219 m、30.597 s，
第 64 步没有 Agent 调用，最终只返回一次 `completed/reached`。

## 10. RIGHT 视图 bbox 锚点回归

任务 `white humanoid robot standing near glass doors` 中，Qwen 正确选择 RIGHT 且置信度 0.95，但返回
`x=715,y=500,coordinate_space=pixels`。旧代码看到坐标超出 640 x 480 后，静默改按 `0..1000` 归一化解释，
把点换算到冰箱 `(457,240)`。

最终改为：

- LEFT/FRONT/RIGHT 三张独立带标签图片，不再依赖超宽拼图坐标；
- Qwen 固定返回所选单图内 `0..1000` bbox；
- Runtime 整理 bbox min/max，拒绝零面积框；
- 黄色框表示物体，红色十字表示 bbox 底边中心下方的可通行地面点。

同一帧重放得到 bbox `[240,220,265,268]` 和地面点 `(252,278)`，恢复到机器人脚前区域。

这次修复与第 6.2 节的 View 归属修复是两个不同层面：前者解决“应该用哪张相机图”，本节解决“选中该图后
坐标如何无歧义地解释”。两者共同保证锚点标签、图片、像素、深度和最终世界点使用同一观测链路。

## 11. Office / White Fridge 首轮决策审计

完成任务 `go to office to find a white fridge` 时，B001 以 0.992 匹配 office，B002 已包含
`refrigerator`、`tall white refrigerator` 和对应 affordance，而且与 B001 拓扑相邻。当前首轮路由却只检索
`destination_query=office`，没有把 `object_goal=white fridge` 加入节点排序，因此先走 B001、执行两次局部探索，
之后才切换 B002。

最初曾把 B002 解释为“离开 office 的错误节点漂移”；结合用户对地图语义的确认后撤回该解释。正确层级应是：

```text
place_node=B001
object_evidence_node=B002
execution_node=B002
```

即地点身份、物体证据节点和当前执行节点分开保存，联合使用 place、object 和 topology。该架构已经确认，
但联合评分尚未实现。英文任务还暴露出中文 `办公室` 无法命中英文 JSON、`fridge/refrigerator` 同义词缺失、
探索锚点过早进入 `verify_goal` 等后续问题。

该轮约 100.18 s，工具执行约 63.4 s，Agent/工具准备约 36.8 s；11 次模型请求累计输入 220,158 tokens。
详见 [`node_first_office_fridge_audit_20260818.md`](node_first_office_fridge_audit_20260818.md)。

## 12. 骨干边跨墙与 Skeleton 邻接

### 12.1 被否定的旧方案

节点来自 medial skeleton，但旧版本用欧氏 MST 建边。该方案只能保证同一大连通分量被一棵树连起来，无法
保证一条边表示局部通道邻接。B002-B003 欧氏距离 6.956 m，实际 skeleton 最短路 21.383 m，于是在 Web 上
出现 `location-004 -> location-005` 穿墙直线。

### 12.2 最终修复

- 算法升级到 `sparse-backbone-v3`；
- skeleton 上执行多源 Dijkstra；
- 只有两个节点最短路 Voronoi 区域在 skeleton 上相接时才建边；
- 距离保存为 skeleton geodesic；
- 回溯最短路并用自由空间可见性简化为 `path_world_xy_m`；
- Web 沿折线绘制；Hermes 投影删除折线，避免几何数组进入 Agent 上下文。

真实地图重算保持 8 个节点 ID 和位置不变，删除 B002-B003 假边，生成新缓存
`63271126e56dc000b09a`。运行时有 8 条固定骨干边和 1 条起点附着边。语义 JSON 只更新 cache key，所有
place/object/landmark 内容保持不变。

| 旧欧氏 MST | 新 Skeleton 邻接 |
| --- | --- |
| ![](../assets/workframe/topology_edge_connectivity_fix_20260818/before_euclidean_mst.png) | ![](../assets/workframe/topology_edge_connectivity_fix_20260818/after_skeleton_adjacency.png) |

## 13. 今日明确放弃或修正的思路

| 思路 | 处理 | 原因 |
| --- | --- | --- |
| 节点语义严格按 LEFT/FRONT/RIGHT 固化 | 放弃 | 身份会依赖观测 yaw，不利于多次巡检和实机迁移 |
| 人工修正 place 时同时改 objects/function | 撤回 | 会伪造未经视觉确认的场景事实 |
| 固定 B 节点继续暴露内部 L ID | 修复 | Agent 会混淆地图节点与历史关键帧位置 |
| 纯 find 看到目标后 Runtime 强制结束 | 撤回 | 用户接受继续接近，由 Finish Verifier/Agent 判断更合适 |
| 64 步直接返回 Agent | 替换 | 健康控制预算耗尽不是语义决策点 |
| 模型自行声明 pixel/normalized 坐标空间 | 替换 | 坐标越界后的猜测造成稳定错点 |
| Node-first 只按 destination place 排名 | 待替换 | 无法利用邻接节点中已知的目标物证据 |
| 同一自由空间分量内使用欧氏 MST | 替换 | 会隔墙连接并跳过真实通道结构 |
| 把 B002 完成解释为错误地点漂移 | 修正判断 | B002 是与 B001 关联的白冰箱证据节点 |

## 14. 验证、资产和运行状态

- 测试数从阶段初期的 105/108/109，经过协议和执行修复增至 111，最终为 **112 项全部通过**；
- 新增 U 形自由空间回归，验证隔墙近点不会直接建边；
- Habitat 真图重算确认 8 个节点 ID/位置不变；
- Web Relay Headless Chrome 截图确认折线沿可通行区域绘制；
- 新建并持续补充 `develop report/`，当前主时间线、过程图索引、今日详细记录、Node-first 审计和拓扑缓存证据
  均已归档；
- Habitat：`127.0.0.1:8766`；
- Web Relay：`127.0.0.1:5916`；
- Hermes：Qwen hierarchical 终端已重启；
- 当前骨干缓存：
  [`1st_floor_exh-63271126e56dc000b09a.json`](topology_edge_connectivity_fix_20260818/1st_floor_exh-63271126e56dc000b09a.json)。

## 15. 尚未完成

1. 实现 destination、object、topology 联合节点候选评分，并分别保存 place/evidence/execution node；
2. 增加中英文 canonical label 和 `fridge/refrigerator` 等同义词归一化；
3. 让 `explore_route` 锚点到达后仍处于 local search，只有目标锚点才进入 verify；
4. 继续压缩 MemoryView、route graph、navigation result 和 Tool schema 中的重复 JSON；
5. 完善独立 Finish Verifier；
6. 把 ROS2 OccupancyGrid Provider、Nav2/移动接口和安全门接入机器狗并进行同任务评测。
