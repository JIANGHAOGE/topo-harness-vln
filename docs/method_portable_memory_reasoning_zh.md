# Method: 基于可迁移 Agent Memory 与结构化任务推理的分层具身导航

## 1. 问题定义

本文研究部分可观测环境中的长程语言导航。给定全局任务指令 \(g\)，机器人在宏观决策时刻
\(t\) 获得同步三视图图像 \(I_t=\{I_t^L,I_t^F,I_t^R\}\) 与带来源说明的位姿证据
\(p_t=(x_t,y_t,\theta_t,f_t,s_t,c_t)\)，其中 \(f_t\)、\(s_t\) 和 \(c_t\) 分别表示坐标系、
定位来源和置信度。系统需要选择一个有界导航动作，而不是由大模型逐帧输出速度控制量。

我们将控制过程分解为两个时间尺度：任务级 Agent 在每个宏观步骤进行一次语义推理，Skill Runtime
负责局部感知、路径规划与闭环执行。局部 Skill 的停止只表示当前子任务结束，不能直接终止全局任务。

## 2. 总体架构

系统由 Task Reasoner、Agent Memory、Protocol/Safety Skill Router 和 Embodiment Adapter 四部分组成：

```text
Global instruction + current observation + bounded MemoryView
                         |
                  Task Reasoner
                         |
             structured NavigationDecision
                         |
              Protocol/Safety Skill Router
        +----------------+----------------+
        |                |                |
 visible anchor    known location   viewpoint adjustment
        |                |                |
        +-------- Point Navigation -------+
                         |
        Habitat-GS / ROS2 Embodiment Adapter
```

语言模型只生成符合 Schema 的任务级决策。所有执行字段均由 Pydantic 校验，Router 再根据
`action_type` 确定性选择 Skill，从而将开放式语义推理与机器人执行协议分离。

## 3. 可迁移 Agent Memory

### 3.1 证据模型

记忆层不读取 Habitat 内部状态，而只消费 Embodiment Adapter 提供的通用观测协议。每条位姿证据均记录
`frame`、`source`、`map_id`、`confidence` 与时间戳；用于回访的三维点统一使用
`metric_position_xyz`。因此仿真中的 Habitat 定位可被 ROS2 的 odometry、SLAM 或融合定位直接替换，
而无需修改记忆算法。

### 3.2 空间语义图

Agent Memory 使用持久化图结构 \(\mathcal{M}_t=(\mathcal{V}_t,\mathcal{E}_t)\)。地点节点
\(v_i\in\mathcal{V}_t\) 包含：

- 地点 ID、位姿证据和可再次调用的导航点；
- 简短场景摘要与最多 12 个语义标签；
- 最近最多 4 条带机器人朝向的 LEFT/FRONT/RIGHT 语义观测；
- 访问次数、分支状态和最近观测 ID。

当新观测与同一 `map_id`、同一坐标系中的最近节点距离不超过 \(r_m=0.75\,\mathrm{m}\) 时，系统将其
视为重访并合并语义；否则创建新节点。机器人从一个节点移动到另一个节点时，添加
`traversed_to` 空间边。原始图保存在 SQLite 中，图节点只保存结构化证据和文件引用，不保存图像字节。

局部控制器在行进过程中输出不含图像的位姿路点，但这些密集控制点不再直接生成拓扑节点。Task
Reasoner 根据当前三视图标记两类语义决策点：位于宽阔区域内部并可向多个方向继续观测的
`open_space_center`，以及存在多个独立可通行延伸方向的 `junction`。实际通行在相邻语义决策点之间
建立空间边，因此长走廊不会按固定距离堆积节点。最多 4 张视觉关键帧仍保留精确位姿、朝向和可导航
坐标，作为独立视觉证据点按需回访，但不会自动进入拓扑路网。

### 3.3 任务工作记忆

空间记忆之外，系统为当前任务维护独立的工作状态：全局目标、目标证据、当前路线假设及置信度、
已否定分支、上一子任务评价、上一动作和修订号。每次结构化决策和地点更新都会递增修订号，便于追踪
记忆变化并拒绝过期观测。

任务开始的三视图只用于第一次决策，下一视觉包到达后即退出活跃上下文，不写入固定视觉锚点。原始
任务指令持续保存在结构化任务计划中；冻结语义骨干、当前定位和节点间连通关系共同构成长期空间参照。
任务执行中产生的轨迹关键帧仍按需保存和召回。

### 3.4 有界检索视图

完整记忆不会直接注入 LLM。系统构造固定大小的 `MemoryView`，包含当前地点和最多 \(K=5\) 个相关地点。
当前实现采用可复现的词项重叠与时间新近度排序：

\[
s(v_i,q)=0.7\frac{|T(q)\cap T(v_i)|}{\max(1,|T(q)|)}+0.15\frac{i+1}{|\mathcal{V}_t|}
+0.15\,\mathbb{I}_{\mathrm{topology}}(v_i),
\]

其中 \(T(\cdot)\) 为规范化词项集合，查询 \(q\) 由全局任务和目标证据组成。当前地点单独保留，其余
节点按 \(s\) 降序截断。工具每轮只返回本轮 `MemoryDelta` 和新的 `MemoryView`；历史消息中的旧
`MemoryView` 会被移除，而对应事实仍保存在服务端。

视觉记忆采用相同的“持久存储、有界检索”原则。每次完整导航子任务生成一个独立 `subtask_id`，并将
覆盖该次轨迹的最多 4 张前视关键帧连同精确 pose、yaw、时间戳、`location_id`、导航点和子任务目标
写入证据节点。最近子任务关键帧自动进入下一次推理；全局任务内更早的关键帧不重复注入上下文，但
保留文件引用。Agent 可按地点或语义检索，也可从完整任务轨迹中均匀采样最多 4 张概览帧。召回图片
在下一次导航结果到达后退出活跃窗口，因此视觉上下文大小与任务轮数无关，而历史证据仍可回查。

## 4. 结构化任务推理

在每个宏观步骤，Task Reasoner 联合使用：结构化全局任务、上一子任务结果、覆盖最近一次
执行轨迹的最多 4 帧关键帧、当前三视图，以及有界 `MemoryView`。其输出 `NavigationDecision` 包括：

1. **Previous assessment**：区分局部执行状态与其对全局任务的作用；
2. **Route hypothesis**：描述下一空间转移为何可能接近目标，并给出置信度；
3. **Candidates**：提出最多 3 个候选，为每个候选标注任务相关性、信息增益和重访惩罚；
4. **Selected action**：给出动作类型、目标、预期证据、成功条件和失败回退；
5. **Directional semantics**：以机器人当前朝向为基准，压缩三路视图中的显著语义。

Schema 强制选中动作必须与候选集合中的一项一致。该设计使模型在调用执行能力之前显式回答
“为什么这个局部目标有助于全局任务”和“执行后应看到什么”，用于增强锚点选择的因果关联性。

候选选择采用目标证据优先的分级策略：若当前视野存在与全局目标直接相关的锚点，则优先选择该锚点；
否则选择图像中可明确定位的可通行空间过渡，例如连续走廊地面、开放门洞或通向更宽阔视野的入口。
在多个空间过渡之间，优先预期信息增益更高且未被路网记忆重复覆盖的方向。仅当视野中没有可靠通路，
或一次有界转向可消除路线歧义时，才选择视角调整。该规则使未知环境探索保持前进性，同时避免将家具、
墙面或不可见房间猜测误作导航锚点。

每个动作还显式携带导航意图：`goal_pursuit` 表示沿直接目标证据前进，`explore_route` 表示目标不可见
时沿可通行空间主动探索，`backtrack` 表示返回历史目标证据点或拓扑决策点。当上一探索子任务无进展、
受阻或进入重复分支时，Reasoner 优先返回最近的岔路口或空旷区域中心，再选择另一条可见通路。该策略
把“看不见目标”视为正常探索状态，而不是立即停止或反复微调视角。
Runtime 对该边界只增加协议与安全保护：验证地点 ID、观测新鲜度、拓扑连通性、路径可达性、超时和
碰撞等客观条件。上一轮是否取得语义进展、失败原因是否成立，以及下一步应回访、调整视角还是继续
锚点探索，均由模型的 Self-Reflection 与 Actor 决定；Runtime 不用确定性文本规则改写模型动作。

## 5. 确定性 Skill 编排

Runtime 将结构化动作映射为三条固定执行路径：

| 动作类型 | Runtime 编排 |
| --- | --- |
| `visible_anchor` | `AnchorGenerationSkill -> PointNavigationSkill` |
| `known_location` | 从 Memory 解析导航点 -> `PointNavigationSkill` |
| `viewpoint_adjustment` | `ViewpointAdjustmentSkill` |

`AnchorGenerationSkill` 先由视觉模型标记可见锚点的地面接触像素，再利用同步深度将像素反投影为统一
世界坐标点。`PointNavigationSkill` 将目标吸附到可行走网格，计算 navmesh 最短路径，并在 Habitat 中用
greedy-follower 单步动作闭环执行。执行过程不调用任务级 LLM，并均匀保留起点、过程和终点共最多
4 帧轨迹关键帧。每个关键帧记录定位来源、坐标系、`map_id`、位置、偏航角、时间戳、定位置信度和
可执行导航点，并具有独立 `location_id`，但不因此成为拓扑节点。若任务级模型在执行后的关键帧中发现目标证据，
下一宏观步骤可直接选择该 `location_id`，无需在控制循环中增加视觉推理，即可通过
`PointNavigationSkill` 返回相应观测位置并用原生转向动作恢复记录朝向。已知地点回退与可见锚点导航共享同一 Point Navigation 接口，便于将 Habitat 执行器
替换为 ROS2 路径规划和移动接口。

在锚点生成与 Point Navigation 之间设置置信度安全门。若 Qwen 锚点置信度低于固定阈值
\(\tau_a=0.80\)，Runtime 丢弃该点，禁止其进入路径规划。Task Reasoner 在提出可见锚点时同步指定恢复
策略：执行一次有界视角调整以消除视觉歧义，或调用相同的 Point Navigation 返回稀疏路网中的上一
地点。若不存在有效上一节点，则确定性降级为视角调整。恢复结果与置信度会进入下一轮复盘，但恢复
执行本身不增加一次任务级模型调用。

## 6. 宏观闭环与上下文控制

一个任务遵循 `begin -> advance* -> finish` 协议。`begin` 只执行一次初始观测和记忆检索；每次
`advance` 完成以下闭环：

\[
\text{review}\rightarrow\text{hypothesize}\rightarrow\text{select}
\rightarrow\text{route}\rightarrow\text{execute}\rightarrow\text{observe}\rightarrow\text{update memory}.
\]

模型上下文固定保留结构化原始指令，并只保留最近一次移动的最多 4 张轨迹关键帧和最新三视图；任务
首帧在下一视觉包到达后删除。若随后仅调整视角，则保留上一移动轨迹并替换当前观测。更早图像被删除，
旧工具结果仅保留不含 `MemoryView` 的结构化摘要。因此视觉
上下文和检索记忆相对于任务轮数均为 \(O(1)\)，而不是把 \(t\) 轮历史累积为 \(O(t)\)。

每个动作仍携带 `observation_id`。Runtime 只接受最新观测，防止慢模型基于旧画面覆盖当前机器人状态。
局部 `reached` 只触发下一轮复盘；全局结束必须由任务级完成判断独立触发。

Web Relay 从相同的结构化记忆读取稀疏路网，在 topdown 地图上叠加地点、朝向和实际遍历边，并用局部
自适应视图展示短距离拓扑。该可视化不向任务模型增加图像或 token，也不依赖 Habitat 真值之外的新接口。

## 7. 实现与复现设置

当前实现使用 Qwen3.6-VL-35B 作为任务推理与锚点视觉模型，任务级工具输出上限为 768 tokens，关闭
额外 thinking 文本。三路相机分辨率为 \(640\times480\)，HFOV 为 \(120^\circ\)，VFOV 为
\(90^\circ\)。Agent Memory 使用 SQLite 持久化，默认数据库为
`runtime/agent_memory/hierarchical.sqlite3`。单元测试覆盖 Schema 校验、持久化恢复、40 个地点下的
固定大小检索、过期观测拒绝、结构化路由、关键帧位姿注册与回访、关键帧窗口和完整分层导航链路。

## 8. 当前边界

当前相关地点排序是确定性的词项检索，不包含学习式向量召回；跨任务空间记忆虽持久化保存，但默认
`MemoryView` 按当前任务隔离。全局完成仍由任务级 Agent 发起，后续可在不改变 Skill 协议的前提下增加
独立 Finish Verifier，并评估学习式检索、跨任务地图复用和 ROS2 定位不确定性对导航成功率的影响。
