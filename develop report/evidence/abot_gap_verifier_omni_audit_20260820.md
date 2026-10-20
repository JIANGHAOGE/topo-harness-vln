# ABot 差异、Verifier 与 Omni Skill 审计：2026-08-20

## 1. 审计范围

本次审计基于分支 `experiment/anchor-chain-node-cognition`、提交基点 `c3e49ad` 及其未提交实验改动，覆盖：

1. 当前分层 Agent 与 ABot-N0、ABot-N1、ABot-AgentOS 的功能映射；
2. Planner、Actor、Self-Reflector、Runtime Verifier 和 Finish Verifier 的真实调用边界；
3. 最近一次 `go to office to find a white fridge` 运行中的协议循环；
4. 长指令 `go out of the room, turn left and go straight to find the robot` 的可执行性；
5. 后续把 OmniNav 暴露为一个有界 Skill 的接入方案。

官方参考：

- ABot-N0：<https://arxiv.org/html/2602.11598>
- ABot-N1：<https://arxiv.org/html/2607.10383>
- ABot-AgentOS：<https://arxiv.org/html/2607.10350>
- ABot Navigation 官方项目：<https://github.com/amap-cvlab/ABot-Navigation>

## 2. 最近运行暴露的问题

2026-08-20 16:42 的完整任务最终成功，Finish Verifier 确认白色冰箱，机器人停在目标约 `0.28 m` 处。
但该轮用了 32 次 API 调用和 31 个 Tool turn，最终输入上下文约 23.2k tokens。主要浪费并非控制执行，
而是以下协议循环：

1. 模型前两次给出了有语义内容的 Planner，但使用 `planner_output.plan.subtasks` 等嵌套别名，兼容层未识别，
   Runtime 连续返回“first Slow decision must include planner_output”；
2. 模型在当前三视图中判断白色冰箱已可见，但 Runtime 仍要求先存在由目标锚点执行产生的
   `active_target_evidence`，连续拒绝 Self-Reflector 的完成判断；
3. 400 响应只保留了错误文本，丢失当前活动 subtask、最新 observation 和 Planner 策略，模型只能重复猜测；
4. `search_semantic_map` 仍在 Python 注册表中公开，存在绕开 Qwen 语义节点选择、退回确定性词项检索的入口；
5. 长指令 Schema 已允许 `Interaction -> Reaching`，但 Runtime 自动补入目的地区域 PointGoal 时仍会把它
   放到最前，可能把“先出门再左转”改写为“先去目标节点再出门”。

## 3. 本次修改清单

### 3.1 模型负责语义，Runtime 负责协议和安全

- 删除 Runtime 在 `no_progress`、`blocked` 或 `hypothesis_rejected` 后自动把模型动作改为回退地点的规则；
- 回访、视角调整、继续锚点或切换节点均由 Qwen Self-Reflector 与 Actor 联合选择；
- Object/POI 子任务不再依赖 Runtime 的硬编码“必须已有高置信锚点记录”才能完成；
- Self-Reflector 可根据当前三视图和轨迹证据提出语义完成，最终全局语义由独立 Qwen Finish Verifier 裁决；
- PointGoal 仍由 Runtime 使用 pose、节点坐标和到达半径验证，因为这是可测量的执行事实，不是语言理解。

### 3.2 完成状态机修复

- 活动 Agentic Plan 只有在最后一个 subtask 经 `advance_navigation` 的 Self-Reflector 完成迁移后，才能进入
  `verify_goal`；
- 在此之前直接调用 `finish_navigation_task(done)` 返回 `finish_not_ready`，不调用 Finish Verifier，也不清空记忆；
- `known_location` 的 `control_steps=0` 现在显式返回
  `motion_performed=false`、`completion_scope=location_arrival_only` 和
  `no_motion_reason=already_at_target_location`；
- 零步回访只证明机器人已经在记录地点，不能证明目标物或全局任务完成。

### 3.3 Planner 与长指令顺序修复

- 去除 `Approaching -> Reaching -> Interaction` 的固定排序约束，只保留 stage/task type 对应关系；
- 允许 ABot-N0 角色按指令交错或重复，例如
  `Interaction(exit/turn) -> Approaching(region) -> Reaching(robot)`；
- 自动补入最终 ObjectGoal 时追加到序列末尾，不再插入到 Interaction 之前；
- 自动补入目的地区域 PointGoal 时放在最终目标 Reaching 之前，保留前置路线指令；
- 只修订最终 Reaching 的目的地与目标，保留中间 POI/Reaching 的独立语义和节点；
- 不再把缺少节点的 Interaction 强行绑定到最终目标节点。

### 3.4 Hermes 协议容错与错误恢复

- 兼容模型常见的 `plan.subtasks`、`target_localization`、`active_target_evidence`、`target_object` 等结构别名；
- 兼容实际日志中的 `agentic_plan.planner_output`、`mode=known_location`、`known_node` 等包装；
- 兼容层只做字段映射、ID 提取和 Schema 补全，不负责选择目标节点或判断任务因果关系；
- Hermes 外层 Tool Schema 改为容错入口，只要求存在一个 action；严格 CoT、Planner、Reflection 和 Action
  校验仍在规范化后的 Runtime Pydantic 层执行，避免 bridge 在中间件运行前拒绝可修复字段；
- HTTP 400 保留 Runtime 返回的 `next_call_contract` 和 `active_target_evidence`，不再压缩成一行错误文本；
- 错误后模型可直接读取权威 observation ID、活动 subtask ID、所需 Self-Reflection 和 Planner 策略；
- `search_semantic_map` 从 Hermes 注册表移除，只保留内部诊断 API；Agent 只看到 begin、recall、advance、finish。

### 3.5 验证

- 新增嵌套 Planner 规范化测试；
- 新增 HTTP 400 保留下一调用合同测试；
- 新增 Hermes 仅注册四个任务级 Tool 测试；
- 新增长指令前置 Interaction 不被重排测试；
- 新增无锚点执行记录时仍可进入独立 Finish Verifier 的测试；
- 新增零步 known-location 执行范围测试；
- 最终完整测试：`159 passed in 12.24s`；
- `compileall` 与 `git diff --check` 通过。

### 3.6 修复后的第一轮 Habitat 实测

17:10 从相同起点再次执行白冰箱任务。该轮在目标证据硬门和错误合同修复已生效、但最终外层 Tool Schema
尚未重启载入时完成：

| 指标 | 16:42 基线 | 17:10 修复后首轮 |
| --- | ---: | ---: |
| API calls | 32 | 12 |
| Tool turns | 31 | 11 |
| 最终调用输入 tokens | 23,173 | 20,819 |
| 目标证据硬门重复 400 | 10 次 | 0 次 |
| 首轮 Planner/Schema 错误 | 2 次 | 2 次 |

首轮错误仍有一次 Hermes bridge 缺少 `cot`，以及一次 `planner_output` 被包装在 `agentic_plan` 内。两者已在
随后提交的外层 Schema/别名修复中覆盖。重启后 Hermes 的 deferred tools 描述预算从约 2,607 tokens 降到
约 1,256 tokens；下一次同任务需要验证首轮两次错误是否归零。

## 4. Verifier 当前真实实现

结论：**当前生产分层配置中的 Finish Verifier 会独立调用 Qwen 模型。**

服务启动时同一个 Qwen provider factory 分别注入语义节点排序器和 Finish Verifier。Finish Verifier 收到：

- 原始全局任务和当前 Agentic Plan；
- 最新 Slow decision；
- 可选的 sticky target evidence；
- 服务端任务记忆摘要；
- 最新同步 LEFT/FRONT/RIGHT 图像。

它必须输出严格 `FinishVerifierAssessment`：

- `evidence_satisfies_task`；
- `terminal_failure_supported`；
- `observed_evidence`。

Runtime 只把该模型结果转换为 `finish / replan / fail`，且授权成功必须包含具体观察证据。

| 判断层 | 当前实现 | 是否调用模型 | 边界 |
| --- | --- | --- | --- |
| Planner / node relation | Hermes Qwen + Qwen semantic node ranker | 是 | 任务拆解、场所-物体关系、目标区域 |
| Actor | Hermes Qwen | 是 | visible anchor / B or L location / viewpoint |
| Anchor grounding | 独立 Qwen vision call | 是 | bbox、视图、置信度 |
| Self-Reflector / Skill semantics | 下一次 Hermes Qwen Slow call | 是 | 上一宏任务是否完成、是否重规划 |
| Finish Verifier | 独立 Qwen provider call | 是 | 原始全局任务是否完成 |
| Runtime Verifier | Pydantic、时间戳、pose、路径和控制状态 | 否 | 协议新鲜度、到达、超时、碰撞、卡住 |

当前 Skill 语义验证虽然由模型完成，但与 Actor 共用一次 Slow 调用，并不是独立的第二个 Skill Verifier 模型。
这与 AgentOS 的职责一致，但隔离强度仍弱于“独立 Skill Runner + 独立 Skill Verifier”实现。

## 5. 哪些判断不应交给模型

“所有因果推理、状态理解和语义判断由模型完成”是正确方向，但不应扩展为所有状态都由模型猜测：

### 模型负责

- 指令分解和有序子指令游标；
- place、function、object、POI 与任务的关联；
- 上一宏动作是否对任务有语义进展；
- 目标是否可见、证据是否满足 find/locate/approach 等语义；
- 应继续锚点、回访、转向、节点导航还是未来 Omni Skill；
- 是否建议完成、失败或重规划。

### Runtime 确定执行

- episode/task/skill/observation ID 和时间戳新鲜度；
- JSON Schema、字段类型、deadline、cancel、timeout；
- pose 距离、节点到达、图连通性、A* 路径是否存在；
- depth/bbox 坐标转换、碰撞、持续 stall、控制安全预算；
- Skill STOP 的作用域和状态机迁移；
- Finish Verifier 是否真正授权。

这仍遵守“语言负责语义，结构化协议负责执行”。让模型判断过期 observation、碰撞或路径合法性会降低安全性，
也不符合 ABot-AgentOS Harness 的作用。

## 6. 与 ABot 系列的差异

| ABot 设计 | 当前对齐情况 | 仍有差异 |
| --- | --- | --- |
| N0 Agentic Planner / Actor / Self-Reflector | 已有严格 Schema 和宏边界 | 外部 Qwen，未使用 N0 训练模型 |
| N0 Episodic Memory | 上一宏任务最多 4 张前视关键帧，旧图按需召回 | 尚无学习型记忆检索和跨任务评测 |
| N0 四层 Topo-Memory | Block/Road/Function/Object-POI 已实现 | 当前仅单楼层、固定 8 节点，缺动态 Map-as-Memory 更新 |
| N0 多任务 Skill | 有 Point/Object/POI/Instruction/Person 类型 | POI、Person 和持续 Instruction Skill 尚未完整落地 |
| N1 task-specific CoT | Instruction 和 Object CoT 已严格化 | 由外部 Qwen生成，未复刻训练监督 |
| N1 target/affordance pixel | bbox 底部中心 + A* 前视投影 | 两个像素由分离模块生成，不是 Slow 一次联合预测 |
| N1 Fast System | A*/Navmesh + 单步控制，宏任务内不调 LLM | 不是 2B 视觉动作专家，也不输出学习型 5 个 SE(2) waypoint |
| N1 异步 Slow/Fast | Slow 仅宏任务前后调用，Fast 缓存路径 | 当前顺序执行，不是真正并发异步更新 |
| AgentOS Skill Runner | SkillContext、Registry、压缩 SkillResult 已有 | 分层路径没有独立 LLM Skill 子代理上下文 |
| AgentOS 三级 Verifier | runtime + model Self-Reflector + model Finish | Skill 语义 Verifier 与 Actor 共用 Qwen 调用 |
| AgentOS Memory / Trace | SQLite、视觉证据、拓扑和事件阶段已具备 | 缺长期实体/事件/来源图、混合检索和分布式完整 Trace |
| AgentOS embodiment portability | Habitat Adapter 与 ROS2 合同分离 | ROS2 Nav2/SLAM、控制租约和实机安全门尚未完成 |
| 训练与评测 | 训练外架构功能对齐 | N0/N1 数据、预训练、GRPO、学习型评测未复刻，且本项目明确不复刻 |

因此当前版本是 **ABot-N0/N1 + AgentOS 非训练功能架构的工程映射**，不是模型能力和运行时细节的完全等价实现。

## 7. 长自然语言指令能力判断

目标示例：

```text
go out of the room, turn left and go straight to find the robot
```

修复后 Planner 可以表达：

```text
Interaction: exit the current room, then turn left and go straight
Approaching: move to the model-selected connected regional node when useful
Reaching: find and verify the robot
```

当前 Actor 可以连续选择：门洞/走廊视觉锚点、90 度转向、连接节点导航和最终 robot target anchor，A* 与单步
控制执行每个宏动作。因此在门洞、左转后的走廊和 robot 均能被当前三视图逐段观测的场景中，系统已经可以尝试
完成该任务。

但现在还不能称为稳定支持：

1. 没有一个持续执行完整有序路线片段的 Instruction-Following Skill；
2. 每个锚点结束后重新调用 Qwen，`left` 的参考帧和子指令游标可能漂移；
3. 没有独立 person/robot tracker；
4. 还没有固定长指令测试集验证成功率和错误恢复。

结论是：**结构上已可表达，单场景可能成功，但长指令鲁棒性仍不足。**

## 8. Omni Skill 接入建议

OmniNav 不应重新成为全局 Agent，也不直接公开低层 step。它应成为 Actor 的第四种宏 Skill：

```text
Actor action = visible_anchor | known_location | viewpoint_adjustment | omninav_skill
```

建议选择规则：

- 已知区域重定位：B-node navigation；
- 当前可见一跳目标：visible anchor；
- 一次转向即可消除歧义：viewpoint adjustment；
- 历史证据更强：L-memory revisit；
- “出门、左转、直走”这类单像素或单节点不足以表达的有界路线片段：Omni Skill。

`SkillRequest` 应携带 episode/task/plan revision/skill run/observation/timestamp/deadline、当前有序子指令、
当前三视图和最多 4 张上一 Omni 轨迹关键帧。Omni 的逐步历史只留在 Skill Runner 局部上下文。

主 Agent 只接收压缩 `SkillResult`：

- local STOP、reached、start/end pose；
- 轨迹和动作摘要；
- 最多 4 张覆盖完整执行轨迹的关键帧；
- 观察到的任务证据；
- stall、path failure、timeout 等失败分类。

Omni STOP 只能结束该 Skill Run。下一次 Qwen Self-Reflector 判断有序路线片段发生了什么，再选择下一 Skill；
只有独立 Finish Verifier 可以结束全局任务。项目已有 `OmniNavSupervisor`、`PersistentOmniNavSkill`、四帧轨迹
记忆和 local-STOP 测试，后续主要工作是把它们接入当前 `advance_navigation` 的统一 Actor 协议。

## 9. 后续优先级

1. 用白冰箱任务重新实测本次协议循环修复，比较 API calls、400 次数和首动作时延；
2. 建立包含 exit/turn/go-straight/object 的固定长指令集；
3. 给 Instruction-Following 增加显式 `instruction_cursor` 和参考帧绑定；
4. 把 OmniNav 作为有界宏 Skill 接入 Actor，并沿用现有 Finish Verifier；
5. 将 Skill Self-Reflector 从 Actor 输出中进一步隔离成可选独立模型调用；
6. 完成 ROS2 NavigabilityGrid、Nav2 执行和实机安全门。
