# CEE-AgentOS 导航 Agent 完整框架

## 1. 目标

CEE-AgentOS 的目标不是把 Code Agent 的提示词改成“你是机器人”，而是保留
Hermes 的通用 Agent/Harness 能力，再增加一个确定性的具身 AgentOS 控制平面。
以后新增 OmniNav、目标检测、建图、抓取、门控或人机交互能力时，只新增原子技能和
适配器，不修改主 Agent 循环。

ABot-AgentOS 官方目前公开的是系统设计和技术报告，而不是可逐行复刻的完整代码。
本项目对应其公开架构实现：Main Agent、隔离 Skill Runner、单一多阶段 Verifier、Context
Management、跨会话 Memory、端云 Router 和受门控 Skill Evolvement。

这里的“框架完整”是指主循环、协议、验证、记忆、安全边界和扩展点形成闭环，后续
增加一种原子能力只需实现并注册 Skill/Adapter；不是宣称所有导航模型、传感器和机器人
驱动都已经内置。

## 2. 分层与权威边界

| 层 | 负责人 | 不能做什么 |
| --- | --- | --- |
| Hermes task client | 用户会话、任务提交、状态查询和取消 | 不能承担具身规划、技能循环或完成判定 |
| AgentOS Main Agent | 场景条件计划、成功条件、技能选择 | 不能绕过能力目录调用不存在的技能 |
| Skill Runner | 隔离上下文、超时、取消、结果压缩 | 不能宣告全局任务完成 |
| Verifier | 一个组件，在运行时、技能级和终局三个阶段验证 | 不能以 RPC 成功或局部 STOP 代替物理证据 |
| Context/Memory | 有界上下文、带来源图记忆 | 不保存无限图像历史，不把旧记忆当当前事实 |
| Embodiment | Habitat/ROS2 执行、碰撞、看门狗、急停 | 不自行创造新的语义目标 |

唯一权威执行链为：

```text
Hermes 提交用户任务
  -> AgentOS 查询能力和当前状态
  -> 新鲜观测 + 小型记忆证据子图
  -> 场景条件计划
  -> 隔离技能执行
  -> 运行时验证
  -> 技能目标验证
  -> 全局终局验证
  -> DONE / RETRY / REPLAN / FAILED
```

## 3. 已实现的 AgentOS 基座

### 3.1 强类型协议

- `Observation`：具名相机、深度、topdown、pose 和 trace。
- `ObservationPolicy`：视图、模态、历史时域、帧数、token预算和新鲜度。
- `SpatialAnchor`：锚点类型、值、坐标系、来源观测、置信度和有效期。
- `NavigationIntent`：语义目标、模式、约束、成功条件、锚点和计划版本。
- `ExecutionReport`：真实命令、实际位移、转角、碰撞、数据年龄和停止来源。
- `SkillRequest/Result`、`VerificationResult`、`Plan`、`Event`：技能生命周期和审计协议。

### 3.2 能力注册与增量扩展

每个原子能力实现 `Skill`，并声明 `SkillDescriptor`：输入输出、所需模态、支持本体、
风险等级、超时和副作用。Hermes 通过 `cee_navigation_capabilities` 动态发现能力。
新增能力不需要向系统提示词硬编码工具名，也不需要修改状态机。

### 3.3 单一 Verifier 的三阶段验证和恢复

1. `VERIFY_RUNTIME` 检查结果新鲜度、执行状态和基础进展。
2. `VERIFY_SKILL` 检查当前子目标是否有足够语义或空间证据。
3. `VERIFY_FINISH` 回看原始任务和全部成功条件，决定全局完成。

验证器可返回 `CONTINUE/RETRY/REPLAN/FAIL/FINISH`。重规划有版本上限，旧版本异步
结果会被拒绝；局部 OmniNav STOP 只作为证据。

### 3.4 AgentOS 记忆层

记忆不等于 Hermes 的对话历史。当前实现使用 SQLite 图存储：

- 节点：source、evidence、entity、place、session、task、skill_run、semantic_event。
- 边：temporal、semantic、spatial、identity、provenance、contains、observed、located_at、supersedes 等。
- 每条具身事实必须有 `source_ref`；Hermes 写入事实时必须引用已经存在的
  `observation_id`。
- 原始图像保存在外部稳定路径，数据库只保存引用、摘要、pose 和来源。
- `BoundedContextManager` 每次只检索少量相关证据，不把完整历史塞入模型上下文。

当前检索是可验证的词法基线；后续可以在 `GraphMemoryStore` 后增加向量召回和局部图
扩展，而不修改运行时。

### 3.5 端云路由与技能演化

- `RuleBasedModelRouter` 根据时延预算、复杂度、隐私和可用性选择端侧或云端模型。
- 失败驱动改进只生成 JSON DSL `EvoAsset`，必须通过目标门和回归门才能发布。
- 每个资产有版本、父版本、触发失败、门控指标和活动版本，支持回滚；禁止 Agent
  直接生成 Python 并在机器人上执行。

### 3.6 OmniNav 原子技能

- `OmniNavSkill` 已实现正式 HTTP 适配，不再是占位类。
- Hermes 只通过 `cee_start_agent_task` 提交一次全局任务；Main Agent 在服务端完成
  场景条件规划、隔离技能执行、动作后新鲜观测、三级验证、恢复与记忆写入。
- 三视图严格按 `left, front, right` 组织，并映射为服务端既有的
  `image_left, image, image_right` multipart 字段；可选前视深度使用 `depth`。
- 服务端轨迹按 `[forward, left]` 米制原样保留，不在 AgentOS 内做符号、尺度或点数变换。
- 持久化 Skill Runner 在隔离上下文中连续执行有界动作，并返回压缩后的轨迹、
  `ExecutionReport` 和结束证据。OmniNav STOP 只是局部技能证据，仍不能替代全局终局验证。

## 4. Hermes 如何从 Code Agent 变成导航 Agent

`configs/navigation_SOUL.md` 是导航身份和不可突破的操作规则，启动脚本会把它安装到
隔离的 Hermes home。规则要求 Hermes：

- 移动前查询状态和能力目录；
- 动作前后都获取新鲜观测；
- 一次只调用有界动作或技能；
- 使用真实 pose、碰撞和视觉证据验证结果；
- 记忆只能辅助，当前观测优先；
- 不发布原始速度，不绕过端侧安全；
- 无进展、观测过期或证据不足时停止盲目重试并进入恢复。

这样转换发生在“身份 + 工具契约 + 确定性运行时”三处，而不是只靠 prompt。

## 5. 后续增加原子能力的固定步骤

1. 在 `agentos/skills/` 实现一个只完成单一职责的 `Skill`。
2. 声明 `SkillDescriptor`，明确模态、坐标系、副作用、风险和超时。
3. 输入使用 `NavigationIntent/SpatialAnchor` 或同等级强类型协议，禁止隐式数组语义。
4. 输出 `SkillResult` 和 `ExecutionReport`，预测与真实执行结果分开。
5. 为运行时、技能完成和任务完成分别增加验证用例。
6. 注册到 `SkillRegistry`，能力会由统一目录暴露给 Hermes。
7. 在 Habitat 做成功、失败、超时、碰撞和旧结果测试后，才接 ROS2。

## 6. 当前仍需逐步接入的能力

框架闭环已经存在，但以下是具体能力适配，不应伪装为已经完成：生产 LLM Planner、
语义感知/建图 Skill、完整 ROS2 命令租约与安全看门狗、向量记忆
检索、Web Relay 和标准评测。它们都可以按第5节逐项加入，AgentOS 主循环无需重写。
