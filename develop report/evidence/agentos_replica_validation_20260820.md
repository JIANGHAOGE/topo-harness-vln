# ABot-AgentOS 严格架构复刻与 Habitat 验证（2026-08-20）

## 1. 目标与版本边界

本轮在 `experiment/agentos-replica` 上工作。`release/cyhy-agentos-version1` 与 tag
`cyhy-agentos-version1` 继续固定在 `90d3339`，作为 Hermes 同时承担规划、工具循环和部分验证职责的稳定回滚版本。

“严格复刻”指严格遵循 ABot-AgentOS 公开技术报告的组件名称、职责边界和数据流。官方没有公开完整实现源码，
因此不声称逐行源码等价。低层 Fast System 暂由现有语义骨干、深度投影、A* 和 Habitat 单步控制工程替代；本轮
未启用 OmniNav。

## 2. 官方架构校正

依据公开报告，Agent Harness 的三个核心机制是：

1. Verification-aware ReAct；
2. Context Management；
3. Skill Evolvement。

在线执行中的三个角色是 Main LLM、Skill Runner 和 Verifier。Verifier 是一个组件，在 runtime、skill-level 和
finish-time 三个阶段工作，不是三个并列 Verifier。Hermes 在严格分支中仅保留任务提交、状态查询、取消和用户会话，
不再拥有具身计划、锚点选择、局部工具循环、记忆检索或全局完成权。

## 3. 本轮实现

### 3.1 Main Agent 与 Skill Runner

- 新增服务端 `HierarchicalMacroPlanner`，每个计划版本只产生一个结构化宏动作。
- 新增注册 Skill `HierarchicalNavigationSkill`，把 node navigation、visible-anchor、viewpoint adjustment、
  historical revisit、Qwen bbox、深度反投影、A* 和单步控制封装在隔离 Skill Runner 内。
- `SkillRunnerSession` 独立记录 subgoal、success conditions、attempt、source observation 和局部进度。
- Hermes 插件只公开 status、start task、task status、cancel 四个任务级工具。

### 3.2 上下文隔离

- 完整 `SkillResult.outputs` 只保留在原始结果、审计和 Memory Writer 路径。
- Main Agent 收到的结果会清空 `outputs`，只保留压缩摘要和 Evidence 引用。
- 分层服务共享状态由原始 `packet` 改为 bounded `handoff`：当前观测、Slow Context、语义路由、上一决策、
  执行指标和最多四张选中轨迹关键帧引用。
- Skill Verifier 输入也改为有界执行证据，不再序列化完整轨迹关键帧与控制 tick。
- 成功任务检查确认三个 `main_agent_results.outputs` 均为空。

### 3.3 单一多阶段 Verifier

- 对外统一为一个 `Verifier` 组件，内部组合 runtime、skill-level 和 finish-time 三阶段策略。
- Runtime 阶段检查执行健康、失败、超时和过期结果。
- Skill-level 阶段由 Qwen 联合 Skill 摘要、证据引用和最新三视图判断局部语义目标。
- Finish-time 阶段由 Qwen 对原始指令和最新证据授权全局完成；只有该阶段成功可进入 `DONE`。
- Verifier JSON 新增最多三次结构化自修复，并校验 status/action 合法组合。

### 3.4 Context、Memory 与 Skill Evolvement

- `BoundedContextManager` 继续负责当前观测、压缩结果、事件摘要和相关图证据的有界选择。
- SQLite 图记忆继续保存 typed/source-grounded task、observation、skill、evidence 和 verification 节点。
- 新增 `TraceDrivenSkillEvolvement`：每轮结束保存不含原始 outputs 的有界 trace，供离线
  diagnose -> propose -> compile -> gate -> promote 流程使用。
- `EvoAsset` 补充 target layer、triggering condition、permitted action、safety constraints 和 provenance；
  仍禁止在线生成并执行 Python，只有目标门与回归门均通过的版本化 JSON DSL 资产可以晋升。

### 3.5 兼容适配边界

旧分层工具内部要求 ABot-N0 `self_reflection`。严格 AgentOS 版本不再要求 Main Agent 重复输出一份与 Verifier
反馈重叠的反思；Skill Adapter 只把模型已经给出的 `previous_assessment` 结构映射为旧工具协议所需字段。
这一步不做场所、物体或节点语义判断。

## 4. Habitat 逐轮验证与修复

固定任务：`go to office to find a white fridge`。每轮均 reset 到同一起点，Qwen 使用
`qwen3.6-vl-35b`，AgentOS 服务端口为 `8765`。

1. 首轮在移动前失败：Qwen 将 `interaction` 错配为 `point_goal`，且 visible anchor 缺少 view。
   修复 Main Agent 结构化校验反馈。
2. 第二轮仍缺 visible-anchor view。补充 action 和 stage/task-type 协议不变量，并将修复上限增至三次。
3. 第三轮完成首段导航后，旧工具要求 `self_reflection`，导致下一 Skill 失败；同时终态 reason 曾错误沿用上一轮
   Finish Verifier 摘要。修复终态错误来源和反思适配。
4. 第四轮 Skill Verifier 没有返回完整 JSON。新增 Verifier 结构化重试与 status/action 一致性校验。
5. 第五轮暴露重复反思职责。移除 Main Agent 的强制 self-reflection，改由 Skill Adapter 映射 Verifier 驱动的
   previous assessment。
6. 最终轮成功：连通骨干约束把高层目标展开为 `backbone-004 -> backbone-001 -> backbone-002`，三个 Skill
   均 reached，Finish-time Verifier 从最新右视图确认白色冰箱并授权 `DONE`。

最终三段执行分别为：51 steps / 7.60 m、56 steps / 9.40 m、38 steps / 5.80 m。总任务耗时约 78.6 s，
其中移动约 40.2 s。首次计划 10.9 s；两次重规划分别约 5.6 s 和 5.3 s；三次 Skill Verifier 分别约
11.2 s、1.1 s 和 0.7 s；最终 Finish Verifier 约 0.5 s。

## 5. 与 CYHY AgentOS Version 1 对比

严格复刻分支的优势：

- Hermes 不再兼任 Main Agent、Skill Runner 和 Verifier，职责、上下文和完成权可以独立测试。
- 局部执行历史不再污染全局推理，主上下文增长由完整轨迹变为有界摘要与证据引用。
- Verifier 的纠错反馈显式回到 Main Agent，局部正常返回和全局成功被严格分开。
- Skill 通过 Registry/Descriptor 暴露，可在不改 Main Agent 的情况下替换 Habitat、ROS2 或后续 OmniNav。
- Event Trace、Memory Writer 和 Skill Evolvement trace 为失败归因、评测和回滚提供稳定接口。

Version 1 当前仍有的优势：

- Hermes 工具循环经过更多交互式测试，终端推理展示更直接。
- 单轮模型调用更少，初始时延通常低于独立 Skill/Finish 验证链。
- 对旧分层协议的容错积累更多，严格分支仍需扩大固定任务集验证。

严格分支的当前代价是额外 Verifier 调用带来的时延，以及公开报告未提供生产源码后必须自行实现的协议重试、
端云路由策略和 evo-asset 离线评测流程。

## 6. 验证与证据

- 完整测试：`167 passed in 11.40s`。
- 成功状态：`logs/agentos-replica-validation-20260820/final-success.json`。
- Skill Evolvement trace：`logs/agentos_evolution/traces/`。
- Habitat 运行目录：`logs/hierarchical/20260820_114039/`。
- Workframe 三视图与地图：`visuals/agentos_replica_20260820/`。

Habitat Runtime 和 Qwen tunnel 在报告完成时保持运行；OmniNav 未启动。
