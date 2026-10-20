# 2026-08-27 Anchor Agent 详细开发记录

> 日期：2026-08-27  
> 分支：`experiment/agentos-replica`  
> 稳定测试标签：`VLN-agentos-test-version-0827-stable-semantic-map-object-search`  
> 目标：稳定语义地图引导的地域选择、目标物搜索、执行验证和可观测链路。

## 1. 今日目标

今天主要解决的不是单个导航 Case，而是语义地图寻物任务中的四条通用链路：

1. Main Agent 如何从“地点 + 物体”指令形成稳定的地域搜索顺序；
2. NavigationSkill 的最新执行证据如何返回 Main Agent，而不泄漏逐帧内部历史；
3. Target 已经识别并到达后，Skill/Finish Verifier 如何避免因末帧视角变化误判失败；
4. Hermes 与 Web Relay 如何准确显示系统真正执行的目标、模块阶段和耗时。

用于持续回归的代表任务为：

```text
go to office to find a white refrigerator
```

该任务同时要求地点理解、语义节点排序、PointGoal 地域导航、ObjectGoal 视觉定位、Target 优先执行和全局完成验证，
因此适合作为当前版本的综合测试。

## 2. 稳定地域搜索队列与 Skill Handoff

### 2.1 发现的问题

早期 Main Agent 在每次 Skill 返回后会重新看到静态 Topo-Memory，但看不到 NavigationSkill 最新的压缩搜索状态。
结果是模型可能重新采用旧地域假设、重复探索已否定节点，或者让最新观测覆盖原始候选顺序。

### 2.2 修改

- 增加显式 `main_agent_handoff` 边界；
- NavigationSkill 只向 Main Agent 返回：
  - 压缩后的语义路由状态；
  - 确定性执行事实；
  - 当前任务 `goal_search`；
  - 语义完成状态；
  - 有界关键帧引用；
- 逐帧控制历史继续隔离在 Skill Runner 内，不进入 Main Agent 上下文；
- 初始 Qwen 排出的地域候选形成任务级稳定搜索队列；
- 每个候选只更新访问次数、搜索状态、证据和活动游标，不因一次新观测任意丢失；
- 使用 `no_target_observed`、`target_observed`、`target_confirmed`、`blocked` 表示当前任务对节点的搜索结果。

### 2.3 结果

Main Agent 现在能区分“地图先验认为这里可能有目标”和“本任务已到这里搜索但未发现目标”。这使系统具备按候选
逐个探索、失败后切换地域、发现目标后结束搜索的基础能力。

## 3. 语义节点 Ranker 的演进

### 3.1 Office 节点语义校准

将 `backbone-000`、`backbone-001`、`backbone-002` 的 `place_type` 统一为 `office`，只修改地点类型，不覆盖各节点
原有的 objects、landmarks、functions、affordances 和置信度。当前关键节点语义为：

| Node | Place type | 与冰箱任务相关的证据 |
| --- | --- | --- |
| `backbone-000` | office | laboratory/workshop、shelving、robot |
| `backbone-001` | office | open hall、red carpet、white cabinets |
| `backbone-002` | office | refrigerator、white cabinets、shelving units |
| `backbone-004` | living room | tall black refrigerator，但地点类型不匹配 office |

### 3.2 两阶段 Ranker 尝试及缺陷

第一版先让 Qwen 判断地点兼容性，再让第二次 Qwen 联合地点和物体排序。它在三个 office 并列时暴露出通用缺陷：
地点阶段随机选择 B000，合并代码又强制将 B000 放到首位，覆盖了联合阶段认为 B002 最合理的结果。

随后尝试保留全部地点兼容节点，再保持联合排序，但仍存在两次模型调用、两份置信度和合并语义重复的问题。

### 3.3 最终单阶段联合 Ranker

稳定版本改为一次 Qwen 联合排序。输入包括：

- 原始全局指令；
- Main Agent 拆出的 destination/object 关系；
- 完整候选节点的 place、object、landmark、function 和 affordance；
- 本任务的 visited/goal-search 状态；
- 当前定位和候选拓扑关系。

Qwen 一次性输出有序 `RegionSearchPlan`、每个候选的语义理由和置信度。Runtime 只验证节点 ID、Schema 和活动节点
一致性，不用固定关键词公式决定语义赢家。

实测结果：

```text
go to office to find a fridge
B002(0.98) -> B000(0.85) -> B004(0.75) -> B001(0.65)
```

B002 同时满足 office 与 refrigerator；B004 虽有 refrigerator 证据，但因 living room 地点冲突被降级。

## 4. 模型输入合同审计

新增 [`docs/model_input_contracts.md`](../../docs/model_input_contracts.md)，逐模块记录：

- 输入文本字段；
- 图像附件及相机归属；
- 结构化输出 Schema；
- 下游消费者；
- 谁拥有语义决策权；
- 哪类错误应回到哪个模块排查。

重点确认：

1. Main Agent、Slow Brain、Skill Verifier 和 Finish Verifier主要消费缓存的结构化三视图审计；
2. Slow 额外接收全局任务、当前委托、上一决策、有界历史关键帧和 Topo-Memory；
3. VisualGrounding 只接收当前单帧、视觉目标描述、排除标签和可选相机，不接收 Topo-Memory、CoT 或任务级
   `spatial_relation`；
4. MemoryWriter 当前不调用 Qwen，高置信 Grounding 事实按来源确定性写入长期图记忆；
5. Skill Verifier 与 Finish Verifier 是两次独立判断，但共享当前三视图审计，不能将二者一致误解为两套独立检测器。

## 5. Hermes 参数协议修复

### 5.1 原因

`cee_start_agent_task` 是 deferred tool，Hermes 首次只看到简短描述时会偶尔猜测 `task`，而服务端要求
`instruction`，因此每轮首次提交常出现参数名错误再重试。

### 5.2 修改

- Tool 描述和 SOUL 明确给出标准负载：`{"instruction": "<完整用户指令>"}`；
- Hermes 边界兼容 `task`、`task_description`、`global_goal`；
- 所有别名在进入 AgentOS 前统一规范化为 `instruction`；
- 缺失指令时在本地直接返回清晰错误，不发送无效 HTTP 请求。

## 6. Target 到达后的 Skill/Finish 验证闭环

### 6.1 失败根因

冰箱已在源观测中以 0.95 置信度定位，Fast 也执行 Target 并返回 reached；但机器人接近后，冰箱可能只剩侧板、过近
或离开最新三视图。Skill Verifier 使用源 Target 证据判定成功，Finish Verifier 却只看末帧并判定“当前看不到冰箱”，
触发重复探索。

### 6.2 修复

- Finish Verifier 可接收一张与当前 ObjectGoal 严格绑定的源 Target artifact；
- 只有 `target confidence >= threshold`、执行角色确为 Target、控制结果 reached 且任务语义一致时才允许引用；
- 普通历史图片、旧计划文本和未执行候选不能成为完成证据；
- Verifier 联合检查源 Target、确定性执行轨迹和 post observation；
- 当结构化完成事实满足而模型首次否定时，执行一次冲突复核，不由 Runtime 直接宣布语义成功。

### 6.3 RegionSearchPlan 协议修复

稳定候选队列允许活动游标从 B000 推进到 B002，但旧 Pydantic 约束同时要求 active 必须永远是列表第一项，导致
任务在已找到 Target 后因 Schema 冲突失败。当前约束改为 active 必须属于稳定候选队列，不再强制首位。

## 7. Target/Affordance 执行与可视化权威

### 7.1 发现的问题

当 Target 和 Affordance 落在同一个冰箱框时，Web Relay 先画 Target、后画 Affordance，最终可见标签变成
AFFORDANCE；另一路逻辑又根据“是否存在 Target 候选”推断执行角色，而不是读取 Fast 真正消费的角色。

### 7.2 修改

- Fast 的 `anchor_role + target_pixel` 成为 Web Relay 的唯一实际执行权威；
- 高置信 Target 执行时优先显示 Target；
- Target/Affordance 像素接近或 bbox 包含时，不再用次级 Affordance 覆盖 Target；
- 两者确实位于不同区域时仍可同时保留候选信息；
- AgentOS task status 新增结构化 `execution_summary`，包含：
  - 实际执行角色；
  - label、view、confidence；
  - 当前 Grounding 证据；
  - Slow 判断摘要和选择理由；
  - state、goal_reached、progress、termination reason；
  - source/post observation ID；
  - Slow、View Binding、Grounding、Fast 时延。

## 8. Main Agent 任务边界修复

运行中曾出现 `Main Agent lost the destination/object boundary`。根因是 Main Agent 已完成任务拆解后，Planner 又调用
第二次 Qwen 重新拆解，并要求两次自由语言结果字符串完全一致。语义等价但措辞不同也会被 Runtime 判为失败。

稳定版本移除冗余的第二次拆解审计：

- Main Agent 输出是 destination/object 边界的唯一任务级权威；
- 单阶段 Region Ranker仍使用原始指令和完整节点语义校验地域选择；
- Runtime 只做结构、字段和执行安全验证，不比较两份自然语言字符串。

## 9. Hermes 可观测性

上一稳定日志版本提供：

- Main Agent 完整任务计划；
- Skill Runner 启动和执行结果；
- Slow Brain、View Binding、Visual Grounding、Fast System 阶段；
- Runtime、Skill、Finish Verifier；
- 每阶段与累计耗时；
- 当前步骤、下一动作和最终结果。

黄色用于模块标题和进行中状态，正文为白色，成功为绿色，失败为红色。通用 Hermes reasoning 与默认 tool progress
保持关闭，减少模型自由叙述。

今天最后尝试过“只显示整块黄色步骤总结、隐藏全部中间阶段、移除返回 timeline”。用户测试认为可读性不如上一版，
该实验已完整回退，不进入本稳定标签。稳定版本继续保留分阶段结构化日志和紧凑 timeline。

## 10. Habitat 回归与证据

关键成功回归：

| 项目 | 结果 |
| --- | --- |
| Task ID | `agent-task-804f5b2b25e541899b5bbfd2ab07af38` |
| 指令 | office + white refrigerator 寻物任务 |
| 状态 | `completed / DONE` |
| Skill 步数 | 3 |
| 总移动距离 | 21.804 m |
| Target | white refrigerator，confidence 0.95 |
| Finish Verifier | success |

状态文件：
[`runtime/agent-task-804f5b2b25e541899b5bbfd2ab07af38_final_status.json`](../../runtime/agent-task-804f5b2b25e541899b5bbfd2ab07af38_final_status.json)

服务日志：
[`runtime/agentos_replica_habitat_20260827-123704_target_verifier_fix.log`](../../runtime/agentos_replica_habitat_20260827-123704_target_verifier_fix.log)

## 11. 稳定版本边界

`VLN-agentos-test-version-0827-stable-semantic-map-object-search` 包含：

1. AgentOS Main Agent / isolated Skill Runner / Runtime-Skill-Finish Verifier 分层；
2. 固定语义骨干与任务级地域搜索状态；
3. 单阶段 Qwen 地点-物体联合 Ranker；
4. Mapping -> PointGoal -> ObjectGoal 的复合寻物计划；
5. N1 风格 Slow + current-view Grounding + Fast 规划控制；
6. Target 优先执行与实际执行目标可视化；
7. 源 Target artifact 到 Finish Verifier 的受控证据链；
8. 长期图记忆、会话隔离和有界 Main-Agent handoff；
9. 上一版分阶段 Hermes 结构化日志。

该标签不宣称：

- 已接入专用检测/分割模型；
- 已完成实机 ROS2 安全门；
- OmniNav 是所有任务的默认执行器；
- 当前 Qwen bbox 精度等价于专用视觉 Grounding 模型。

## 12. 验证

- Hermes 插件回退测试：`9 passed`；
- 完整测试：`271 passed in 14.64s`；
- `git diff --check`：通过；
- AgentOS/Habitat：`127.0.0.1:8765`；
- Web Relay：`http://127.0.0.1:5916`；
- Hermes：单客户端运行。
