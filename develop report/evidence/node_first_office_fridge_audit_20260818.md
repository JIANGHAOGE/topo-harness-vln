# Node-first Office / White Fridge 任务审计

> 日期：2026-08-18
> Hermes 会话：`20260818_171530_0ac6a4`
> Task ID：`hierarchical-task-20260818T091528526421`
> 用户任务：`go to office to find a white fridge`

## 1. 完整执行链路

| 阶段 | 决策或结果 | 关键数据 |
| --- | --- | --- |
| Task begin | 拆为 `destination_query=office`、`object_goal=white fridge` | B001 以 0.992 命中 `office area` |
| Node approach | Runtime 将首个可见走廊锚点改写为固定节点 B001 | 104 个 Habitat 动作、17.009 m、2 个内部执行块 |
| Local search 1 | 沿红色地毯探索 office 区域 | 2 个动作、0.2 m、grounding 0.95 |
| Local search 2 | 沿白色柜体和开阔地面继续探索 | 10 个动作、1.2 m、grounding 0.95 |
| Node switch | Agent 从语义图发现 B002 记录了白色冰箱 | 导航 B002，29 个动作、约 5.0 m |
| Object anchor | 当前视图明确出现 white refrigerator | grounding 1.00，15 个动作、0.8 m |
| Finish | Agent 提交 `outcome=done` | 最终观测 `hierarchical-obs-790` |

## 2. 为什么首轮没有直接选择 B002

语义地图在任务开始前已经具备以下信息：

- B001：`place_type=office area`；
- B002：`objects` 包含 `refrigerator`；
- B002：`landmarks` 包含 `tall white refrigerator`；
- B002：`task_affordances` 包含 `navigate to refrigerator`；
- 骨干图中 B001 与 B002 一跳相邻。

但 `_create_semantic_route_plan` 首轮只调用
`search_semantic_nodes(destination_query)`，因此只使用 `office` 排名。`object_goal=white fridge` 被写入计划，
却没有参与执行节点选择。Agent 到达 B001 后重新读取同一份语义图，才发现本可在任务开始时使用的 B002 信息。

这说明 Node-first 不应等价于“只按 place_type 选择一个节点”，而应进行任务条件化的联合检索：

```text
place_node = B001                 # office 身份
object_evidence_node = B002       # white refrigerator 证据
topology_relation = one hop       # B001 与 B002 邻接
initial_execution_node = B002     # 首轮可直接执行
```

逻辑上的地点节点与当前执行节点需要分别保存，避免把前往邻接目标证据节点误判为场所身份漂移。

## 3. 同时发现的边界问题

1. 当前英文 `office` 可以命中 B001，但中文 `办公室` 无法匹配英文语义 JSON，随后会退化为最近节点探索。
2. 当前词项搜索没有稳定处理 `fridge/refrigerator` 同义词。
3. 任意 `visible_anchor` 到达后都会把阶段切到 `verify_goal`，没有区分 `explore_route` 与真正目标锚点。
4. 联合检索需要保持结构化字段关系，避免把“white washing machine”和“black refrigerator”错误拼成
   “white refrigerator”。优先使用完整 landmark/object 短语，而不是无关系的词袋命中。

## 4. 延迟与上下文

- 用户指令到完成：约 100.18 s；
- 工具执行累计：约 63.4 s；
- Agent 推理、工具准备及协议往返：约 36.8 s；
- Hermes API 调用：11 次；
- Tool 调用：10 次；
- 累计模型输入：220,158 tokens；
- 累计模型输出：4,531 tokens；
- 平均每次模型请求输入约 20k tokens。

若首轮联合选择 B002，可以消除先到 B001 后的两次局部锚点探索及一次节点重规划，从而同时减少控制距离、
Agent 调用次数和重复结构化上下文。

## 5. 当前状态

本轮完成的是日志审计和目标架构确认。`destination + object + topology` 联合节点评分尚未写入 Runtime，仍属于
下一项实现任务；不能把 Agent 后续正确切到 B002 当作首轮联合规划已经完成。
