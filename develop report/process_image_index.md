# Anchor Agent 过程图与错误截图索引

> 所有图片均复制到报告目录中，使用相对路径，避免原日志清理后失效。中间布局、失败截图和已回滚实验图均保留。

## 1. 初期 Habitat 与 Hermes 验证

### Habitat RGB-D 传感器拼图

![Habitat sensor sheet](assets/early_validation/habitat_sensor_sheet.png)

### Habitat 动作前后观测联系图

![Habitat contact sheet](assets/early_validation/habitat_smoke_contact_sheet.png)

### Hermes 读取 Habitat 观测验证

![Hermes observation contact sheet](assets/early_validation/hermes_observation_contact_sheet.png)

## 2. 分层锚点与 Habitat 原生单步执行

### 锚点标注

![Anchor](assets/workframe/habitat_native_step_20260813/anchor_front.jpg)

### 起点 Topdown

![Topdown start](assets/workframe/habitat_native_step_20260813/topdown_start.png)

### 覆盖完整子任务轨迹的四张关键帧

| frame 0 | frame 4 | frame 8 | frame 12 |
| --- | --- | --- | --- |
| ![](assets/workframe/habitat_native_step_20260813/keyframe_00.jpg) | ![](assets/workframe/habitat_native_step_20260813/keyframe_04.jpg) | ![](assets/workframe/habitat_native_step_20260813/keyframe_08.jpg) | ![](assets/workframe/habitat_native_step_20260813/keyframe_12.jpg) |

## 3. 自动后观测完整任务结果

| LEFT | FRONT | RIGHT |
| --- | --- | --- |
| ![](assets/workframe/auto_observe_benchmark_20260814/final_left.jpg) | ![](assets/workframe/auto_observe_benchmark_20260814/final_front.jpg) | ![](assets/workframe/auto_observe_benchmark_20260814/final_right.jpg) |

| 最终锚点 | 最终 Topdown |
| --- | --- |
| ![](assets/workframe/auto_observe_benchmark_20260814/final_anchor.jpg) | ![](assets/workframe/auto_observe_benchmark_20260814/final_topdown.png) |

## 4. Scene Memory v1 历史实验

> 状态：已回滚。图片用于保留当时任务停止位置和场景记忆实验结果。

| LEFT | FRONT | RIGHT | TOPDOWN |
| --- | --- | --- | --- |
| ![](assets/workframe/scene_memory_v1_20260814/final_left.jpg) | ![](assets/workframe/scene_memory_v1_20260814/final_front.jpg) | ![](assets/workframe/scene_memory_v1_20260814/final_right.jpg) | ![](assets/workframe/scene_memory_v1_20260814/final_topdown.png) |

## 5. Web Relay 布局迭代

### 工作帧中保留的三个主要版本

| 初版 | 放大第二行 | 填充版 |
| --- | --- | --- |
| ![](assets/workframe/web_relay_layout_20260813.png) | ![](assets/workframe/web_relay_layout_enlarged_20260813.png) | ![](assets/workframe/web_relay_layout_filled_20260813.png) |

### 全部中间布局

<details>
<summary>展开 11 张布局迭代图</summary>

#### Desktop 初版
![desktop](assets/web_layout_iterations/desktop.png)

#### 1366 x 768 初版
![desktop 1366](assets/web_layout_iterations/desktop_1366x768.png)

#### 1366 x 768 final
![desktop 1366 final](assets/web_layout_iterations/desktop_1366x768_final.png)

#### 1366 x 768 verified
![desktop 1366 verified](assets/web_layout_iterations/desktop_1366x768_verified.png)

#### Desktop final
![desktop final](assets/web_layout_iterations/desktop_final.png)

#### Enlarged 1366 x 768
![enlarged 1366](assets/web_layout_iterations/enlarged_1366x768.png)

#### Enlarged 1600 x 1000
![enlarged 1600](assets/web_layout_iterations/enlarged_1600x1000.png)

#### Enlarged without fullscreen
![no fullscreen](assets/web_layout_iterations/enlarged_no_fullscreen.png)

#### Filled 1366 x 768
![filled 1366](assets/web_layout_iterations/filled_1366x768.png)

#### Filled 2048 x 1200
![filled 2048](assets/web_layout_iterations/filled_2048x1200.png)

#### Narrow viewport
![narrow](assets/web_layout_iterations/narrow.png)

</details>

## 6. 主动视角与语义定位地图

### 主动左转后的三视图

| LEFT | FRONT | RIGHT |
| --- | --- | --- |
| ![](assets/workframe/portable_memory_habitat_smoke/left_after_turn.jpg) | ![](assets/workframe/portable_memory_habitat_smoke/front_after_turn.jpg) | ![](assets/workframe/portable_memory_habitat_smoke/right_after_turn.jpg) |

### 转向后语义 Topdown

![Semantic topdown](assets/workframe/portable_memory_habitat_smoke/semantic_topdown_after_turn.png)

### 第一版语义地点地图

![Live semantic map](assets/workframe/semantic_map_memory/live_topdown.png)

### 锚点和历史地点共享 Point Navigation 冒烟测试

![Shared point skill](assets/workframe/semantic_map_memory/shared_point_skill_smoke_20260814.png)

## 7. 关键帧位置注册与回访

| location-001 | location-002 | location-003 | location-004 |
| --- | --- | --- | --- |
| ![](assets/workframe/keyframe_pose_revisit_smoke/location-001.jpg) | ![](assets/workframe/keyframe_pose_revisit_smoke/location-002.jpg) | ![](assets/workframe/keyframe_pose_revisit_smoke/location-003.jpg) | ![](assets/workframe/keyframe_pose_revisit_smoke/location-004.jpg) |

### 返回历史观测位并恢复朝向

![Revisit result](assets/workframe/keyframe_pose_revisit_smoke/revisit-result.jpg)

## 8. 首帧全局锚点与稀疏路网

### 首帧全局锚点

![Initial global anchor](assets/workframe/sparse_route_global_anchor_smoke/initial-global-anchor.jpg)

### 向外行进形成地点

| location-002 | location-003 |
| --- | --- |
| ![](assets/workframe/sparse_route_global_anchor_smoke/outbound-location-002.jpg) | ![](assets/workframe/sparse_route_global_anchor_smoke/outbound-location-003.jpg) |

### 回到 location-001 并恢复 yaw

![Yaw restored](assets/workframe/sparse_route_global_anchor_smoke/revisit-location-001-yaw-restored.jpg)

### 路网可视化桌面版

![Route graph](assets/workframe/sparse_route_global_anchor_smoke/web-relay-route-graph.png)

### 局部 inset 和窄屏版

| Local inset | Mobile/narrow |
| --- | --- |
| ![](assets/workframe/sparse_route_global_anchor_smoke/web-relay-route-graph-inset.png) | ![](assets/workframe/sparse_route_global_anchor_smoke/web-relay-route-graph-mobile.png) |

## 9. 静态骨干拓扑原型

### 原 Navmesh 与提取骨架对比

![Backbone comparison](assets/workframe/topology_skeleton_prototype/backbone_topology_comparison.png)

### 完整地图叠加

![Backbone full](assets/workframe/topology_skeleton_prototype/backbone_topology_full.png)

节点数据：[`backbone_topology_nodes.json`](assets/workframe/topology_skeleton_prototype/backbone_topology_nodes.json)
可复现脚本：[`extract.py`](evidence/topology_skeleton_prototype/extract.py)

## 10. 在线骨干语义地图与 Node-first 导航

### Reset 后载入固定骨干

![Backbone after reset](assets/workframe/reset_backbone_web_relay/web_relay_after_reset.png)

### 完整节点探索与地点中心语义

| 语义骨干地图 | 节点多方向观测联系图 |
| --- | --- |
| ![](assets/workframe/semantic_backbone_full_exploration/semantic_backbone_map.jpg) | ![](assets/workframe/semantic_backbone_full_exploration/node_six_view_contact_sheet.jpg) |

### 当前可编辑语义节点

![Current semantic nodes](assets/workframe/current_semantic_nodes/current_semantic_nodes.png)

### Node-first bedroom 路由实测

| 任务启动前视图 | B003 bedroom 节点前视图 |
| --- | --- |
| ![](assets/workframe/semantic_route_live_20260818/initial_front.jpg) | ![](assets/workframe/semantic_route_live_20260818/bedroom_node_front.jpg) |

### bbox 坐标回归故障与修复

| 错误地落在冰箱 | 正确目标框与脚前地面点 |
| --- | --- |
| ![](assets/workframe/bbox_anchor_regression_20260818/before_wrong_point.jpg) | ![](assets/workframe/bbox_anchor_regression_20260818/after_bbox_ground_point.jpg) |

原始 RIGHT 图：[`source_right.jpg`](assets/workframe/bbox_anchor_regression_20260818/source_right.jpg)
原始三联图：[`input_triptych.jpg`](assets/workframe/bbox_anchor_regression_20260818/input_triptych.jpg)

### 骨干拓扑跨墙边修复

| 欧氏 MST 直线建边 | Skeleton 邻接与自由空间折线 |
| --- | --- |
| ![](assets/workframe/topology_edge_connectivity_fix_20260818/before_euclidean_mst.png) | ![](assets/workframe/topology_edge_connectivity_fix_20260818/after_skeleton_adjacency.png) |

左图保留了用户发现问题时的旧行为：任务内 `location-004 -> location-005` 对应固定 B002-B003，直线穿过
不可通行区域。右图来自真实 Habitat/Web Relay 的 `sparse-backbone-v3`：该假边已删除，其他弯曲边沿
`path_world_xy_m` 折线绘制。新版拓扑 JSON 位于
[`evidence/topology_edge_connectivity_fix_20260818/`](evidence/topology_edge_connectivity_fix_20260818/1st_floor_exh-63271126e56dc000b09a.json)。

## 11. 用户调试截图与对应问题

### Gemini Flash 免费层 429

![Gemini 429](assets/user_screenshots/20260813_093708_gemini_quota_429.png)

### AgentOS/Habitat 服务未启动

![Service unavailable](assets/user_screenshots/20260813_144814_navigation_service_unavailable.png)

### 低层工具和 observation ID 猜测造成循环

![Tool loop](assets/user_screenshots/20260813_162043_hierarchical_tool_loop.png)

### 用户提出 Web Relay 第二行直接填充

![Fill request](assets/user_screenshots/20260813_174724_web_layout_fill_request.png)

### 没有显式 STOP，但普通文本结束当前 Hermes 轮

![Premature exit](assets/user_screenshots/20260814_163913_premature_plain_text_exit.png)

### 强制工具实验中的 Unknown Tool 重试

![Unknown tool](assets/user_screenshots/20260814_170442_unknown_tool_retry.png)

### 单次输出达到 max tokens，终端只显示省略号

![Output length](assets/user_screenshots/20260814_170807_output_length_missing_trace.png)

### Qwen 锚点 grounding 未返回合法 JSON

![Anchor grounding 400](assets/user_screenshots/20260817_180706_anchor_grounding_http400.png)

### begin/advance 工具准备与本地工具错误

![Begin navigation tool error](assets/user_screenshots/20260817_182243_begin_navigation_tool_error.png)

### 模型知道历史地点，但 location ID 字段格式错误

![Known location ID error](assets/user_screenshots/20260818_092350_known_location_id_error.png)

### Agent 识别到 bed 但 grounding 收到泛化目标

![Bed semantic target lost](assets/workframe/bed_anchor_target_loss_20260818/obs-1202-generic-anchor.jpg)

## 12. 非图片证据

- [`auto_observe_task_benchmark_20260814.json`](evidence/benchmarks/auto_observe_task_benchmark_20260814.json)
- [`last_hierarchical_task_timing_20260814.json`](evidence/benchmarks/last_hierarchical_task_timing_20260814.json)
- [`bounded_visual_context_benchmark_20260814.json`](evidence/benchmarks/bounded_visual_context_benchmark_20260814.json)
- [`scene_memory_v1_experiment_20260814.json`](evidence/benchmarks/scene_memory_v1_experiment_20260814.json)
- [`scene_memory_failed_task_cleanup_20260814.json`](evidence/benchmarks/scene_memory_failed_task_cleanup_20260814.json)
- [`recent_task_context_latency_20260818.md`](evidence/recent_task_context_latency_20260818.md)
- [`daily_development_20260818.md`](evidence/daily_development_20260818.md)
- [`node_first_office_fridge_audit_20260818.md`](evidence/node_first_office_fridge_audit_20260818.md)
- [`1st_floor_exh-63271126e56dc000b09a.json`](evidence/topology_edge_connectivity_fix_20260818/1st_floor_exh-63271126e56dc000b09a.json)
- [`live_notes_snapshot_20260818.md`](evidence/live_notes_snapshot_20260818.md)
