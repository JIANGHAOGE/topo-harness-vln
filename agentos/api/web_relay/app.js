const $ = (id) => document.getElementById(id);
let lastImageKey = '';
const TARGET_FRAME_INTERVAL_MS = 50;

function mediaUrl(path) {
  return path ? `/api/media?path=${encodeURIComponent(path)}&t=${Date.now()}` : '';
}

function setImage(id, path, key) {
  const element = $(id);
  if (!path) {
    element.removeAttribute('src');
    element.dataset.key = '';
    element.classList.add('empty');
    return Promise.resolve();
  }
  if (element.dataset.key === key) return Promise.resolve();
  element.classList.remove('empty');
  return new Promise((resolve) => {
    const settled = () => {
      element.removeEventListener('load', settled);
      element.removeEventListener('error', settled);
      resolve();
    };
    element.addEventListener('load', settled);
    element.addEventListener('error', settled);
    element.src = mediaUrl(path);
    element.dataset.key = key;
  });
}

function findValue(value, names) {
  if (!value || typeof value !== 'object') return undefined;
  for (const name of names) if (value[name] !== undefined) return value[name];
  for (const child of Object.values(value)) {
    const found = findValue(child, names);
    if (found !== undefined) return found;
  }
}

function drawTrajectory(points) {
  const canvas = $('trajectory');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#3b4947';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(canvas.width / 2, 20); ctx.lineTo(canvas.width / 2, canvas.height - 24); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(24, canvas.height - 34); ctx.lineTo(canvas.width - 24, canvas.height - 34); ctx.stroke();
  if (!Array.isArray(points) || points.length === 0) return;
  const valid = points.filter((p) => Array.isArray(p) && p.length >= 2).map((p) => [Number(p[0]), Number(p[1])]);
  if (!valid.length) return;
  const extent = Math.max(1, ...valid.flatMap(([forward, left]) => [Math.abs(forward), Math.abs(left)]));
  const scale = Math.min((canvas.width - 60) / (2 * extent), (canvas.height - 60) / extent);
  const xy = ([forward, left]) => [canvas.width / 2 - left * scale, canvas.height - 34 - forward * scale];
  ctx.strokeStyle = '#4fd18b'; ctx.fillStyle = '#4fd18b'; ctx.lineWidth = 5; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath(); ctx.moveTo(canvas.width / 2, canvas.height - 34);
  valid.forEach((point) => { const [x, y] = xy(point); ctx.lineTo(x, y); });
  ctx.stroke();
  const [x, y] = xy(valid.at(-1)); ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.fill();
}

function drawRouteGraph(graph, observation) {
  const canvas = $('route-graph');
  const image = $('topdown-view');
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(width * pixelRatio));
  canvas.height = Math.max(1, Math.round(height * pixelRatio));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const nodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph?.edges) ? graph.edges : [];
  $('route-count').textContent = `${nodes.length} nodes · ${edges.length} edges`;
  const map = observation?.metadata?.topdown_map;
  const origin = map?.origin_world_xz_m;
  const metersPerPixel = Number(map?.meters_per_pixel);
  if (!nodes.length || !Array.isArray(origin) || origin.length !== 2 || !metersPerPixel) return;
  if (!image.naturalWidth || !image.naturalHeight || !width || !height) return;

  const fitScale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  const offsetX = (width - image.naturalWidth * fitScale) / 2;
  const offsetY = (height - image.naturalHeight * fitScale) / 2;
  const screenPoint = (node) => [
    offsetX + ((Number(node.x) - Number(origin[0])) / metersPerPixel) * fitScale,
    offsetY + ((Number(node.y) - Number(origin[1])) / metersPerPixel) * fitScale,
  ];
  const byId = new Map(nodes.map((node) => [node.location_id, node]));
  const nodeLabel = (node) => node.topology_node_id
    ? String(node.topology_node_id).replace('backbone-', 'B')
    : String(node.location_id).replace('location-', 'L');
  const nodeColor = (node) => {
    if (node.is_global_anchor) return '#e9b949';
    if (node.is_current) return '#4fd18b';
    const history = Array.isArray(node.goal_search) ? node.goal_search : [];
    const status = history.length ? history[history.length - 1].status : null;
    if (status === 'target_confirmed') return '#4fd18b';
    if (status === 'no_target_observed') return '#f28c52';
    if (node.visited) return '#e9b949';
    return '#62aef7';
  };

  ctx.strokeStyle = '#41d7c2';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  edges.forEach((edge) => {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (!source || !target) return;
    const path = Array.isArray(edge.path_world_xy_m) && edge.path_world_xy_m.length >= 2
      ? edge.path_world_xy_m.map(([x, y]) => screenPoint({x, y}))
      : [screenPoint(source), screenPoint(target)];
    ctx.beginPath();
    ctx.moveTo(path[0][0], path[0][1]);
    path.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.stroke();
  });

  ctx.font = '700 11px "Space Mono", monospace';
  nodes.forEach((node) => {
    const [x, y] = screenPoint(node);
    const color = nodeColor(node);
    ctx.fillStyle = '#08100d';
    ctx.strokeStyle = color;
    ctx.lineWidth = node.is_current || node.is_global_anchor ? 4 : 2;
    ctx.beginPath();
    ctx.arc(x, y, node.is_current || node.is_global_anchor ? 8 : 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    const yaw = Number(node.yaw);
    if (Number.isFinite(yaw)) {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - Math.sin(yaw) * 15, y - Math.cos(yaw) * 15);
      ctx.stroke();
    }
    ctx.fillStyle = color;
    ctx.fillText(nodeLabel(node), x + 10, y - 8);
  });

  const insetWidth = Math.min(190, width * 0.38);
  const insetHeight = Math.min(135, height * 0.38);
  const insetX = width - insetWidth - 10;
  const insetY = 10;
  const centerX = insetX + insetWidth / 2;
  const centerY = insetY + insetHeight / 2 + 8;
  const valuesX = nodes.map((node) => Number(node.x));
  const valuesY = nodes.map((node) => Number(node.y));
  const midpointX = (Math.min(...valuesX) + Math.max(...valuesX)) / 2;
  const midpointY = (Math.min(...valuesY) + Math.max(...valuesY)) / 2;
  const extent = Math.max(
    1,
    Math.max(...valuesX) - Math.min(...valuesX),
    Math.max(...valuesY) - Math.min(...valuesY),
  );
  const insetScale = Math.min(insetWidth - 42, insetHeight - 48) / extent;
  const insetPoint = (node) => [
    centerX + (Number(node.x) - midpointX) * insetScale,
    centerY - (Number(node.y) - midpointY) * insetScale,
  ];

  ctx.fillStyle = 'rgba(6, 14, 14, 0.9)';
  ctx.strokeStyle = '#425250';
  ctx.lineWidth = 1;
  ctx.fillRect(insetX, insetY, insetWidth, insetHeight);
  ctx.strokeRect(insetX, insetY, insetWidth, insetHeight);
  ctx.fillStyle = '#82918e';
  ctx.font = '700 9px "Space Mono", monospace';
  ctx.fillText('ROUTE NETWORK', insetX + 9, insetY + 15);
  ctx.strokeStyle = '#41d7c2';
  ctx.lineWidth = 2;
  edges.forEach((edge) => {
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (!source || !target) return;
    const path = Array.isArray(edge.path_world_xy_m) && edge.path_world_xy_m.length >= 2
      ? edge.path_world_xy_m.map(([x, y]) => insetPoint({x, y}))
      : [insetPoint(source), insetPoint(target)];
    ctx.beginPath();
    ctx.moveTo(path[0][0], path[0][1]);
    path.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.stroke();
  });
  nodes.forEach((node) => {
    const [x, y] = insetPoint(node);
    const color = nodeColor(node);
    ctx.fillStyle = '#08100d';
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, y, node.is_current || node.is_global_anchor ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = '700 9px "Space Mono", monospace';
    ctx.fillText(nodeLabel(node), x + 7, y - 5);
  });
}

async function render(state) {
  const observation = state.observation || {};
  const omni = state.omninav || null;
  const task = state.navigation_task || {status: 'idle'};
  const routeGraph = state.route_graph || {nodes: [], edges: []};
  const refs = observation.rgb_refs || {};
  const observationId = observation.trace?.observation_id || 'n/a';
  const imageKey = `${observationId}:${observation.trace?.frame_timestamp || ''}`;
  if (imageKey !== lastImageKey) {
    await Promise.all([
      setImage('left-view', refs.left, imageKey),
      setImage('front-view', refs.front, imageKey),
      setImage('right-view', refs.right, imageKey),
      setImage('topdown-view', observation.topdown_ref, `${imageKey}:topdown`),
    ]);
    lastImageKey = imageKey;
  }
  drawRouteGraph(routeGraph, observation);
  await setImage('anchor-view', task.anchor_annotation_ref, `anchor:${task.anchor_annotation_ref || ''}`);
  const executedRole = task.generated_point?.anchor_role || task.anchor_role;
  const visualTarget = executedRole === 'target' && task.generated_point?.target_label
    ? {
        label: task.generated_point.target_label,
        view: task.generated_point.target_view,
        confidence: task.generated_point.target_confidence,
      }
    : null;
  const groundedView = task.generated_point?.annotation_view
    || visualTarget?.view
    || task.generated_point?.view
    || task.anchor_view
    || task.view;
  const anchorSource = task.anchor_source_observation_id || task.generated_point?.source_observation_id;
  const anchorThreshold = Number(
    task.grounding_confidence_threshold ?? state.health?.anchor_confidence_threshold
  );
  const targetThreshold = Number(
    task.target_confidence_threshold ?? state.health?.target_confidence_threshold
  );
  const groundingConfidence = Number(visualTarget?.confidence ?? task.grounding_confidence);
  const groundingThreshold = visualTarget ? targetThreshold : anchorThreshold;
  const confidenceSummary = Number.isFinite(groundingConfidence)
    ? ` · CONF ${groundingConfidence.toFixed(2)}${Number.isFinite(groundingThreshold) ? ` / ${groundingThreshold.toFixed(2)}` : ''}`
    : '';
  $('anchor-caption').textContent = task.anchor_annotation_ref
    ? `${visualTarget ? 'VISUAL TARGET' : 'ANCHOR PIXEL'} · ${String(groundedView || 'unknown').toUpperCase()}${confidenceSummary} · ${anchorSource || 'unknown source'}`
    : 'VISUAL ANCHOR · NONE';
  $('observation-id').textContent = observationId;
  const pose = observation.pose;
  $('pose').textContent = pose ? `x=${Number(pose.x).toFixed(2)}, y=${Number(pose.y).toFixed(2)}, yaw=${Number(pose.yaw).toFixed(2)}` : 'n/a';
  $('instruction').textContent = task.instruction || task.anchor || omni?.instruction || 'No task dispatched yet';
  $('agent-phase').textContent = task.agent_phase || task.agent_state || task.phase || 'idle';
  $('subtask').textContent = task.active_subtask || (task.anchor ? `Navigate towards ${task.anchor}` : 'n/a');
  const visibleAnchor = visualTarget?.label || task.visible_anchor || task.generated_point?.anchor || task.anchor;
  const anchorView = groundedView;
  $('visible-anchor').textContent = visibleAnchor
    ? `${visibleAnchor} · ${String(anchorView || 'unknown').toUpperCase()}`
    : 'n/a';
  $('anchor-confidence').textContent = Number.isFinite(groundingConfidence)
    ? `${visualTarget ? 'TARGET' : 'AFFORDANCE'} ${groundingConfidence.toFixed(2)}${Number.isFinite(groundingThreshold) ? ` / ${groundingThreshold.toFixed(2)}` : ''}${task.anchor_rejected || (Number.isFinite(groundingThreshold) && groundingConfidence < groundingThreshold) ? ' · below gate' : ' · eligible'}`
    : 'n/a';
  const recovery = task.recovery;
  $('anchor-recovery').textContent = recovery?.executed
    ? `${recovery.executed} · ${recovery.target}`
    : 'n/a';
  $('agent-calls').textContent = String(task.agent_calls || 0);
  const keyframes = task.trajectory_keyframes || task.result?.trajectory_keyframes || task.trajectory_memory || [];
  $('memory-count').textContent = `${Array.isArray(keyframes) ? keyframes.length : 0} / 4`;
  const step = Number(task.step || 0);
  $('skill-state').textContent = task.status === 'running' ? `running · step ${step}` : task.status;
  const result = omni?.result || task.result || {};
  $('action').textContent = findValue(result, ['action']) || 'n/a';
  $('controller').textContent = findValue(result, ['controller']) || 'n/a';
  const seconds = findValue(result, ['model_infer_sec', 'inference_sec', 'infer_sec']);
  $('latency').textContent = Number.isFinite(Number(seconds)) ? `${(Number(seconds) * 1000).toFixed(0)} ms` : 'n/a';
  drawTrajectory(findValue(result, ['trajectory', 'model_trajectory', 'waypoints']) || []);
  $('model-label').textContent = state.health?.version_profile === 'hierarchical' ? 'Hierarchical Qwen + Habitat Steps' : 'OmniNav original';
  $('status-label').textContent = task.status === 'running'
    ? `Habitat live / Navigation running · step ${step}`
    : `Habitat live / Navigation ${task.status}`;
  document.querySelector('.status').classList.toggle('ready', Boolean(observation));
}

async function refresh() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await render(await response.json());
  } catch (error) {
    $('status-label').textContent = `Relay unavailable: ${error.message}`;
    document.querySelector('.status').classList.remove('ready');
  }
}

async function refreshLoop() {
  const startedAt = performance.now();
  await refresh();
  const delay = Math.max(0, TARGET_FRAME_INTERVAL_MS - (performance.now() - startedAt));
  window.setTimeout(refreshLoop, delay);
}

refreshLoop();
