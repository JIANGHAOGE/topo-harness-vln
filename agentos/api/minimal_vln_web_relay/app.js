const $ = (id) => document.getElementById(id);
const ACTIONS = {0: 'STOP', 1: 'FORWARD', 2: 'LEFT', 3: 'RIGHT'};

function media(path) {
  return path ? `/api/media?path=${encodeURIComponent(path)}&t=${Date.now()}` : '';
}

async function refresh() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    const observation = state.observation || {};
    const episode = state.episode || {};
    if (observation.rgb_refs?.left) $('left').src = media(observation.rgb_refs.left);
    if (observation.rgb_refs?.front) $('front').src = media(observation.rgb_refs.front);
    if (observation.rgb_refs?.right) $('right').src = media(observation.rgb_refs.right);
    if (observation.topdown_ref) $('topdown').src = media(observation.topdown_ref);
    $('episode').textContent = episode.episode_id || 'n/a';
    $('status').textContent = episode.status || 'n/a';
    $('actions').textContent = `${episode.actions_used || 0} / ${state.health?.action_budget || 500}`;
    $('last-action').textContent = ACTIONS[episode.last_action] || 'n/a';
    $('connection').textContent = 'LIVE';
    $('connection').classList.add('live');
  } catch (error) {
    $('connection').textContent = `OFFLINE · ${error.message}`;
    $('connection').classList.remove('live');
  } finally {
    window.setTimeout(refresh, 400);
  }
}

refresh();
