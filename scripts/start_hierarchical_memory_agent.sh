#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HIERARCHICAL_MEMORY_HERMES_HOME:-${PROJECT_ROOT}/runtime/hierarchical_memory_hermes_home}"
HERMES_BIN="${PROJECT_ROOT}/runtime/hermes_venv/bin/hermes"
TOOL_URL="${CEE_HIERARCHICAL_TOOL_URL:-http://127.0.0.1:8766}"

test -x "${HERMES_BIN}" || {
  echo "Hermes is not installed. Run: bash scripts/bootstrap_hermes.sh" >&2
  exit 1
}

mkdir -p "${HERMES_HOME}"
cp "${PROJECT_ROOT}/configs/hierarchical_memory_hermes.yaml" "${HERMES_HOME}/config.yaml"
cp "${PROJECT_ROOT}/configs/hierarchical_memory_SOUL.md" "${HERMES_HOME}/SOUL.md"
export HERMES_HOME
export HERMES_ENABLE_PROJECT_PLUGINS=1
export CEE_HIERARCHICAL_TOOL_URL="${TOOL_URL}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:18011/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"

if [[ "${CEE_HIERARCHICAL_RESET_ON_START:-1}" != "0" ]]; then
  echo "Resetting Habitat before starting hierarchical memory Hermes..."
  curl --fail --silent --show-error --max-time 60 -X POST "${TOOL_URL%/}/reset" -d '{}' >/dev/null
fi

MEMORY_ENABLED="$(curl --fail --silent --show-error --max-time 5 "${TOOL_URL%/}/health" | \
  "${PROJECT_ROOT}/runtime/hermes_venv/bin/python" -c 'import json,sys; print(str(json.load(sys.stdin).get("scene_memory_enabled", False)).lower())')"
if [[ "${MEMORY_ENABLED}" != "true" ]]; then
  echo "Scene memory is disabled. Start Habitat with: bash scripts/start_hierarchical_memory_habitat.sh" >&2
  exit 1
fi

TASK_ID="$(curl --fail --silent --show-error --max-time 5 "${TOOL_URL%/}/health" | \
  "${PROJECT_ROOT}/runtime/hermes_venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["task_id"])')"

cleanup_active_memory() {
  local current_task_id
  current_task_id="$(curl --silent --max-time 2 "${TOOL_URL%/}/health" | \
    "${PROJECT_ROOT}/runtime/hermes_venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin).get("task_id", ""))' \
    2>/dev/null || true)"
  if [[ -n "${TASK_ID}" && "${current_task_id}" == "${TASK_ID}" ]]; then
    curl --silent --max-time 5 -X POST "${TOOL_URL%/}/task/finish" \
      -H 'Content-Type: application/json' -d '{"outcome":"cancelled"}' >/dev/null || true
  fi
}
trap cleanup_active_memory EXIT

printf '\033]0;Hermes Qwen - Hierarchical Scene Memory\007'
cd "${PROJECT_ROOT}"
"${HERMES_BIN}" chat --toolsets cee-hierarchical --max-turns 40 --cli "$@"
