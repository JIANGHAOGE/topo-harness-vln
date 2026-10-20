#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERMES_HOME="${HIERARCHICAL_HERMES_HOME:-${PROJECT_ROOT}/runtime/hierarchical_hermes_home}"
HERMES_BIN="${PROJECT_ROOT}/runtime/hermes_venv/bin/hermes"
TOOL_URL="${CEE_AGENTOS_TOOL_URL:-${CEE_HIERARCHICAL_TOOL_URL:-http://127.0.0.1:8765}}"

test -x "${HERMES_BIN}" || {
  echo "Hermes is not installed. Run: bash scripts/bootstrap_hermes.sh" >&2
  exit 1
}

mkdir -p "${HERMES_HOME}"
cp "${PROJECT_ROOT}/configs/hermes_qwen.yaml" "${HERMES_HOME}/config.yaml"
cp "${PROJECT_ROOT}/configs/navigation_SOUL.md" "${HERMES_HOME}/SOUL.md"
export HERMES_HOME
export HERMES_ENABLE_PROJECT_PLUGINS=1
export CEE_AGENTOS_TOOL_URL="${TOOL_URL}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:18011/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local-qwen}"
RESET_ON_START="${CEE_HERMES_RESET_ON_START:-${CEE_HIERARCHICAL_RESET_ON_START:-1}}"
if [[ "${RESET_ON_START}" != "0" ]]; then
  echo "Resetting Habitat before starting AgentOS Hermes..."
  curl --fail --silent --show-error \
    --connect-timeout 3 \
    --max-time 60 \
    -X POST "${TOOL_URL%/}/reset" \
    -H "Content-Type: application/json" \
    -d '{}' \
    >/dev/null
  echo "Habitat reset complete."
fi

printf '\033]0;Hermes Qwen - AgentOS Navigation\007'
cd "${PROJECT_ROOT}"
exec "${HERMES_BIN}" chat --toolsets cee-navigation --max-turns 20 --cli "$@"
