#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OMNINAV_URL="${OMNINAV_URL:-http://127.0.0.1:5808/eval}"
AGENTOS_CONFIG="${AGENTOS_CONFIG:-${PROJECT_ROOT}/configs/default.yaml}"
export QWEN_API_KEY="${QWEN_API_KEY:-local-qwen}"

exec bash "${PROJECT_ROOT}/scripts/start_habitat_tool_server.sh" \
  --omninav-url "${OMNINAV_URL}" \
  --sensor-views left front right \
  --agentos-config "${AGENTOS_CONFIG}" \
  "$@"
