#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CEE_AGENTOS_TOOL_URL="${CEE_AGENTOS_TOOL_URL:-http://127.0.0.1:8765}"
export HERMES_HOME="${HERMES_HOME:-${PROJECT_ROOT}/runtime/agentos_replica_hermes_home}"

exec bash "${PROJECT_ROOT}/scripts/start_hermes_qwen.sh" "$@"

