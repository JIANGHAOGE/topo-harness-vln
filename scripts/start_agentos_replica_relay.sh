#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CEE_HIERARCHICAL_RELAY_PORT="${CEE_AGENTOS_RELAY_PORT:-5916}"
export CEE_HIERARCHICAL_TOOL_URL="${CEE_AGENTOS_TOOL_URL:-http://127.0.0.1:8765}"

exec bash "${PROJECT_ROOT}/scripts/start_hierarchical_relay.sh" "$@"
