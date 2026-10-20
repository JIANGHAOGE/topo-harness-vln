#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AGENTOS_CONFIG="${AGENTOS_CONFIG:-${PROJECT_ROOT}/configs/default.yaml}"
export CEE_HIERARCHICAL_CONFIG="${AGENTOS_CONFIG}"
export CEE_HIERARCHICAL_PORT="${CEE_AGENTOS_PORT:-8765}"

exec bash "${PROJECT_ROOT}/scripts/start_hierarchical_habitat.sh" "$@"
