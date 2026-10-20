#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HABITAT_ENV="${HABITAT_ENV:-/home/jianghaoge/miniconda3/envs/habitat-gs}"
export PYTHONPATH="${PROJECT_ROOT}:/home/jianghaoge/habitat-gs/repo/habitat-lab/habitat-lab${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${HABITAT_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${HABITAT_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export QWEN_API_KEY="${QWEN_API_KEY:-local-qwen}"
OMNINAV_URL="${OMNINAV_URL:-http://127.0.0.1:5808/eval}"

cd "${PROJECT_ROOT}"
exec "${HABITAT_ENV}/bin/python" -m agentos.api.hierarchical_navigation_server \
  --host "${CEE_HIERARCHICAL_HOST:-127.0.0.1}" \
  --port "${CEE_HIERARCHICAL_PORT:-8766}" \
  --agentos-config "${CEE_HIERARCHICAL_CONFIG:-${PROJECT_ROOT}/configs/default.yaml}" \
  --omninav-url "${OMNINAV_URL}" \
  "$@"
