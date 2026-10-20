#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HABITAT_ENV="${HABITAT_ENV:-/home/jianghaoge/miniconda3/envs/habitat-gs}"
export PYTHONPATH="${PROJECT_ROOT}:/home/jianghaoge/habitat-gs/repo/habitat-lab/habitat-lab${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${HABITAT_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${HABITAT_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${PROJECT_ROOT}"
exec "${HABITAT_ENV}/bin/python" -m agentos.api.minimal_vln_server \
  --host "${MINIMAL_VLN_HOST:-127.0.0.1}" \
  --port "${MINIMAL_VLN_PORT:-8770}" \
  --action-budget "${MINIMAL_VLN_ACTION_BUDGET:-500}" \
  "$@"
