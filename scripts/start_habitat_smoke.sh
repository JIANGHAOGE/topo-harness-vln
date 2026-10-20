#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/jianghaoge/CEE-AgentOS}"
CONDA_PREFIX="${HABITAT_CONDA_PREFIX:-/home/jianghaoge/miniconda3/envs/habitat-gs}"
CUDA_RUNTIME="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"

unset DISPLAY AMENT_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
export LD_LIBRARY_PATH="${CUDA_RUNTIME}:${CONDA_PREFIX}/lib"
export PYTHONPATH="${PROJECT_ROOT}:/home/jianghaoge/habitat-gs/repo/habitat-lab/habitat-lab"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"

cd "${PROJECT_ROOT}"
exec "${CONDA_PREFIX}/bin/python" scripts/run_habitat_smoke.py "$@"
