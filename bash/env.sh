#!/bin/bash
# Environment bootstrap. Sourced by sample.sh; do not execute directly.
#
# Override any variable below via shell env before sourcing. Examples:
#   CUDA_VISIBLE_DEVICES=1 source bash/env.sh
#   CKPT_ROOT=/path/to/checkpoints source bash/env.sh

# Activate the conda env (adjust CONDA_SH if conda lives elsewhere).
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "${CONDA_ENV:-hogdiff}"
fi

# Project root (resolve relative to this script).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CKPT_ROOT="${CKPT_ROOT:-$ROOT}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-123}"

# Let torch's bundled CUDA libs take precedence if present.
TORCH_LIB=$(python -c 'import torch, os; print(os.path.join(torch.__path__[0], "lib"))' 2>/dev/null)
if [ -n "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
fi

echo "[env.sh] env=${CONDA_DEFAULT_ENV:-<none>}  CUDA=$CUDA_VISIBLE_DEVICES  ROOT=$ROOT"
