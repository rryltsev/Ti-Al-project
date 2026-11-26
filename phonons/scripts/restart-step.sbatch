#!/bin/bash
set -eo pipefail
umask 077
XC_NAME="$1"
STEP_NAME="$2"
PARENT_JOB="$3"
cd "$SLURM_SUBMIT_DIR"
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLURM_KILL_BAD_EXIT=1
exec /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python restart_step.py "$XC_NAME" "$STEP_NAME" "$PARENT_JOB"
