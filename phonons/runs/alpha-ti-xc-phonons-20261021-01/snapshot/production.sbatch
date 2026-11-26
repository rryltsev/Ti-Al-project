#!/bin/bash
set -eo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
test -f setup.success
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python workflow.py run "$1"
