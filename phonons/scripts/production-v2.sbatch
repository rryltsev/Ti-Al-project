#!/bin/bash
set -eo pipefail
umask 077
XC_NAME="$1"
case "$XC_NAME" in PBE|PBEsol|r2SCAN) ;; *) exit 2 ;; esac
cd "$SLURM_SUBMIT_DIR"
test -f setup.success
# Intel environment scripts modify shell positional arguments; preserve XC first.
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python workflow.py run "$XC_NAME"
