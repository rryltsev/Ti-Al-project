#!/bin/bash
set -eo pipefail
umask 077
XC_NAME="$1"
ARRAY_INDEX="$SLURM_ARRAY_TASK_ID"
case "$XC_NAME" in PBE|PBEsol|r2SCAN) ;; *) exit 2 ;; esac
cd /home6/u3129/tial-testsets/test50-dft-20261021-01
test -f preflight.json
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python workflow.py run "$XC_NAME" "$ARRAY_INDEX"
