#!/bin/bash
set -eo pipefail
umask 077
cd /home6/u3129/tial-testsets/test50-dft-20261021-01
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home6/u3129/mlip-research/envs/kplib_111/bin/python workflow.py prepare
