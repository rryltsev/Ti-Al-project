#!/bin/bash
set -eo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd /home6/u3129/tial-fm-research/dpa4-replicates-full-20261106-02
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python aggregate_expansion.py --group "$1"
