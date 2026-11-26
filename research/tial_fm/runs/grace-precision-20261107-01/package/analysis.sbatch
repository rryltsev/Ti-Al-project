#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd /home6/u3129/tial-fm-research/grace-precision-20261107-01
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TF_USE_LEGACY_KERAS=1
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache/$SLURM_JOB_ID" GRACE_CACHE="$PWD/cache/grace"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=""
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python analyze_grace_precision.py --device "$1"
