#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd /home6/u3129/tial-fm-research/grace-2l-preparation-20261107-01
export CUDA_VISIBLE_DEVICES="" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TF_USE_LEGACY_KERAS=1
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache" GRACE_CACHE="$PWD/cache/grace"
/home6/u3129/mlip-research/envs/grace_ft_a100/bin/python grace_mask_preflight.py
