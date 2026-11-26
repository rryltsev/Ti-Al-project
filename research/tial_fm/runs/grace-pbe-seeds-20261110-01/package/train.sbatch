#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd /home6/u3129/tial-fm-research/grace-pbe-seeds-20261110-01
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache/$SLURM_JOB_ID" GRACE_CACHE="$PWD/cache/grace"
export GRACE_CUDA_ROOT=/home6/u3129/mlip-research/envs/grace_ft_a100/lib/python3.11/site-packages/nvidia/cuda_nvcc
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$GRACE_CUDA_ROOT"
export PATH="$GRACE_CUDA_ROOT/bin:$PATH"
RUN=/home6/u3129/tial-fm-research/grace-pbe-seeds-20261110-01
CASES=(PBE-full-N200-seed20 PBE-restricted-N200-seed20 PBE-restricted_mlp-N200-seed20 PBE-restricted_radial-N200-seed20 PBE-full-N200-seed30 PBE-restricted-N200-seed30 PBE-restricted_mlp-N200-seed30 PBE-restricted_radial-N200-seed30)
CASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}
/home6/u3129/mlip-research/envs/grace_ft_a100/bin/python -c 'import json; assert json.load(open("optimizer-gate.json"))["status"] == "PASS"'
/home6/u3129/mlip-research/envs/grace_ft_a100/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/train.log" 2>&1
/home6/u3129/mlip-research/envs/grace_ft_a100/bin/python "$RUN/grace_pilot_evaluate.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/evaluate.log" 2>&1
/home6/u3129/mlip-research/envs/grace_ft_a100/bin/python "$RUN/grace_pack_optimizer.py" "$CASE_NAME"
