#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd /home6/u3129/tial-fm-research/ti3al-pbe-phonons-20261109-01
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_USE_LEGACY_KERAS=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export GRACE_CUDA_ROOT=/home6/u3129/mlip-research/envs/grace_ft_a100/lib/python3.11/site-packages/nvidia/cuda_nvcc
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$GRACE_CUDA_ROOT"
export PATH="$GRACE_CUDA_ROOT/bin:$PATH"
export DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1
INDEX=$SLURM_ARRAY_TASK_ID
if [ "$INDEX" -lt 12 ]; then
  PY=/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python
else
  PY=/home6/u3129/mlip-research/envs/grace_ft_a100/bin/python
fi
"$PY" ti3al_force_worker.py "$INDEX"
