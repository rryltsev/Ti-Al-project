#!/bin/bash
#SBATCH --cpus-per-task=3
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
mlp=/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
sha256sum "$mlp"
"$py" target_check.py run --model fitted.mtp --mlp "$mlp" --workers 3
tar -czf target_traces.tar.gz target_trace_*.npz
sha256sum target_check.json target_traces.tar.gz
