#!/bin/bash
#SBATCH --cpus-per-task=8
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
mlp=/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
sha256sum "$mlp"
"$py" tail_check.py run --model fitted.mtp --mlp "$mlp" --blocks 8 --workers 8 --samples 8192
sha256sum tail_check.json tail_samples.npz
