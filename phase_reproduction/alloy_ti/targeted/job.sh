#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
mlp=/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
sha256sum "$mlp"
"$py" targeted_cage.py run --model fitted.mtp --mlp "$mlp" --workers 12 --draws 8192
tar -czf targeted_draws.tar.gz gaussian_k*.npz student_k*.npz
sha256sum targeted_blocks.json targeted_draws.tar.gz
