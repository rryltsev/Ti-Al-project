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
"$py" refine_n16.py --model fitted.mtp --mlp "$mlp" --workers 8 --draws-per-group 8192
tar -czf n16_refined_draws.tar.gz raw_k04_0-1-4-5.npz raw_k04_0-2-4-6.npz
sha256sum n16_refined.json n16_refined_draws.tar.gz
