#!/bin/bash
#SBATCH --cpus-per-task=6
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
mlp=/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
sha256sum "$mlp"
"$py" n16_ti.py --model fitted.mtp --mlp "$mlp" --workers 6 --replicas 4 --samples 1024
tar -czf n16_draws.tar.gz raw_*.npz
sha256sum n16_ti.json n16_draws.tar.gz
