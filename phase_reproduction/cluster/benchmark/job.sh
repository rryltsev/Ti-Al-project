#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
mlp=/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp
sha256sum -c inputs.sha256
echo "704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661  fitted.mtp" | sha256sum -c -
test "$(id -un)" = u3129
hostname
sha256sum "$mlp"
/usr/bin/time -v -o time.txt "$py" label_cell.py --cell Ti48Al16_candidate.cfg \
    --ratio 1 --supercell 2 2 3 --mesh 6 --mlp "$mlp" --model fitted.mtp --output result
tar -czf result.tar.gz result
sha256sum result.tar.gz
