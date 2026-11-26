#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
sha256sum -c inputs.sha256
echo "704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661  fitted.mtp" | sha256sum -c -
test "$(id -un)" = u3129
hostname
"$py" run_design.py
