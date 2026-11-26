#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python sweep.py
