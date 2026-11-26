#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
mlp=/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
g++ --version | head -1
g++ -std=c++17 -O3 -march=native engine_check.cpp -o engine_check
g++ -std=c++17 -O3 -march=native sgmc.cpp -o sgmc
./engine_check fitted.mtp
"$mlp" calculate_efs fitted.mtp engine_cases.cfg --output_filename=mlip_cases.cfg > mlip.log
"$py" engine_report.py
