#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
py=/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python
mlp=/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp
sha256sum -c inputs.sha256
echo "704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661  fitted.mtp" | sha256sum -c -
test "$(id -un)" = u3129
sha256sum "$mlp"
hostname
mkdir results
while read -r structure ratio nx ny nz; do
    "$py" label_cell.py --prototype "$structure" --ratio "$ratio" \
        --supercell "$nx" "$ny" "$nz" --mlp "$mlp" --model fitted.mtp \
        --output "results/${structure}_${ratio}"
done < labels.txt
tar -czf results.tar.gz results
sha256sum results.tar.gz
