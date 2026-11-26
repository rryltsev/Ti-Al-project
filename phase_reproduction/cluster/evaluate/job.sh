#!/bin/bash
set -euo pipefail
MLP=/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
sha256sum -c inputs.sha256
test "$(id -un)" = u3129
hostname
sha256sum "$MLP"
echo "704f7fd900c7446286066b741084c30ebc7fcf26e5b9b8fd2f29a18ba4ffd661  fitted.mtp" | sha256sum -c -
cfg=( *.cfg )
test "${#cfg[@]}" = 1
"$MLP" calculate_efs fitted.mtp "${cfg[0]}" --output_filename=efs.result
if [[ -f efs.result.0 && ! -f efs.result ]]; then mv efs.result.0 efs.result; fi
test -s efs.result
test "$(grep -c BEGIN_CFG "${cfg[0]}")" = "$(grep -c END_CFG efs.result)"
test "$(grep -c BEGIN_CFG "${cfg[0]}")" = "$(grep -c Energy efs.result)"
sha256sum efs.result
