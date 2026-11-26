#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
bin=/home6/u3129/tial-pd-20261124-atomistic/atat-build-01/atat/src
sha256sum "$bin/genstr" "$bin/corrdump" > binaries.sha256
"$bin/genstr" -l=lat.in -n=12 -sig=14 > enum.out
"$bin/corrdump" -l=lat.in -2=1.90 -3=1.90 -clus -sig=14
"$bin/corrdump" -l=lat.in -c -s=enum.out -sig=14 > correlations.out
tar -czf enum.tar.gz enum.out lat.in clusters.out sym.out correlations.out binaries.sha256 job.sh
sha256sum enum.tar.gz
