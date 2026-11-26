#!/bin/bash
#SBATCH --job-name=TiPD-atat64
#SBATCH --partition=nompi,apollo,debug
#SBATCH --account=imet-2026-2
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=384M
#SBATCH --time=00:05:00
set -eu -o pipefail
test "$(id -un)" = u3129
sha256sum -c inputs.sha256
bin=/home6/u3129/tial-pd-20261124-atomistic/atat-build-sqs-01/atat/src
sha256sum "$bin/mcsqs" "$bin/corrdump" > binaries.sha256
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python generate.py "$bin/mcsqs" "$bin/corrdump" lat.in clusters.full.out cells64
tar czf cells64.tar.gz cells64 binaries.sha256
sha256sum cells64.tar.gz
