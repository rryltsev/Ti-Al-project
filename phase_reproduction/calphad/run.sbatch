#!/bin/bash
#SBATCH --job-name=TiPD-Liang
#SBATCH --partition=nompi,apollo,debug
#SBATCH --account=imet-2026-2
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:20:00
set -eu -o pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
sha256sum -c inputs.sha256
PYTHONPATH="$PWD/packages" /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python liang.py Liang2024.tdb out
