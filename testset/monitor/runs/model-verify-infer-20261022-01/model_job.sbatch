#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
mkdir -p cache
export XDG_CACHE_HOME="$PWD/cache"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/inductor" TRITON_CACHE_DIR="$PWD/cache/triton"
export TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
hostname > node.txt
export CUDA_VISIBLE_DEVICES=""
python verify_exports.py /home6/u3129/tial-testsets/model-export-20261022-02-cpu
mv result.json verification.json
python infer_all.py all50-geometries.json
