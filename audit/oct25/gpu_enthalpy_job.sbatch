#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
export LOCAL_RANK=0
unset DEVICE
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu --format=csv > gpu-at-start.txt
mkdir -p cache/15000 cache/20000
srun --exclusive --exact -N1 -n1 -c4 --gres=gpu:a100:1 --mem=12G --cpu-bind=cores env XDG_CACHE_HOME="$PWD/cache/15000" python gpu_enthalpy_worker.py 15000 > worker-15000.log 2>&1 &
p15=$!
srun --exclusive --exact -N1 -n1 -c4 --gres=gpu:a100:1 --mem=12G --cpu-bind=cores env XDG_CACHE_HOME="$PWD/cache/20000" python gpu_enthalpy_worker.py 20000 > worker-20000.log 2>&1 &
p20=$!
status15=0
status20=0
wait "$p15" || status15=$?
wait "$p20" || status20=$?
echo "WORKER_EXIT_CODES $status15 $status20"
test "$status15" -eq 0 && test "$status20" -eq 0
