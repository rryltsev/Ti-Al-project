#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export XDG_CACHE_HOME="$PWD/cache/$SLURM_JOB_ID"
mkdir -p "$XDG_CACHE_HOME"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
export DP_TF32_INFER=0 DP_AMP_INFER=0 DP_TRITON_INFER=0 DP_COMPILE_INFER=0
source /home/u3129/miniconda3/etc/profile.d/conda.sh
conda activate lammps_deepmd_dpa4
export CUDA_HOME=/opt/cuda-13.0
export CXX=/opt/rh/gcc-toolset-12/root/usr/bin/g++
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=""
RUN=/home6/u3129/tial-fm-research/dpa4-n694-seeds-20261107-01
GROUP="$1"
test "$GROUP" = N694
CASES=(PBE-lora16-N694-seed20 PBE-full-N694-seed20 r2SCAN-lora16-N694-seed20 r2SCAN-full-N694-seed20 PBE-lora16-N694-seed30 PBE-full-N694-seed30 r2SCAN-lora16-N694-seed30 r2SCAN-full-N694-seed30)
CASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}
cd "$RUN/training-plan01/cases/$CASE_NAME"
test -f evaluation/summary.json
python "$RUN/trained_numerical_qc.py" --case "$CASE_NAME" > numerical-QC.log 2>&1
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$RUN/expansion_case_physics.py" --case "$CASE_NAME" > physical-diagnostics.log 2>&1
