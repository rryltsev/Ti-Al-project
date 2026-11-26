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
RUN=/home6/u3129/tial-fm-research/dpa4-replicates-full-20261106-02
GROUP="$1"
if [ "$GROUP" = N200 ]; then
CASES=(PBE-lora16-N200-seed20 PBE-full-N200-seed20 r2SCAN-lora16-N200-seed20 r2SCAN-full-N200-seed20 PBE-lora16-N200-seed30 PBE-full-N200-seed30 r2SCAN-lora16-N200-seed30 r2SCAN-full-N200-seed30)
else
test "$GROUP" = N694
CASES=(PBE-lora16-N694-seed10 PBE-full-N694-seed10 r2SCAN-lora16-N694-seed10 r2SCAN-full-N694-seed10)
fi
CASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}
cd "$RUN/training-plan01/cases/$CASE_NAME"
test ! -e attempt.json
python -c 'import json,os,socket; open("attempt.json","x").write(json.dumps(dict(job=os.environ["SLURM_JOB_ID"],array=os.environ["SLURM_ARRAY_JOB_ID"],task=os.environ["SLURM_ARRAY_TASK_ID"],host=socket.gethostname())))'
python "$RUN/verify_case.py"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv > hardware.txt
python ../../train_observer.py --pt train input.json --finetune /home6/u3129/tial-testsets/pretrained-neo-20261022-01/DPA4-Neo-OMat24-v20260805.pt > train.log 2>&1
python ../../evaluate_pilot.py > evaluate.log 2>&1
unset PYTHONPATH
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python ../../evaluate_pilot_harmonics.py > harmonics.log 2>&1
N=${GROUP#N}
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$RUN/analyze_predictions.py" --package "$RUN" --data-root "$RUN/data/$GROUP" --train-size "$N" --predictions evaluation/predictions --output evaluation/metrics > metrics.log 2>&1
