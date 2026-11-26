#!/bin/bash
set -eo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
RUN=/home6/u3129/tial-fm-research/dpa4-n200-pilot-20261106-02
CASES=(PBE-lora16 PBE-full r2SCAN-lora16 r2SCAN-full)
CASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}
cd "$RUN/training-plan01/cases/$CASE_NAME"
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$RUN/analyze_predictions.py" --package "$RUN" --predictions evaluation/predictions --output evaluation/metrics
