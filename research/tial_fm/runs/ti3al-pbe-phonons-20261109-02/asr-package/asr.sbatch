#!/bin/bash
set -eo pipefail
umask 077
ulimit -c 0
cd /home6/u3129/tial-fm-research/ti3al-pbe-phonons-20261109-02
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python ti3al_asr_sensitivity.py
