#!/bin/bash
set -eo pipefail
umask 077
cd /home6/u3129/tial-fm-research/dpa4-n200-pilot-20261106-02
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python -c 'import json; assert json.load(open("baseline/summary.json"))["status"]=="PASS"'
if test -s harmonics/summary.json; then
  /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python -c 'import json; assert json.load(open("harmonics/summary.json"))["status"]=="PASS"'
  exit 0
fi
test ! -d harmonics
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python harmonic_worker.py
