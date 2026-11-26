#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/home6/u3129/tial-pd-20261124-calphad/packages
sha256sum --quiet -c inputs.sha256
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python liang.py Liang2024.tdb . \
    --temperatures 600 1000 1273.15 1323.15 1373.15 1423.15 --pressure 0
tar czf results.tar.gz liang_contacts.csv metadata.json
sha256sum results.tar.gz > results.sha256
