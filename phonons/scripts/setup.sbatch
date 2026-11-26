#!/bin/bash
set -eo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
BASE=/home6/u3129/mlip-research/envs/kplib_111/bin/python
ENV=/home6/u3129/tial-phonons/envs/phonopy-4.8.2
export PIP_CACHE_DIR="$PWD/pip-cache"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"
test ! -e "$ENV"
"$BASE" -m venv "$ENV"
"$ENV/bin/python" -m pip install --only-binary=:all: phonopy==4.8.2 ase==3.29.0
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze > environment.lock.txt
"$ENV/bin/python" workflow.py prepare
touch setup.success
