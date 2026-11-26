#!/bin/bash
set -eo pipefail
umask 077
TI_DRIVER="$1"
TI_XC="$2"
TI_STEP="$3"
TI_PARENT="${4:-}"
TI_ROOT=/home6/u3129/tial-phonons/alpha-ti-xc-phonons-20261021-01
case "$TI_DRIVER" in
  restart_step_v2.py|production_step_v3.py) ;;
  *) exit 64 ;;
esac
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
cd "$TI_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLURM_KILL_BAD_EXIT=1
test -r "$TI_ROOT/$TI_DRIVER"
if [ "${TIAL_LAUNCH_CHECK:-0}" = 1 ]; then
  printf 'Launch check: cwd=%s driver=%s xc=%s step=%s parent=%s\n' "$PWD" "$TI_ROOT/$TI_DRIVER" "$TI_XC" "$TI_STEP" "$TI_PARENT"
  exit 0
fi
if [ "$TI_DRIVER" = restart_step_v2.py ]; then
  exec /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$TI_ROOT/$TI_DRIVER" "$TI_XC" "$TI_STEP" "$TI_PARENT"
else
  exec /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$TI_ROOT/$TI_DRIVER" "$TI_XC" "$TI_STEP"
fi
