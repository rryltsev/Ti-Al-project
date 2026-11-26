#!/bin/bash
set -eo pipefail
umask 077
TI_DRIVER="$1"
TI_XC="$2"
TI_STEP="$3"
TI_PARENT="${4:-}"
TI_ROOT=/home6/u3129/tial-phonons/alpha-ti-xc-phonons-20261021-01
case "$TI_DRIVER" in
  restart_step_v2.py|restart_step_v3.py|restart_step_v4.py|restart_step_v5.py|restart_step_v6.py|restart_step_v7.py|production_step_v3.py|production_step_v4.py|production_step_v5.py|production_step_v6.py|production_step_v7.py|production_step_v8.py) ;;
  *) exit 64 ;;
esac
if [ "${TIAL_LAUNCH_CHECK:-0}" != 1 ]; then
  test "${SLURM_JOB_PARTITION:-}" = apollo || { echo "VASP requires Apollo; A100 forbidden" >&2; exit 65; }
  case "${SLURM_JOB_NODELIST:-}" in *tesla-a101*) exit 65 ;; esac
fi
source /home/u3129/.local/opt/vasp/6.6.1-oneapi-2025/env.sh
cd "$TI_ROOT"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLURM_KILL_BAD_EXIT=1
test -r "$TI_ROOT/$TI_DRIVER"
if [ "${TIAL_LAUNCH_CHECK:-0}" = 1 ]; then
  printf 'Launch check: cwd=%s driver=%s xc=%s step=%s parent=%s\n' "$PWD" "$TI_ROOT/$TI_DRIVER" "$TI_XC" "$TI_STEP" "$TI_PARENT"
  exit 0
fi
if [[ "$TI_DRIVER" == restart_step_v*.py ]]; then
  exec /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$TI_ROOT/$TI_DRIVER" "$TI_XC" "$TI_STEP" "$TI_PARENT"
else
  exec /home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$TI_ROOT/$TI_DRIVER" "$TI_XC" "$TI_STEP"
fi
