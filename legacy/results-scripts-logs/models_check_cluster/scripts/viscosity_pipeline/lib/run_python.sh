#!/bin/bash
# ============================================================
# run_python.sh — запуск python-скрипта шага с логированием
# ============================================================
set -euo pipefail

[ -n "${PYTHON_SCRIPT:-}" ] || { echo "ERROR: PYTHON_SCRIPT not set"; exit 1; }
[ -f "$PYTHON_SCRIPT" ]    || { echo "ERROR: not found: $PYTHON_SCRIPT"; exit 1; }

echo "[$(date '+%H:%M:%S')] python ${PYTHON_SCRIPT}"
echo "[$(date '+%H:%M:%S')] args: ${STEP_ARGS[*]}"

python "$PYTHON_SCRIPT" "${STEP_ARGS[@]}"
rc=$?

echo "[$(date '+%H:%M:%S')] python exit code: ${rc}"
exit "$rc"