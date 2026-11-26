#!/bin/bash
# ============================================================
# run_pipeline.sh — единая точка запуска пайплайна viscosity
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR

PIPELINE_DIR="${SCRIPT_DIR}/models_check/scripts/viscosity_pipeline"
export PIPELINE_DIR

source "${PIPELINE_DIR}/common.sh"

init_run

rc=0
run_step "01_generate_liquid"     || rc=$?
[ $rc -eq 0 ] && { run_step "02_nvt_snapshots"       || rc=$?; }
[ $rc -eq 0 ] && { run_step "03_stress_autocorr"     || rc=$?; }
[ $rc -eq 0 ] && { run_step "04_viscosity_analysis"  || rc=$?; }

finalize_run "$rc"
exit "$rc"

# ============================================================
# КОМАНДЫ ДЛЯ ЗАПУСКА:
# ============================================================
# 1. Запуск в фоне:
#    nohup bash /home/katkoff/deepmd_check_pipline/run_pipeline_viscosity.sh > /dev/null &
#
# 2. Мониторинг master-лога (последний запуск viscosity):
#    tail -f /home/katkoff/deepmd_check_pipline/logs/latest_viscosity/master.log
#
# 3. Мониторинг лога шага 01 (последний запуск):
#    tail -f /home/katkoff/deepmd_check_pipline/logs/latest_viscosity/01_generate_liquid.log
#
# 4. Мониторинг, если запусков много (по времени):
#    tail -f "$(ls -td /home/katkoff/deepmd_check_pipline/logs/viscosity_run_* | head -1)/master.log"
#
# 5. Проверка статуса:
#    ps aux | grep generate_liquid_viscosity
#
# 6. Посмотреть все запуски:
#    ls -lt /home/katkoff/deepmd_check_pipline/logs/ | head
#
# 7. Изменить модель: править MODE в начале run_pipeline_viscosity.sh
#    или экспортировать перед запуском:
#      MODE=grace-2L-pretrained bash run_pipeline_viscosity.sh
# ============================================================