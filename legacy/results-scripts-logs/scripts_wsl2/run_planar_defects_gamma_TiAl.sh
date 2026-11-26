#!/bin/bash

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================
LOGS_DIR="/home/katkoff/deepmd_check_pipline/logs"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/run_planar_defects_gamma_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# ВРЕМЯ СТАРТА
# ============================================================
START_TIME=$(date +%s)
START_DATE=$(date)
echo "Start time: ${START_DATE}"

# ============================================================
# ПЕРЕМЕННЫЕ
# ============================================================
ROOT_DIR="/home/katkoff/deepmd_check_pipline"

# Режим модели: <family>-<variant>-<use_type>
# family: dpa4, dpa4c, grace
# variant: nano, mini, neo, air, plus / 2L, 3L
# use_type: pretrained, fit, finetune, finetune_half
MODE="dpa4-air-finetune_half"   # ← МЕНЯЙ ЗДЕСЬ!
#MODE="grace-2L-pretrained"   # ← МЕНЯЙ ЗДЕСЬ!

# Калькулятор
CALCULATOR="deepmd"           # deepmd, grace, etc.

# Система (только gamma_AlTi для этого скрипта)
SYSTEM_NAME="gamma_AlTi"

GPU_DEVICE="0"                # "" (CPU), 0 (RTX 5060 Ti), 1 (RTX 3050), 0,1 (обе)

# Настройки расчёта
N_LAYERS="16"
VACUUM="12"
REPEAT_INPLANE="3,3"
FMAX="0.001"
MAX_STEPS="1000"

# Сдвиги
SHIFT_STEP="0.025"
MAX_FRACTION="1.0"

# Дефекты (по статье)
DEFECT_TYPES="111_1-10,111_11-2,111_combined"

# ============================================================
# ОПРЕДЕЛЕНИЕ DEVICE И DTYPE
# ============================================================
if [ -z "${GPU_DEVICE}" ]; then
    DEVICE="cpu"
    DTYPE="float64"
    export CUDA_VISIBLE_DEVICES=""
    export CUDA_AVAILABLE_DEVICES=""
    echo "No GPU specified. Using CPU with float64"
else
    DEVICE="cuda"
    DTYPE="float32"
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
    echo "Using GPU device(s): ${GPU_DEVICE} (RTX 5060 Ti = 0, RTX 3050 = 1)"
    echo "Device: ${DEVICE}, Dtype: ${DTYPE}"
fi

# ============================================================
# РАЗБОР MODE
# ============================================================
FAMILY=$(echo "$MODE" | cut -d'-' -f1)
VARIANT=$(echo "$MODE" | cut -d'-' -f2)
USE_TYPE=$(echo "$MODE" | cut -d'-' -f3)

if [ -z "$FAMILY" ] || [ -z "$VARIANT" ] || [ -z "$USE_TYPE" ]; then
    echo "ERROR: MODE must be in format <family>-<variant>-<use_type>"
    echo "Example: dpa4c-neo-pretrained"
    exit 1
fi

echo "=========================================="
echo "Parsed MODE:"
echo "  Family:    ${FAMILY}"
echo "  Variant:   ${VARIANT}"
echo "  Use type:  ${USE_TYPE}"
echo "=========================================="

# ============================================================
# ПУТИ ИЗ КОНФИГА
# ============================================================
CONFIG_MODEL_PATHS="${ROOT_DIR}/configs/model_paths.json"

if [ ! -f "${CONFIG_MODEL_PATHS}" ]; then
    echo "ERROR: model_paths.json not found: ${CONFIG_MODEL_PATHS}"
    exit 1
fi

# Преобразуем в нужный формат
FAMILY_JSON=$(echo "$FAMILY" | tr '[:lower:]' '[:upper:]')
if [[ "$VARIANT" =~ ^[0-9] ]]; then
    VARIANT_JSON="$VARIANT"
else
    VARIANT_JSON=$(echo "$VARIANT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
fi

MODEL_PATH=$(jq -r ".${FAMILY_JSON}[\"${VARIANT_JSON}\"].${USE_TYPE}.path" "${CONFIG_MODEL_PATHS}")
MODEL_NAME=$(jq -r ".${FAMILY_JSON}[\"${VARIANT_JSON}\"].${USE_TYPE}.name" "${CONFIG_MODEL_PATHS}")

if [ -z "${MODEL_PATH}" ] || [ -z "${MODEL_NAME}" ]; then
    echo "ERROR: Model path or name is empty for MODE=${MODE}"
    echo "Check ${CONFIG_MODEL_PATHS}"
    exit 1
fi

MODEL_FULL_PATH="${MODEL_PATH}/${MODEL_NAME}"

echo "=========================================="
echo "Model configuration:"
echo "  Path:      ${MODEL_PATH}"
echo "  Name:      ${MODEL_NAME}"
echo "  Full path: ${MODEL_FULL_PATH}"
echo "=========================================="

# ============================================================
# ПРОВЕРКА СУЩЕСТВОВАНИЯ МОДЕЛИ
# ============================================================
if [ "${CALCULATOR}" == "deepmd" ]; then
    if [ ! -f "${MODEL_FULL_PATH}" ]; then
        echo "ERROR: Model file not found: ${MODEL_FULL_PATH}"
        exit 1
    fi
elif [ "${CALCULATOR}" == "grace" ]; then
    if [ ! -d "${MODEL_FULL_PATH}" ]; then
        echo "ERROR: Model directory not found: ${MODEL_FULL_PATH}"
        exit 1
    fi
else
    echo "ERROR: Unknown CALCULATOR: ${CALCULATOR}"
    exit 1
fi

echo "Model found: ${MODEL_FULL_PATH}"

# ============================================================
# ПУТЬ К РЕЗУЛЬТАТАМ
# ============================================================
WORKDIR_BASE="${ROOT_DIR}/models_check/results/planar_defects"
WORKDIR="${WORKDIR_BASE}/${FAMILY}/${VARIANT}/${USE_TYPE}/${SYSTEM_NAME}"

mkdir -p "${WORKDIR}"

echo "=========================================="
echo "Results directory:"
echo "  ${WORKDIR}"
echo "=========================================="

# ============================================================
# АКТИВИРУЕМ CONDA ОКРУЖЕНИЕ
# ============================================================
if [ "${CALCULATOR}" == "deepmd" ]; then
    CONDA_ENV="ase_deepmd_dpa4"
elif [ "${CALCULATOR}" == "grace" ]; then
    CONDA_ENV="ase_project_grace"
else
    echo "ERROR: Unknown CALCULATOR: ${CALCULATOR}"
    exit 1
fi

if [[ -f "/home/katkoff/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "/home/katkoff/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "Activated: ${CONDA_ENV}"
    echo "Python: $(which python)"
else
    echo "ERROR: Conda not found at /home/katkoff/miniconda3"
    exit 1
fi

# ============================================================
# CPU THREADS
# ============================================================
CPU_THREADS=8
export OMP_NUM_THREADS="${CPU_THREADS}"
export DP_INTRA_OP_PARALLELISM_THREADS="${CPU_THREADS}"
export DP_INTER_OP_PARALLELISM_THREADS=1

# ============================================================
# ПЕРЕХОДИМ В WORKDIR
# ============================================================
cd "${WORKDIR}" || exit 1

# ============================================================
# LOCAL CACHE
# ============================================================
if [ "${CALCULATOR}" == "deepmd" ]; then
    export XDG_CACHE_HOME="${PWD}/.cache"
    mkdir -p "${XDG_CACHE_HOME}"

    export TRITON_HOME="${PWD}/.triton"
    mkdir -p "${TRITON_HOME}"

    echo "=========================================="
    echo "Cache directories (DeepMD):"
    echo "  XDG_CACHE_HOME: ${XDG_CACHE_HOME}"
    echo "  TRITON_HOME:    ${TRITON_HOME}"
    echo "=========================================="

elif [ "${CALCULATOR}" == "grace" ]; then
    export NUMBA_CACHE_DIR="${PWD}/.numba_cache"
    mkdir -p "${NUMBA_CACHE_DIR}"

    export XDG_CACHE_HOME="${PWD}/.cache"
    mkdir -p "${XDG_CACHE_HOME}"

    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_CPP_MIN_LOG_LEVEL=2
    export TF_USE_LEGACY_KERAS=1
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

    echo "=========================================="
    echo "Cache directories (GRACE):"
    echo "  NUMBA_CACHE_DIR: ${NUMBA_CACHE_DIR}"
    echo "  XDG_CACHE_HOME:  ${XDG_CACHE_HOME}"
    echo "=========================================="

else
    echo "ERROR: Unknown CALCULATOR: ${CALCULATOR}"
    exit 1
fi

# ============================================================
# ИНФОРМАЦИЯ О ЗАПУСКЕ
# ============================================================
echo "=========================================="
echo "Planar defect calculation (WSL2) - gamma_AlTi"
echo "=========================================="
echo "Date:          $(date)"
echo "System:        ${SYSTEM_NAME}"
echo "Mode:          ${MODE}"
echo "Calculator:    ${CALCULATOR}"
echo "Model path:    ${MODEL_FULL_PATH}"
echo "Defects:       ${DEFECT_TYPES}"
echo "Shift step:    ${SHIFT_STEP}"
echo "Max fraction:  ${MAX_FRACTION}"
echo "Slab:          ${N_LAYERS} layers, ${VACUUM} Å vacuum, ${REPEAT_INPLANE} repeat"
echo "Fmax:          ${FMAX} eV/Å"
echo "Max steps:     ${MAX_STEPS}"
echo "Workdir:       ${WORKDIR}"
echo "Log file:      ${LOG_FILE}"
echo "=========================================="

# ============================================================
# ЗАПУСК
# ============================================================
python "${ROOT_DIR}/models_check/scripts/planar_defects_gamma_tial.py" \
    --calculator "${CALCULATOR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --system "${SYSTEM_NAME}" \
    --family "${FAMILY}" \
    --variant "${VARIANT}" \
    --use-type "${USE_TYPE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --workdir "${WORKDIR_BASE}" \
    --defect-types "${DEFECT_TYPES}" \
    --shift-step "${SHIFT_STEP}" \
    --max-fraction "${MAX_FRACTION}" \
    --n-layers "${N_LAYERS}" \
    --vacuum "${VACUUM}" \
    --repeat-inplane "${REPEAT_INPLANE}" \
    --fmax "${FMAX}" \
    --max-steps "${MAX_STEPS}"

EXIT_CODE=$?

# ============================================================
# РЕЗУЛЬТАТ
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
END_DATE=$(date)

echo "=========================================="
echo "Planar defect calculation finished"
echo "=========================================="
echo "Start time:    ${START_DATE}"
echo "End time:      ${END_DATE}"
echo "Elapsed time:  ${ELAPSED} seconds ($((ELAPSED/60)) minutes)"
echo "Exit code:     ${EXIT_CODE}"
echo "Results:       ${WORKDIR}"
echo "Log file:      ${LOG_FILE}"
echo "=========================================="

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ SUCCESS!"
else
    echo "❌ ERROR: Calculation failed with code ${EXIT_CODE}"
    echo "Check log: tail -100 ${LOG_FILE}"
fi

echo "=========================================="
exit ${EXIT_CODE}

# ============================================================
# КОМАНДЫ ДЛЯ ЗАПУСКА:
# ============================================================
# 1. Запуск в фоне:
#    nohup bash /home/katkoff/deepmd_check_pipline/run_planar_defects_gamma_TiAl.sh > /dev/null &
#
# 2. Мониторинг лога:
#    tail -f /home/katkoff/deepmd_check_pipline/logs/run_planar_defects_gamma_*.log
#
# 3. Проверка статуса:
#    ps aux | grep planar_defects_gamma
# ============================================================