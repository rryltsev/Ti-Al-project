#!/bin/bash

# ============================================================
# ВЫБОР УСТРОЙСТВА (CPU / GPU)
# ============================================================
GPU_DEVICE="0,1"                # "" (CPU), 0 (RTX 5060 Ti), 1 (RTX 3050), 0,1 (обе)

# ============================================================
# ЛОГИРОВАНИЕ (как в первом скрипте)
# ============================================================
LOGS_DIR="/home/katkoff/finetune_deepmd_dpa4/logs"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/finetune_dpa4_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# ПЕРЕМЕННЫЕ
# ============================================================

NAME="TiAl"
VARIANCE="Neo"
DATASET="pbe"
SHARD="shard"
FIT="tune"

ROOT_DIR="/home/katkoff/finetune_deepmd_dpa4"
MODEL_CONFIG="${NAME}_${FIT}_${DATASET}_DPA4-${SHARD}-${VARIANCE}-OMat24-v20260805.json"
MODEL_PATH="/home/katkoff/ml_models/deepmd/dpa4/models/DPA4-${VARIANCE}-OMat24-v20260805.pt"

RUN_DIR="${ROOT_DIR}/run_fits/${NAME}_${FIT}_${DATASET}_${VARIANCE}_DPA4_${SHARD}_$(date +%Y%m%d_%H%M%S)"

# ============================================================
# АКТИВИРУЕМ CONDA ОКРУЖЕНИЕ
# ============================================================

CONDA_ENV="ase_deepmd_dpa4"

if [[ -f "/home/katkoff/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "/home/katkoff/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    echo "Activated: ${CONDA_ENV}"
    echo "Python: $(which python)"
    echo "Python version: $(python --version 2>&1)"
else
    echo "ERROR: Conda activate script not found"
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
# СОЗДАЁМ RUN DIRECTORY
# ============================================================

mkdir -p "${RUN_DIR}/output_model"

# ============================================================
# ПЕРЕХОДИМ В RUN DIRECTORY
# ============================================================

cd "${RUN_DIR}" || exit 1

# ============================================================
# LOCAL CACHE
# ============================================================

export XDG_CACHE_HOME="${PWD}/.cache"
mkdir -p "${XDG_CACHE_HOME}"

export TRITON_HOME="${PWD}/.triton"
mkdir -p "${TRITON_HOME}"

echo "XDG_CACHE_HOME: ${XDG_CACHE_HOME}"
echo "TRITON_HOME:    ${TRITON_HOME}"

# ============================================================
# УСТАНАВЛИВАЕМ ДЕВАЙС
# ============================================================

if [ -z "${GPU_DEVICE}" ]; then
    # CPU режим
    DEVICE="cpu"
    export CUDA_VISIBLE_DEVICES=""
    echo "=========================================="
    echo "Running on CPU"
    echo "=========================================="
else
    # GPU режим
    DEVICE="cuda"
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
    echo "=========================================="
    echo "Running on GPU device(s): ${GPU_DEVICE}"
    echo "=========================================="
fi

# ============================================================
# КОПИРУЕМ КОНФИГ
# ============================================================

CONFIG_SOURCE="${ROOT_DIR}/models/${MODEL_CONFIG}"

if [ ! -f "${CONFIG_SOURCE}" ]; then
    echo "ERROR: Configuration file not found:"
    echo "${CONFIG_SOURCE}"
    exit 1
fi

cp "${CONFIG_SOURCE}" "${RUN_DIR}/input.json" || exit 1

# Заменяем пути с кластера на WSL
#sed -i 's|/home/u2626/_scratch2/finetune_pipline_deepmd|/home/katkoff/finetune_deepmd_dpa4|g' "${RUN_DIR}/input.json"
#echo "Updated paths in input.json: /home/u2626/... → /home/katkoff/..."

# ============================================================
# ИНФОРМАЦИЯ О ЗАПУСКЕ
# ============================================================

echo "=========================================="
echo "${NAME} ${VARIANCE} ${DATASET} ${FIT} DPA4 training"
echo "=========================================="
echo "Date:                $(date)"
echo "Node:                $(hostname)"
echo "Run directory:       ${RUN_DIR}"
echo "Config:              ${MODEL_CONFIG}"
echo "Log file:            ${LOG_FILE}"
echo "=========================================="
echo "Environment"
echo "=========================================="
echo "Conda env:           ${CONDA_DEFAULT_ENV}"
echo "Python:              $(which python)"
echo "DP:                  $(which dp)"
echo "=========================================="
echo "CPU resources"
echo "=========================================="
echo "OMP_NUM_THREADS:     ${OMP_NUM_THREADS}"
echo "DP intra threads:    ${DP_INTRA_OP_PARALLELISM_THREADS}"
echo "DP inter threads:    ${DP_INTER_OP_PARALLELISM_THREADS}"
echo "=========================================="
echo "Cache directories"
echo "=========================================="
echo "XDG_CACHE_HOME:      ${XDG_CACHE_HOME}"
echo "TRITON_HOME:         ${TRITON_HOME}"
echo "=========================================="

# ============================================================
# DEEPMD VERSION
# ============================================================

echo "DeepMD version:"
dp --version

echo "=========================================="
echo "Starting DPA4 training"
echo "=========================================="

# ============================================================
# TRAIN (прямой запуск, без &)
# ============================================================

#dp --pt train input.json --finetune "${MODEL_PATH}" -o out.json

torchrun --standalone --nproc_per_node=2 $(which dp) --pt train input.json --finetune "${MODEL_PATH}" -o out.json

EXIT_CODE=$?

# ============================================================
# РЕЗУЛЬТАТ
# ============================================================

echo "=========================================="
echo "Training finished"
echo "=========================================="
echo "Exit code: ${EXIT_CODE}"
echo "Results:   ${RUN_DIR}"
echo "Log file:  ${LOG_FILE}"
echo "=========================================="

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ SUCCESS! Training completed."    
else
    echo "❌ ERROR: Training failed with code ${EXIT_CODE}"
    echo "Check log: tail -100 ${LOG_FILE}"
fi

echo "=========================================="

exit ${EXIT_CODE}

# ============================================================
# КОМАНДЫ ДЛЯ ЗАПУСКА:
# ============================================================
# 1. Запуск в фоне:
#    nohup bash /home/katkoff/finetune_deepmd_dpa4/run_finetune_dpa4_shard.sh > /dev/null &
# 
# 2. Мониторинг лога (рекомендую):
#    tail -f /home/katkoff/finetune_deepmd_dpa4/logs/finetune_dpa4_*.log
# 
# 3. Проверка статуса:
#    ps aux | grep "dp --pt finetune"
# 
# 4. Посмотреть папки с запусками:
#    ls -lt /home/katkoff/finetune_deepmd_dpa4/run_fits/ | head
