#!/bin/bash

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================
LOGS_DIR="/home/katkoff/deepmd_check_pipline/logs"
mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/run_phonons_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

# ============================================================
# ПЕРЕМЕННЫЕ
# ============================================================
ROOT_DIR="/home/katkoff/deepmd_check_pipline"

# Режим модели: <family>-<variant>-<use_type>
# family: dpa4, dpa4c, grace
# variant: nano, mini, neo, air, plus / 2L, 3L (по возрастанию сложности)
# use_type: pretrained, fit, finetune, finetune_half
MODE="dpa4c-air-finetune"   # ← МЕНЯЙ ЗДЕСЬ!
#MODE="grace-2L-pretrained"   # ← МЕНЯЙ ЗДЕСЬ!

# Калькулятор (для phonon_calculator.py)
CALCULATOR="deepmd"           # deepmd, grace, etc.

# Система (для выбора CIF файла)
SYSTEM_NAME="alpha2_AlTi3"      # alpha_Ti, fcc_Al, gamma_AlTi, alpha2_AlTi3

# Метод DFT (для моделей с несколькими тренировочными режимами)
# "" (пусто) — плоская структура JSON
# "pbe" / "scan" / "sol" — четырёхуровневая структура JSON
DFT_METHOD="pbe"                # pbe, scan, sol, или ""

# Словарь: номер карты → имя
declare -A GPU_NAMES=(
    ["0"]="RTX 5060 Ti"
    ["1"]="RTX 3050"
)

GPU_DEVICE="1"                # "" (CPU), 0 (RTX 5060 Ti), 1 (RTX 3050), 0,1 (обе)

# Настройки расчёта (можно менять)
SUPERCELL="5,5,5"
DELTA="0.01"
KPTS_DOS="50,50,50"
DOS_WIDTH="0.0002"
DOS_NPTS="1000"
BAND_NPOINTS="200"
USE_THZ="true"
FMAX="0.0001"
MAX_STEPS="1000"
PRESSURE="0.0"

# Для FCC (fcc_Al) можно указать перестановку X точек
X_PERMUTATION="type3"         # type1, type2, type3 или оставить пустым ""

# ============================================================
# ОПРЕДЕЛЕНИЕ DEVICE И DTYPE
# ============================================================
if [ -z "${GPU_DEVICE}" ]; then
    GPU_NAME="CPU"
    DEVICE="cpu"
    DTYPE="float64"
    export CUDA_VISIBLE_DEVICES=""
    export CUDA_AVAILABLE_DEVICES=""
    echo "No GPU specified. Using CPU with float64"
else
    DEVICE="cuda"
    DTYPE="float32"
    export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
    FIRST_GPU="${GPU_DEVICE%%,*}"
    GPU_NAME="${GPU_NAMES[$FIRST_GPU]:-unknown}"
    echo "Using GPU device(s): ${GPU_DEVICE}, GPU: ${GPU_NAME}"    
    echo "Device: ${DEVICE}, Dtype: ${DTYPE}"
fi

# Зона Бриллюэна в зависимости от системы
if [ "$SYSTEM_NAME" == "alpha_Ti" ]; then
    BAND_PATH="GKMGA"
elif [ "$SYSTEM_NAME" == "fcc_Al" ]; then
    BAND_PATH="GXKGL"
elif [ "$SYSTEM_NAME" == "gamma_AlTi" ]; then
    BAND_PATH="GXMGZRAZ"
elif [ "$SYSTEM_NAME" == "alpha2_AlTi3" ]; then
    BAND_PATH="GMKGALHA"    
else
    BAND_PATH="GKMGA"
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

# Преобразуем в нужный формат:
# family: dpa4 → DPA4, grace → GRACE
# variant: 2L → 2L (оставляем как есть), neo → Neo
FAMILY_JSON=$(echo "$FAMILY" | tr '[:lower:]' '[:upper:]')
# Для варианта: если начинается с цифры, оставляем как есть, иначе Capitalize
if [[ "$VARIANT" =~ ^[0-9] ]]; then
    VARIANT_JSON="$VARIANT"
else
    VARIANT_JSON=$(echo "$VARIANT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
fi

if [ -z "${DFT_METHOD}" ]; then
    JQ_PATH=".${FAMILY_JSON}[\"${VARIANT_JSON}\"].${USE_TYPE}"
else
    JQ_PATH=".${FAMILY_JSON}[\"${VARIANT_JSON}\"][\"${DFT_METHOD}\"].${USE_TYPE}"
fi

MODEL_PATH=$(jq -r "${JQ_PATH}.path" "${CONFIG_MODEL_PATHS}")
MODEL_NAME=$(jq -r "${JQ_PATH}.name" "${CONFIG_MODEL_PATHS}")

if [ -z "${MODEL_PATH}" ] || [ "${MODEL_PATH}" == "null" ] || \
   [ -z "${MODEL_NAME}" ] || [ "${MODEL_NAME}" == "null" ]; then
    echo "ERROR: Model path or name is empty/null for MODE=${MODE}"
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
    # DeepMD — это файл .pt или .pth
    if [ ! -f "${MODEL_FULL_PATH}" ]; then
        echo "ERROR: Model file not found: ${MODEL_FULL_PATH}"
        exit 1
    fi
elif [ "${CALCULATOR}" == "grace" ]; then
    # GRACE — это папка с моделью
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
WORKDIR_BASE="${ROOT_DIR}/models_check/results/phonons"
WORKDIR="${WORKDIR_BASE}/${FAMILY}/${VARIANT}/${USE_TYPE}/${SYSTEM_NAME}"

if [ -n "${DFT_METHOD}" ]; then
    WORKDIR="${WORKDIR}/${DFT_METHOD}"
fi

mkdir -p "${WORKDIR}"

echo "=========================================="
echo "Results directory:"
echo "  ${WORKDIR}"
echo "=========================================="

# ============================================================
# ПУТЬ К STRUCTURES
# ============================================================
STRUCTURES_DIR="${ROOT_DIR}/models_check/structures"
CIF_FILE="${STRUCTURES_DIR}/${SYSTEM_NAME}.cif"

if [ ! -f "${CIF_FILE}" ]; then
    echo "ERROR: CIF file not found: ${CIF_FILE}"
    exit 1
fi

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
# LOCAL CACHE (в зависимости от CALCULATOR)
# ============================================================
if [ "${CALCULATOR}" == "deepmd" ]; then
    # DeepMD — Triton и XDG
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
    # GRACE — Numba и XDG (без Triton)
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
echo "Phonon calculation (WSL2)"
echo "=========================================="
echo "Date:          $(date)"
echo "System:        ${SYSTEM_NAME}"
echo "Mode:          ${MODE}"
echo "Model path:    ${MODEL_FULL_PATH}"
echo "DFT method:    ${DFT_METHOD:-<none>}"
echo "CIF file:      ${CIF_FILE}"
echo "Supercell:     ${SUPERCELL}"
echo "Delta:         ${DELTA}"
echo "Kpts DOS:      ${KPTS_DOS}"
echo "DOS width:     ${DOS_WIDTH}"
echo "DOS npts:      ${DOS_NPTS}"
echo "Band path:     ${BAND_PATH}"
echo "Band npoints:  ${BAND_NPOINTS}"
echo "Use THz:       ${USE_THZ}"
echo "Fmax:          ${FMAX}"
echo "Max steps:     ${MAX_STEPS}"
echo "Pressure:      ${PRESSURE}"
echo "Workdir:       ${WORKDIR}"
echo "Log file:      ${LOG_FILE}"
echo "=========================================="

# ============================================================
# ЗАПУСК PHONON CALCULATOR
# ============================================================
python "${ROOT_DIR}/models_check/scripts/phonon_calculator.py" \
    --calculator "${CALCULATOR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --cif "${CIF_FILE}" \
    --system "${SYSTEM_NAME}" \
    --family "${FAMILY}" \
    --variant "${VARIANT}" \
    --use-type "${USE_TYPE}" \
    --dft-method "${DFT_METHOD}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --gpu-name "${GPU_NAME}" \
    --workdir "${WORKDIR_BASE}" \
    --fmax "${FMAX}" \
    --max-steps "${MAX_STEPS}" \
    --pressure "${PRESSURE}" \
    --supercell "${SUPERCELL}" \
    --delta "${DELTA}" \
    --kpts-dos "${KPTS_DOS}" \
    --dos-width "${DOS_WIDTH}" \
    --dos-npts "${DOS_NPTS}" \
    --band-path "${BAND_PATH}" \
    --band-npoints "${BAND_NPOINTS}" \
    $( [ -n "${X_PERMUTATION}" ] && echo "--x-permutation ${X_PERMUTATION}" ) \
    $( [ "$USE_THZ" == "true" ] && echo "--use-thz" )

EXIT_CODE=$?

# ============================================================
# РЕЗУЛЬТАТ
# ============================================================
echo "=========================================="
echo "Phonon calculation finished"
echo "=========================================="
echo "Exit code: ${EXIT_CODE}"
echo "Results:   ${WORKDIR}"
echo "Log file:  ${LOG_FILE}"
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
#    nohup bash /home/katkoff/deepmd_check_pipline/run_phonons.sh > /dev/null &
#
# 2. Мониторинг лога:
#    tail -f /home/katkoff/deepmd_check_pipline/logs/run_phonons_*.log
#
# 3. Проверка статуса:
#    ps aux | grep phonon_calculator
#
# 4. Посмотреть все запуски:
#    ls -lt /home/katkoff/deepmd_check_pipline/logs/ | head
#
# 5. Изменить модель: править MODE в начале скрипта
#    dpa4-air-pretrained, dpa4c-neo-fit, etc.
# ============================================================