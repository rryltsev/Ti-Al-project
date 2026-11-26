#!/bin/bash
# ============================================================
# common.sh — общее ядро пайплайна viscosity
# ============================================================
set -euo pipefail

# ------------------------------------------------------------
# Корень и имя пайплайна
# ------------------------------------------------------------
ROOT_DIR="/home/katkoff/deepmd_check_pipline"
PIPELINE_NAME="viscosity"
PIPELINE_DIR="${ROOT_DIR}/models_check/scripts/viscosity_pipeline"

# ------------------------------------------------------------
# Логирование
# ------------------------------------------------------------
LOGS_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOGS_DIR"

RUN_ID="${PIPELINE_NAME}_run_$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="${LOGS_DIR}/${RUN_ID}"
mkdir -p "$RUN_LOG_DIR"
# Симлинк на последний запуск этого пайплайна
ln -sfn "$RUN_LOG_DIR" "${LOGS_DIR}/latest_${PIPELINE_NAME}"
LOG_FILE="${RUN_LOG_DIR}/master.log"

exec > >(tee -a "$LOG_FILE") 2>&1

START_EPOCH=$(date +%s)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

# ------------------------------------------------------------
# Хелперы лога
# ------------------------------------------------------------
log()  { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
hr()   { printf '%s\n' "=========================================="; }
banner() { hr; log "$*"; hr; }

# ------------------------------------------------------------
# Компоненты: словари масс и радиусов
# ------------------------------------------------------------
declare -A ATOMIC_MASSES=(
    [Al]=26.98
    [Ti]=47.88
)

declare -A ATOMIC_RADII=(
    [Al]=1.43
    [Ti]=1.47
)

# ------------------------------------------------------------
# Поведение шагов: FORCE_CLEAR / CONTINUE_ON_ERROR / SKIP
# ------------------------------------------------------------
declare -A STEP_FORCE_CLEAR=(
    [01_generate_liquid]=false
    [02_nvt_snapshots]=false
    [03_stress_autocorr]=false
    [04_viscosity_analysis]=false
)

declare -A STEP_CONTINUE_ON_ERROR=(
    [01_generate_liquid]=false
    [02_nvt_snapshots]=false
    [03_stress_autocorr]=false
    [04_viscosity_analysis]=true
)

declare -A STEP_SKIP=(
    [01_generate_liquid]=false
    [02_nvt_snapshots]=false
    [03_stress_autocorr]=false
    [04_viscosity_analysis]=false
)

# ------------------------------------------------------------
# Инициализация запуска
# ------------------------------------------------------------
init_run() {
    # --- Пользовательские настройки ---
    MODE="${MODE:-dpa4-neo-finetune}"
    CALCULATOR="${CALCULATOR:-deepmd}"
    DFT_METHOD="${DFT_METHOD:-pbe}"

    GPU_DEVICE="${GPU_DEVICE:-0}"
    CPU_THREADS="${CPU_THREADS:-8}"

    # --- Состав / система ---
    ELEMENTS_STR="${ELEMENTS_STR:-Al,Ti}"
    T_MEASURE="${T_MEASURE:-2000}"

    # --- Вычисляем ELEMENTS, SYSTEM_NAME, MASSES_STR, RADII_STR ---
    IFS=',' read -ra ELEMENTS <<< "$ELEMENTS_STR"

    SYSTEM_NAME=""
    for el in "${ELEMENTS[@]}"; do
        SYSTEM_NAME+="$el"
    done

    MASSES_STR=""
    RADII_STR=""
    for el in "${ELEMENTS[@]}"; do
        MASSES_STR+="${ATOMIC_MASSES[$el]:-0},"
        RADII_STR+="${ATOMIC_RADII[$el]:-0},"
    done
    MASSES_STR="${MASSES_STR%,}"
    RADII_STR="${RADII_STR%,}"

    # --- GPU / CPU ---
    declare -A GPU_NAMES=(
        ["0"]="RTX 5060 Ti"
        ["1"]="RTX 3050"
    )
    if [ -z "${GPU_DEVICE}" ]; then
        GPU_NAME="CPU"; DEVICE="cpu"; DTYPE="float64"
        export CUDA_VISIBLE_DEVICES=""
        export CUDA_AVAILABLE_DEVICES=""
        log "No GPU specified. Using CPU with float64"
    else
        DEVICE="cuda"; DTYPE="float32"
        export CUDA_VISIBLE_DEVICES="${GPU_DEVICE}"
        local first_gpu="${GPU_DEVICE%%,*}"
        GPU_NAME="${GPU_NAMES[$first_gpu]:-unknown}"
        log "Using GPU device(s): ${GPU_DEVICE}, GPU: ${GPU_NAME}"
        log "Device: ${DEVICE}, Dtype: ${DTYPE}"
    fi

    # --- Разбор MODE ---
    FAMILY=$(echo "$MODE" | cut -d'-' -f1)
    VARIANT=$(echo "$MODE" | cut -d'-' -f2)
    USE_TYPE=$(echo "$MODE" | cut -d'-' -f3)
    if [ -z "$FAMILY" ] || [ -z "$VARIANT" ] || [ -z "$USE_TYPE" ]; then
        log "ERROR: MODE must be <family>-<variant>-<use_type>, e.g. grace-2L-pretrained"
        exit 1
    fi
    banner "Parsed MODE"
    log "  Family:   ${FAMILY}"
    log "  Variant:  ${VARIANT}"
    log "  Use type: ${USE_TYPE}"

    # --- Model paths из JSON ---
    local cfg="${ROOT_DIR}/configs/model_paths.json"
    [ -f "$cfg" ] || { log "ERROR: ${cfg} not found"; exit 1; }

    local family_json variant_json jq_path
    family_json=$(echo "$FAMILY" | tr '[:lower:]' '[:upper:]')
    if [[ "$VARIANT" =~ ^[0-9] ]]; then
        variant_json="$VARIANT"
    else
        variant_json=$(echo "$VARIANT" | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
    fi

    if [ -z "${DFT_METHOD}" ]; then
        jq_path=".${family_json}[\"${variant_json}\"].${USE_TYPE}"
    else
        jq_path=".${family_json}[\"${variant_json}\"][\"${DFT_METHOD}\"].${USE_TYPE}"
    fi

    MODEL_PATH=$(jq -r "${jq_path}.path" "$cfg")
    MODEL_NAME=$(jq -r "${jq_path}.name" "$cfg")

    if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" = "null" ] || \
       [ -z "$MODEL_NAME" ] || [ "$MODEL_NAME" = "null" ]; then
        log "ERROR: model path/name empty for MODE=${MODE}, DFT_METHOD=${DFT_METHOD}"
        exit 1
    fi
    MODEL_FULL_PATH="${MODEL_PATH}/${MODEL_NAME}"

    banner "Model configuration"
    log "  Path:      ${MODEL_PATH}"
    log "  Name:      ${MODEL_NAME}"
    log "  Full path: ${MODEL_FULL_PATH}"

    case "$CALCULATOR" in
        deepmd)
            [ -f "$MODEL_FULL_PATH" ] || { log "ERROR: model file not found: ${MODEL_FULL_PATH}"; exit 1; }
            ;;
        grace)
            [ -d "$MODEL_FULL_PATH" ] || { log "ERROR: model dir not found: ${MODEL_FULL_PATH}"; exit 1; }
            ;;
        *)
            log "ERROR: unknown CALCULATOR: ${CALCULATOR}"
            exit 1
            ;;
    esac
    log "Model found: ${MODEL_FULL_PATH}"

    # --- База результатов ---
    WORKDIR_BASE="${ROOT_DIR}/models_check/results/${PIPELINE_NAME}/${FAMILY}/${VARIANT}/${USE_TYPE}/${SYSTEM_NAME}/${T_MEASURE}K"
    [ -n "${DFT_METHOD}" ] && WORKDIR_BASE="${WORKDIR_BASE}/${DFT_METHOD}"
    mkdir -p "$WORKDIR_BASE"

    banner "Results base"
    log "  ${WORKDIR_BASE}"

    # --- Conda ---
    case "$CALCULATOR" in
        deepmd) CONDA_ENV="lammps_deepmd_dpa4_env" ;;
        grace)  CONDA_ENV="lammps_grace_env" ;;
    esac

    if [[ -f "/home/katkoff/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "/home/katkoff/miniconda3/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        log "Activated: ${CONDA_ENV}"
        log "Python: $(which python)"
    else
        log "ERROR: conda not found at /home/katkoff/miniconda3"
        exit 1
    fi

    # --- CPU threads ---
    export OMP_NUM_THREADS="${CPU_THREADS}"
    export DP_INTRA_OP_PARALLELISM_THREADS="${CPU_THREADS}"
    export DP_INTER_OP_PARALLELISM_THREADS=1

    # --- Cache dirs ---
    cd "$WORKDIR_BASE" || exit 1
    if [ "$CALCULATOR" = "deepmd" ]; then
        export XDG_CACHE_HOME="${PWD}/.cache";  mkdir -p "$XDG_CACHE_HOME"
        export TRITON_HOME="${PWD}/.triton";    mkdir -p "$TRITON_HOME"
        log "Cache (DeepMD): XDG=${XDG_CACHE_HOME}, TRITON=${TRITON_HOME}"
    elif [ "$CALCULATOR" = "grace" ]; then
        export NUMBA_CACHE_DIR="${PWD}/.numba_cache"; mkdir -p "$NUMBA_CACHE_DIR"
        export XDG_CACHE_HOME="${PWD}/.cache";        mkdir -p "$XDG_CACHE_HOME"
        export TF_FORCE_GPU_ALLOW_GROWTH=true
        export TF_CPP_MIN_LOG_LEVEL=2
        export TF_USE_LEGACY_KERAS=1
        export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
        log "Cache (GRACE): NUMBA=${NUMBA_CACHE_DIR}, XDG=${XDG_CACHE_HOME}"
    fi

    banner "Viscosity pipeline — run info"
    log "Date:        ${START_TIME}"
    log "Run ID:      ${RUN_ID}"
    log "System:      ${SYSTEM_NAME}"
    log "Elements:    ${ELEMENTS_STR}"
    log "Masses:      ${MASSES_STR}"
    log "Radii:       ${RADII_STR}"
    log "Mode:        ${MODE}"
    log "Model:       ${MODEL_FULL_PATH}"
    log "DFT method:  ${DFT_METHOD:-<none>}"
    log "T measure:   ${T_MEASURE} K"
    log "Workdir:     ${WORKDIR_BASE}"
    log "Log dir:     ${RUN_LOG_DIR}"
    hr
}

# ------------------------------------------------------------
# Запуск одного шага
# ------------------------------------------------------------
run_step() {
    local step_name="$1"; shift
    local step_script="${PIPELINE_DIR}/steps/${step_name}.sh"
    local step_log="${RUN_LOG_DIR}/${step_name}.log"
    local step_workdir="${WORKDIR_BASE}/${step_name}"

    # --- Per-step поведение: env override > глобальная env > словарь ---
    local fc_var="FORCE_CLEAR_${step_name}"
    local coe_var="CONTINUE_ON_ERROR_${step_name}"
    local skip_var="SKIP_${step_name}"

    local fc="${!fc_var:-${FORCE_CLEAR:-${STEP_FORCE_CLEAR[$step_name]:-false}}}"
    local coe="${!coe_var:-${CONTINUE_ON_ERROR:-${STEP_CONTINUE_ON_ERROR[$step_name]:-false}}}"
    local skip="${!skip_var:-${STEP_SKIP[$step_name]:-false}}"

    # --- SKIP ---
    if [ "$skip" = "true" ]; then
        log ">>> ${step_name}  (SKIPPED by config)"
        return 0
    fi

    if [ ! -f "$step_script" ]; then
        log "ERROR: step script not found: ${step_script}"
        return 1
    fi

    mkdir -p "$step_workdir"

    log ">>> ${step_name}  (workdir: ${step_workdir})"
    local t0; t0=$(date +%s)

    export ROOT_DIR PIPELINE_NAME PIPELINE_DIR RUN_ID RUN_LOG_DIR LOG_FILE
    export MODE FAMILY VARIANT USE_TYPE
    export CALCULATOR SYSTEM_NAME ELEMENTS ELEMENTS_STR DFT_METHOD GPU_DEVICE
    export MASSES_STR RADII_STR
    export DEVICE DTYPE GPU_NAME
    export MODEL_PATH MODEL_NAME MODEL_FULL_PATH
    export WORKDIR_BASE T_MEASURE
    export CPU_THREADS
    export FC_THIS_STEP="$fc"
    export COE_THIS_STEP="$coe"
    export STEP_NAME="${step_name}"
    export STEP_WORKDIR="${step_workdir}"

    # Вывод шага — ТОЛЬКО в его лог (не дублируем в master.log)
    bash "$step_script" > "$step_log" 2>&1
    local rc=$?

    local t1; t1=$(date +%s)
    local dt=$((t1 - t0))
    log "<<< ${step_name} done in $(printf '%02d:%02d:%02d' \
        $((dt/3600)) $(((dt%3600)/60)) $((dt%60))) (rc=${rc})"

    return $rc
}

# ------------------------------------------------------------
# Финализация
# ------------------------------------------------------------
finalize_run() {
    local rc="${1:-0}"
    local end_epoch; end_epoch=$(date +%s)
    local total=$((end_epoch - START_EPOCH))

    banner "Viscosity pipeline finished"
    log "Started:  ${START_TIME}"
    log "Ended:    $(date '+%Y-%m-%d %H:%M:%S')"
    log "Elapsed:  $((total/3600))h $(((total%3600)/60))m $((total%60))s"
    log "Exit:     ${rc}"
    log "Log dir:  ${RUN_LOG_DIR}"
    hr

    if [ "$rc" -eq 0 ]; then
        echo "✅ SUCCESS"
    else
        echo "❌ ERROR: exit code ${rc}"
        echo "Check: tail -100 ${LOG_FILE}"
    fi
    return "$rc"
}