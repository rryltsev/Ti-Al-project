#!/bin/bash
# ============================================================
# STEP 04 — Green-Kubo viscosity analysis
# ============================================================
set -euo pipefail

PYTHON_SCRIPT="${PIPELINE_DIR}/viscosity_analysis.py"

STEP_ARGS=(
    # --- A. Общие ---
    --calculator        "${CALCULATOR}"
    --model-path        "${MODEL_PATH}"
    --model-name        "${MODEL_NAME}"
    --system            "${SYSTEM_NAME}"
    --family            "${FAMILY}"
    --variant           "${VARIANT}"
    --use-type          "${USE_TYPE}"
    --dft-method        "${DFT_METHOD}"
    --device            "${DEVICE}"
    --dtype             "${DTYPE}"
    --gpu-name          "${GPU_NAME}"
    --workdir           "${WORKDIR_BASE}"
    --step-workdir      "${STEP_WORKDIR}"
    --prev-step-workdir "${WORKDIR_BASE}/03_stress_autocorr"

    # --- B. Термодинамика ---
    --T                 "${T_MEASURE}"

    # --- C. Параметры анализа ---
    #--tmin              10
    #--tmax              5000
    #--tstep             5
    --n-snapshots       25
    #--fit-right         5000

    # --- D. Состав ---
    --elements          "${ELEMENTS_STR}"
    --masses            "${MASSES_STR}"
)
[ "${FC_THIS_STEP}"  = "true" ] && STEP_ARGS+=( --force-clear )
[ "${COE_THIS_STEP}" = "true" ] && STEP_ARGS+=( --continue-on-error )

source "${PIPELINE_DIR}/lib/run_python.sh"