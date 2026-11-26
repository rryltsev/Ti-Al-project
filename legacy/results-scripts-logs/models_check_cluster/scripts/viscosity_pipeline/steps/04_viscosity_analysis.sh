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
    --workdir           "${WORKDIR_BASE}"
    --step-workdir      "${STEP_WORKDIR}"
    --prev-step-workdir "${WORKDIR_BASE}/03_stress_autocorr"

    # --- B. Термодинамика ---
    --T                 "${T_MEASURE}"

    # --- C. Параметры анализа ---
    --n-snapshots       50

    # --- D. Состав ---
    --compositions      "0.9,0.8,0.7,0.6,0.4,0.3,0.2,0.1"
    --elements          "${ELEMENTS_STR}"
    --masses            "${MASSES_STR}"
)
[ "${FC_THIS_STEP}"  = "true" ] && STEP_ARGS+=( --force-clear )
[ "${COE_THIS_STEP}" = "true" ] && STEP_ARGS+=( --continue-on-error )

source "${PIPELINE_DIR}/lib/run_python.sh"