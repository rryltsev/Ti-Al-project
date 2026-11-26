#!/bin/bash
# ============================================================
# STEP 03 — Stress autocorrelation (SACF)
# ============================================================
set -euo pipefail

PYTHON_SCRIPT="${PIPELINE_DIR}/stress_autocorr_viscosity.py"

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
    --prev-step-workdir "${WORKDIR_BASE}/02_nvt_snapshots"

    # --- B. Термодинамика ---
    --T           "${T_MEASURE}"
    --timestep    0.001
    --Tdamp       0.2

    # --- C. SACF ---
    --run-steps        20000 #20000
    --corr-length      5000  #5000 
    --sample-interval  1
    --n-snapshots      15    #50

    # --- D. Состав ---
    --compositions "1.00"
    --elements     "${ELEMENTS_STR}"
    --masses       "${MASSES_STR}"
)
[ "${FC_THIS_STEP}"  = "true" ] && STEP_ARGS+=( --force-clear )
[ "${COE_THIS_STEP}" = "true" ] && STEP_ARGS+=( --continue-on-error )

source "${PIPELINE_DIR}/lib/run_python.sh"