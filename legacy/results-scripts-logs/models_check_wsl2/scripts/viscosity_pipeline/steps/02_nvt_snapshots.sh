#!/bin/bash
# ============================================================
# STEP 02 — NVT snapshots generation
# ============================================================
set -euo pipefail

PYTHON_SCRIPT="${PIPELINE_DIR}/nvt_snapshots_viscosity.py"

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
    --prev-step-workdir "${WORKDIR_BASE}/01_generate_liquid"

    # --- B. Термодинамика ---
    --T           "${T_MEASURE}"
    --P           1.0
    --timestep    0.001
    --seed        1

    # --- C. Протокол ---
    --equil-steps 5000 #5000
    --run-steps   1000 #1000
    --n-snapshots 15   #50

    # --- D. Состав ---
    --compositions "1.00"
    --elements     "${ELEMENTS_STR}"
    --masses       "${MASSES_STR}"
)
[ "${FC_THIS_STEP}"  = "true" ] && STEP_ARGS+=( --force-clear )
[ "${COE_THIS_STEP}" = "true" ] && STEP_ARGS+=( --continue-on-error )

source "${PIPELINE_DIR}/lib/run_python.sh"