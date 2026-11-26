#!/bin/bash
# ============================================================
# STEP 01 — генерация жидкой структуры (LAMMPS)
# ============================================================
set -euo pipefail

PYTHON_SCRIPT="${PIPELINE_DIR}/generate_liquid_viscosity.py"

STEP_ARGS=(
    # --- A. Общие ---
    --calculator   "${CALCULATOR}"
    --model-path   "${MODEL_PATH}"
    --model-name   "${MODEL_NAME}"
    --system       "${SYSTEM_NAME}"
    --family       "${FAMILY}"
    --variant      "${VARIANT}"
    --use-type     "${USE_TYPE}"
    --dft-method   "${DFT_METHOD}"
    --device       "${DEVICE}"
    --dtype        "${DTYPE}"
    --workdir      "${WORKDIR_BASE}"
    --step-workdir "${STEP_WORKDIR}"

    # --- B. Состав системы ---
    --step         0.2
    --compositions "0.9,0.8,0.7,0.6,0.4,0.3,0.2,0.1"
    --nl           8
    --rho          0.045
    --natmtypes    2
    --elements     "${ELEMENTS_STR}"
    --masses       "${MASSES_STR}"
    --radii        "${RADII_STR}"

    # --- C. Термодинамика / потенциал ---
    --T            "${T_MEASURE}"
    --timestep     0.001
    --enr-mie      0.001
    --seed         42
    --run-steps    100000
)
[ "${FC_THIS_STEP}"  = "true" ] && STEP_ARGS+=( --force-clear )
[ "${COE_THIS_STEP}" = "true" ] && STEP_ARGS+=( --continue-on-error )

source "${PIPELINE_DIR}/lib/run_python.sh"