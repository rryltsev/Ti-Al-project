"""
stacie_pipline/stacie_pipline.py

Расчёт shear viscosity через STACIE по всем комбинациям:
    dft × T × c_Ti

Данные:
    models_check/results/viscosity/dpa4/neo/finetune/{SYSTEM_NAME}/{T}K/{dft}/03_stress_autocorr/c_Ti_{c:.2f}/visc*/{pressure_*.data, mol.data}

Выход:
    stacie/results/{SYSTEM_NAME}/{T}/{dft}/c_Ti_{c:.2f}/
        spectrum.png
        extras.png
        result.json
    stacie/results/summary.json
    stacie/results/summary.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.io.lammpsdata import read_lammps_data

from stacie import (
    UnitConfig,
    compute_spectrum,
    estimate_acint,
    LorentzModel,
    plot_fitted_spectrum,
    plot_extras,
)
from stacie.utils import block_average


# ============================================================
# Константы
# ============================================================
SYSTEM_NAME = "AlTi"
DFT_TYPES = ["pbe", "scan", "sol"]
TEMPERATURES = [1773, 1873,2000]
CONCENTRATIONS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.8, 0.90, 1.00]

TIMESTEP_PS = 0.001
BLOCK_SIZE = 1

# Пути
DATA_ROOT = Path(
    "/home/katkoff/deepmd_check_pipline/models_check/results/viscosity/"
    f"dpa4/neo/finetune/{SYSTEM_NAME}"
)
RESULTS_ROOT = Path("/home/katkoff/deepmd_check_pipline/stacie_pipline/results")

# Константы СИ
KB = 1.380649e-23
ANGSTROM = 1e-10
BAR_TO_PA = 1e5
PA_TO_MPA = 1e3


# ============================================================
# Утилиты
# ============================================================
def discover_replicas(case_dir: Path) -> list:
    """Найти все visc{i}/ с pressure_{i}.data и mol.data."""
    replicas = []
    if not case_dir.exists():
        return replicas
    for d in sorted(case_dir.glob("visc*")):
        if not d.is_dir():
            continue
        try:
            i = int(d.name.replace("visc", ""))
        except ValueError:
            continue
        pressure = d / f"pressure_{i}.data"
        mol = d / "mol.data"
        if not pressure.exists():
            print(f"    skip {d.name}: no {pressure.name}")
            continue
        if not mol.exists():
            print(f"    skip {d.name}: no mol.data")
            continue
        replicas.append((i, pressure, mol))
    return replicas


def get_indep_paniso(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz):
    """5 независимых анизотропных комбинаций тензора давлений."""
    return np.array([
        (Pxx - 0.5 * Pyy - 0.5 * Pzz) / np.sqrt(3),
        0.5 * Pyy - 0.5 * Pzz,
        Pxy,
        Pxz,
        Pyz,
    ])


def load_all_replicas(case_dir: Path, T: float, block_size: int = 1):
    """Возвращает sequences (M, N), prefactors (M,), h_s."""
    replicas = discover_replicas(case_dir)
    n_rep = len(replicas)
    print(f"    replicas: {n_rep}")

    if n_rep == 0:
        return None, None, None

    seqs_list = []
    pref_list = []
    nstep_ref = None
    h_s = TIMESTEP_PS * block_size * 1e-12

    for i, pressure_file, mol_file in replicas:
        atoms = read_lammps_data(str(mol_file), atom_style="atomic")
        vol_m3 = atoms.get_volume() * (ANGSTROM ** 3)
        F = vol_m3 / (KB * T)

        data = np.loadtxt(str(pressure_file), skiprows=2)
        nstep = data.shape[0]
        if nstep_ref is None:
            nstep_ref = nstep
        elif nstep != nstep_ref:
            print(f"    warn: replica {i}: {nstep} steps, expected {nstep_ref}")

        Pxx = data[:, 1] * BAR_TO_PA
        Pyy = data[:, 2] * BAR_TO_PA
        Pzz = data[:, 3] * BAR_TO_PA
        Pxy = data[:, 4] * BAR_TO_PA
        Pxz = data[:, 5] * BAR_TO_PA
        Pyz = data[:, 6] * BAR_TO_PA

        pcomps = get_indep_paniso(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)

        if block_size > 1:
            pcomps = block_average(pcomps, block_size)

        seqs_list.append(pcomps)
        pref_list.append(np.full(5, F))

    sequences = np.concatenate(seqs_list, axis=0)
    prefactors = np.concatenate(pref_list, axis=0)

    return sequences, prefactors, h_s


# ============================================================
# Один прогон STACIE для точки
# ============================================================
def run_one_point(dft: str, T: int, c_Ti: float) -> dict | None:
    case_dir = DATA_ROOT / f"{T}K" / dft / "03_stress_autocorr" / f"c_Ti_{c_Ti:.2f}"
    out_dir = RESULTS_ROOT / SYSTEM_NAME / str(T) / dft / f"c_Ti_{c_Ti:.2f}"

    print(f"  [dft={dft}, T={T}, c_Ti={c_Ti:.2f}]")
    print(f"    case_dir: {case_dir}")

    if not case_dir.exists():
        print(f"    WARNING: case_dir not found, skip")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    sequences, prefactors, h_s = load_all_replicas(case_dir, float(T), BLOCK_SIZE)
    if sequences is None:
        print(f"    WARNING: no valid replicas, skip")
        return None

    M, N = sequences.shape
    print(f"    sequences: {M} × {N}, timestep: {h_s:.3e} s")

    # Спектр
    spectrum = compute_spectrum(
        sequences,
        prefactors=prefactors,
        timestep=h_s,
    )

    # Оценка
    uc = UnitConfig(
        acint_fmt=".4f",
        acint_symbol="η",
        acint_unit_str="mPa*s",
        acint_unit=1e-3,
        freq_unit_str="THz",
        freq_unit=1e12,
        time_fmt=".3f",
        time_unit_str="ps",
        time_unit=1e-12,
    )

    result = estimate_acint(spectrum, LorentzModel(), verbose=False, uc=uc)

    eta_pa_s = result.acint
    eta_std_pa_s = result.acint_std
    eta_mpa_s = eta_pa_s * PA_TO_MPA
    eta_std_mpa_s = eta_std_pa_s * PA_TO_MPA

    print(f"    η = {eta_mpa_s:.4f} ± {eta_std_mpa_s:.4f} mPa·s")
    print(f"    neff = {result.neff:.1f}")

    # τ_exp (может не быть)
    try:
        tau_exp_ps = result.corrtime_exp * 1e12
        tau_exp_std_ps = result.corrtime_exp_std * 1e12
    except ValueError:
        tau_exp_ps = None
        tau_exp_std_ps = None

    tau_int_ps = result.corrtime_int * 1e12

    # Графики
    fig, ax = plt.subplots()
    plot_fitted_spectrum(ax, uc, result)
    fig.savefig(out_dir / "spectrum.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot_extras(axs, uc, result)
    fig.savefig(out_dir / "extras.png", dpi=120)
    plt.close(fig)

    # result.json
    point = {
        "system": SYSTEM_NAME,
        "T": int(T),
        "dft": dft,
        "c_Ti": float(c_Ti),
        "eta_mPa_s": float(eta_mpa_s),
        "eta_std_mPa_s": float(eta_std_mpa_s),
        "eta_pa_s": float(eta_pa_s),
        "eta_std_pa_s": float(eta_std_pa_s),
        "neff": float(result.neff),
        "n_replicas": int(M // 5),
        "n_steps": int(N),
        "timestep_ps": float(TIMESTEP_PS),
        "block_size": int(BLOCK_SIZE),
        "tau_exp_ps": None if tau_exp_ps is None else float(tau_exp_ps),
        "tau_exp_std_ps": None if tau_exp_std_ps is None else float(tau_exp_std_ps),
        "tau_int_ps": float(tau_int_ps),
    }

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(point, f, indent=2, ensure_ascii=False)

    return point


# ============================================================
# main
# ============================================================
def main():
    print("=" * 60)
    print("STACIE — viscosity pipeline")
    print("=" * 60)
    print(f"system:         {SYSTEM_NAME}")
    print(f"DFT types:      {DFT_TYPES}")
    print(f"temperatures:   {TEMPERATURES}")
    print(f"concentrations: {CONCENTRATIONS}")
    print(f"block size:     {BLOCK_SIZE}")
    print("=" * 60)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    all_points = []

    for dft in DFT_TYPES:
        for T in TEMPERATURES:
            for c_Ti in CONCENTRATIONS:
                p = run_one_point(dft, T, c_Ti)
                if p is not None:
                    all_points.append(p)

    print()
    print("=" * 60)
    print(f"Total points computed: {len(all_points)}")
    print("=" * 60)

    # summary.json — вложенная структура: system → T → dft → points
    summary = {"system": SYSTEM_NAME, "temperatures": {}}

    # сгруппировать
    by_T = {}
    for p in all_points:
        by_T.setdefault(p["T"], {}).setdefault(p["dft"], []).append(p)

    for T in sorted(by_T.keys()):
        summary["temperatures"][str(T)] = {}
        for dft in DFT_TYPES:
            if dft not in by_T[T]:
                continue
            points_sorted = sorted(by_T[T][dft], key=lambda x: x["c_Ti"])
            summary["temperatures"][str(T)][dft] = {"points": points_sorted}

    with open(RESULTS_ROOT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # summary.csv — плоско, сортировка T, dft, c_Ti
    if all_points:
        df = pd.DataFrame(all_points)
        df = df[[
            "system", "T", "dft", "c_Ti",
            "eta_mPa_s", "eta_std_mPa_s",
            "eta_pa_s", "eta_std_pa_s",
            "neff", "n_replicas", "n_steps",
            "tau_exp_ps", "tau_exp_std_ps", "tau_int_ps",
        ]]
        # порядок dft — как в DFT_TYPES
        dft_order = {d: i for i, d in enumerate(DFT_TYPES)}
        df["_dft_order"] = df["dft"].map(dft_order)
        df = df.sort_values(["T", "_dft_order", "c_Ti"]).drop(columns=["_dft_order"])
        df = df.reset_index(drop=True)
        df.to_csv(RESULTS_ROOT / "summary.csv", index=False)

    print(f"summary.json -> {RESULTS_ROOT / 'summary.json'}")
    print(f"summary.csv  -> {RESULTS_ROOT / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())