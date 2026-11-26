#test_scripts/test_stacie.py
"""
test_stacie.py

Тестовый скрипт: расчёт вязкости c_Ti_0.55 через STACIE.
Загружает все доступные реплики (visc*/pressure_*.data).
"""

import sys
from pathlib import Path

import numpy as np
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
import matplotlib.pyplot as plt

dft_type = "pbe"
conc = "1.00"
T = 2000

# ============================================================
# Параметры
# ============================================================
CASE_DIR = Path(
    f"/home/katkoff/deepmd_check_pipline/models_check/results/viscosity/"
    f"dpa4/neo/finetune/AlTi/{T}K/{dft_type}/03_stress_autocorr/"
    f"c_Ti_{conc}"
)

OUT_DIR = Path(f"/home/katkoff/deepmd_check_pipline/test_scripts/stacie_result_{dft_type}_{T}_{conc}")

TIMESTEP_PS = 0.001       # ps (шаг LAMMPS)
BLOCK_SIZE = 1            # без block averages пока

# Константы СИ
KB = 1.380649e-23         # Дж/К
ANGSTROM = 1e-10          # м
BAR_TO_PA = 1e5
PA_TO_MPA = 1e3


# ============================================================
# Обнаружение реплик
# ============================================================
def discover_replicas(case_dir: Path) -> list:
    """Найти все visc{i}/ с pressure_{i}.data и mol.data."""
    replicas = []
    for d in sorted(case_dir.glob("visc*")):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("visc"):
            continue
        try:
            i = int(name.replace("visc", ""))
        except ValueError:
            continue
        #if i > 30:
        #    continue    
        pressure = d / f"pressure_{i}.data"
        mol = d / "mol.data"
        if not pressure.exists():
            print(f"  skip {d}: no {pressure.name}")
            continue
        if not mol.exists():
            print(f"  skip {d}: no mol.data")
            continue
        replicas.append((i, pressure, mol))
    return replicas


# ============================================================
# Преобразование 6 компонент → 5 независимых
# ============================================================
def get_indep_paniso(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz):
    """5 независимых анизотропных комбинаций тензора давлений."""
    return np.array([
        (Pxx - 0.5 * Pyy - 0.5 * Pzz) / np.sqrt(3),
        0.5 * Pyy - 0.5 * Pzz,
        Pxy,
        Pxz,
        Pyz,
    ])


# ============================================================
# Загрузка всех реплик
# ============================================================
def load_all_replicas(case_dir: Path, block_size: int = 1):
    """
    Возвращает:
        sequences: (M, N) — 5 * n_replicas независимых последовательностей
        prefactors: (M,) — F = V_m / (kB * T) для каждой реплики
        timestep_s: h в секундах
    """
    replicas = discover_replicas(case_dir)
    n_rep = len(replicas)
    print(f"Found {n_rep} replicas")

    if n_rep == 0:
        raise RuntimeError(f"No valid replicas in {case_dir}")

    seqs_list = []
    pref_list = []
    nstep_ref = None
    h_s = TIMESTEP_PS * block_size * 1e-12

    for i, pressure_file, mol_file in replicas:
        # --- Объём ---
        atoms = read_lammps_data(str(mol_file), atom_style="atomic")
        vol_m3 = atoms.get_volume() * (ANGSTROM ** 3)
        F = vol_m3 / (KB * T)   # 1/Pa

        # --- Давления ---
        data = np.loadtxt(str(pressure_file), skiprows=2)
        # data: (N, 7) = [TimeStep, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz]

        nstep = data.shape[0]
        if nstep_ref is None:
            nstep_ref = nstep
        elif nstep != nstep_ref:
            print(f"  warn: replica {i} has {nstep} steps, expected {nstep_ref}")

        # bar -> Pa
        Pxx = data[:, 1] * BAR_TO_PA
        Pyy = data[:, 2] * BAR_TO_PA
        Pzz = data[:, 3] * BAR_TO_PA
        Pxy = data[:, 4] * BAR_TO_PA
        Pxz = data[:, 5] * BAR_TO_PA
        Pyz = data[:, 6] * BAR_TO_PA

        # 5 независимых комбинаций: (5, N)
        pcomps = get_indep_paniso(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz)

        # Block average (если нужно)
        if block_size > 1:
            
            pcomps = block_average(pcomps, block_size)   # (5, N//B)

        seqs_list.append(pcomps)               # (5, N')
        pref_list.append(np.full(5, F))        # (5,)

        print(f"  replica {i}: {nstep} steps, V = {vol_m3:.4e} m^3, F = {F:.4e} 1/Pa")

    # Складываем по репликам → (5*n_rep, N')
    sequences = np.concatenate(seqs_list, axis=0)
    prefactors = np.concatenate(pref_list, axis=0)

    return sequences, prefactors, h_s


# ============================================================
# main
# ============================================================
def main():
    print("=" * 60)
    print("STACIE test — c_Ti_0.55")
    print("=" * 60)
    print(f"Case dir:  {CASE_DIR}")
    print(f"T = {T} K, timestep = {TIMESTEP_PS} ps, block = {BLOCK_SIZE}")
    print()

    # 1. Загрузка
    sequences, prefactors, h_s = load_all_replicas(CASE_DIR, BLOCK_SIZE)

    print()
    print(f"sequences shape: {sequences.shape}")   # (5*n_rep, N)
    print(f"prefactors shape: {prefactors.shape}")
    print(f"timestep: {h_s:.6e} s")
    print()

    # 2. Спектр
    spectrum = compute_spectrum(
        sequences,
        prefactors=prefactors,
        timestep=h_s,
    )

    # 3. Оценка
    uc = UnitConfig(
        acint_fmt=".4f",
        acint_symbol="η",
        acint_unit_str="mPa*s",
        acint_unit=1e-3,          # ← Pa*s → mPa*s
        freq_unit_str="THz",
        freq_unit=1e12,
        time_fmt=".3f",
        time_unit_str="ps",
        time_unit=1e-12,
    )

    result = estimate_acint(spectrum, LorentzModel(), verbose=True, uc=uc)

    # 4. Результат
    eta_pa_s = result.acint
    eta_std_pa_s = result.acint_std
    eta_mpa_s = eta_pa_s * PA_TO_MPA
    eta_std_mpa_s = eta_std_pa_s * PA_TO_MPA

    print()
    print("=" * 60)
    print(f"Viscosity:  {eta_mpa_s:.4f} ± {eta_std_mpa_s:.4f} mPa*s")
    print(f"            {eta_pa_s:.4e} ± {eta_std_pa_s:.4e} Pa*s")
    print("=" * 60)
    print(f"neff:       {result.neff:.1f}")
    print(f"corrtime_int: {result.corrtime_int:.3e} s")
    try:
        print(f"corrtime_exp: {result.corrtime_exp:.3e} s")
    except ValueError:
        print("corrtime_exp: n/a")

    # 5. Графики (сохраняем в PNG)
    out_dir = OUT_DIR
    out_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    plot_fitted_spectrum(ax, uc, result)
    fig.savefig(out_dir / "spectrum.png", dpi=120)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot_extras(axs, uc, result)
    fig.savefig(out_dir / "extras.png", dpi=120)
    plt.close(fig)

    # 5b. Свой summary-график с основными числами
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis("off")

    n_rep = len(discover_replicas(CASE_DIR))
    M = sequences.shape[0]
    N = sequences.shape[1]

    lines = [
        f"STACIE — shear viscosity",
        f"",
        f"Case:  c_Ti = 0.55",
        f"T = {T:.0f} K, block = {BLOCK_SIZE}",
        f"",
        f"η = {eta_mpa_s:.3f} ± {eta_std_mpa_s:.3f} mPa·s",
        f"",
        f"M = {M}  ({n_rep} replicas × 5 components)",
        f"N = {N}  steps  ({N * TIMESTEP_PS * BLOCK_SIZE:.1f} ps)",
        f"neff = {result.neff:.1f}",
    ]
    try:
        lines.append(f"τ_exp = {result.corrtime_exp * 1e12:.3f} ps")
    except ValueError:
        pass

    ax.text(
        0.5, 0.5,
        "\n".join(lines),
        ha="center", va="center",
        fontsize=13, family="monospace",
    )
    fig.savefig(out_dir / "summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())