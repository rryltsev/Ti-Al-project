"""
viscosity_analysis.py

Шаг 4 пайплайна viscosity: расчёт вязкости через Green-Kubo.

Для каждой готовой концентрации c_Ti (по чекпойнту шага 3):
    1. Integrator: интегрирует SACF для каждого из 50 прогонов,
       усредняет → result.txt, result_all.txt
    2. Fitter: фитит двойной экспонентой → асимптота = вязкость
    3. Plotter: рисует η(t) + fit → visc_vs_time.png
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid


# ============================================================
# Логирование
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ============================================================
# Константы
# ============================================================
AMU = 1.66053904e-27       # kg
ANGSTROM = 1e-10           # m
KB = 1.38065e-23           # J/K
DT_FS = 1e-15              # s (шаг 1 фс)
LMP_P_2_PA = 1e10          # bar^2 -> Pa^2
MPA = 1e3                  # Pa -> mPa


# ============================================================
# Argparse
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Green-Kubo viscosity analysis for viscosity pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- A. Общие ---
    p.add_argument("--calculator",         type=str, required=True)
    p.add_argument("--model-path",         type=str, required=True)
    p.add_argument("--model-name",         type=str, required=True)
    p.add_argument("--system",             type=str, required=True)
    p.add_argument("--family",             type=str, required=True)
    p.add_argument("--variant",            type=str, required=True)
    p.add_argument("--use-type",           type=str, required=True)
    p.add_argument("--dft-method",         type=str, default="")
    p.add_argument("--device",             type=str, required=True)
    p.add_argument("--dtype",              type=str, required=True)
    p.add_argument("--gpu-name",           type=str, default="")
    p.add_argument("--workdir",            type=str, required=True)
    p.add_argument("--step-workdir",       type=str, required=True)
    p.add_argument("--prev-step-workdir",  type=str, required=True)
    p.add_argument("--force-clear",        action="store_true")
    p.add_argument("--continue-on-error",  action="store_true")

    # --- B. Термодинамика ---
    p.add_argument("--T", type=float, default=1943.0)

    # --- C. Параметры анализа ---
    p.add_argument("--n-snapshots", type=int, default=50)

    # --- D. Состав ---
    p.add_argument("--elements", type=str, default="Al,Ti")
    p.add_argument("--masses",   type=str, default="26.98,47.88")
    p.add_argument("--compositions", type=str, default="", help="Comma-separated list of c_Ti values; empty = all ready from step 3")

    return p.parse_args()


# ============================================================
# Force-clear
# ============================================================
def force_clear_step_workdir(step_workdir: Path, workdir: Path, expected_name: str) -> None:
    if not step_workdir.exists():
        return

    step_resolved = step_workdir.resolve()
    workdir_resolved = workdir.resolve()

    if workdir_resolved not in step_resolved.parents:
        raise RuntimeError(f"Refusing to remove {step_resolved}: not inside {workdir_resolved}")

    if step_resolved.name != expected_name:
        raise RuntimeError(
            f"Refusing to remove {step_resolved}: unexpected name "
            f"(expected '{expected_name}')"
        )

    logger.warning(f"FORCE CLEAR: removing {step_resolved}")
    shutil.rmtree(step_resolved)


# ============================================================
# Helper: данные о структуре через ASE
# ============================================================
def data_info_read(fname: Path) -> dict:
    from ase.io.lammpsdata import read_lammps_data

    mylammps = read_lammps_data(str(fname), atom_style="atomic")
    masses = mylammps.get_masses()
    mtot = float(sum(masses))
    vol_a3 = mylammps.get_volume()
    vol_m3 = vol_a3 * (ANGSTROM ** 3)
    rho = mtot * AMU / vol_m3

    logger.debug(f"masses_sum = {mtot:.4f} amu")
    logger.debug(f"vol_a3     = {vol_a3:.4f} A^3")
    logger.debug(f"vol_m3     = {vol_m3:.6e} m^3")
    logger.debug(f"rho        = {rho:.3f} kg/m^3")

    return {
        "masses_sum": mtot,
        "vol_a3": vol_a3,
        "vol_m3": vol_m3,
        "rho": rho,
    }


# ============================================================
# Integrator: Green-Kubo
# ============================================================
class Integrator:
    """
    Считает накопленный интеграл Грина-Кубо η(t) для каждого SACF-прогона,
    усредняет по всем прогонам.
    """

    def __init__(self, input_dir: Path, output_dir: Path, T: float,
                 n_snapshots: int, timestep_ps: float):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.T = T
        self.n_snapshots = n_snapshots
        self.timestep_ps = timestep_ps
        self.dt_s = timestep_ps * 1e-12    # ps -> s

        self.rho = None
        self.volume = None
        self.convert = None
    '''
    def _integrate_one(self, num: int):
        """Интеграл η(t) для одного SACF-файла."""
        sacf_path = self.input_dir / f"visc{num}" / f"sacf_{num}.data"
        mol_path = self.input_dir / f"visc{num}" / f"mol.data"
        if not sacf_path.exists():
            raise FileNotFoundError(f"Missing: {sacf_path}")

        data = np.loadtxt(str(sacf_path), unpack=True, skiprows=4)
        # data[0] - Index, data[1] - TimeDelta, data[2] - Ncount
        # data[3] - v_pxy*v_pxy, data[4] - v_pxz*v_pxz, data[5] - v_pyz*v_pyz

        data_info = data_info_read(mol_path)
        rho = data_info.get("rho", None)
        volume = data_info.get("vol_m3", None)
        kT = KB * self.T
        convert = volume / kT * DT_FS * MPA * LMP_P_2_PA

        viscosity = []
        for time_limit in range(1, len(data[1]) + 1):
            val = (
                trapezoid(data[3][0:time_limit], dx=1)
                + trapezoid(data[4][0:time_limit], dx=1)
                + trapezoid(data[5][0:time_limit], dx=1)
            ) / 3.0
            viscosity.append(convert * val)
        return viscosity, rho, volume, convert
    '''
    def _integrate_one(self, num: int):        

        sacf_path = self.input_dir / f"visc{num}" / f"sacf_{num}.data"
        mol_path = self.input_dir / f"visc{num}" / "mol.data"
        if not sacf_path.exists():
            raise FileNotFoundError(f"Missing: {sacf_path}")

        data = np.loadtxt(str(sacf_path), unpack=True, skiprows=4)
        # data[0] - Index, data[1] - TimeDelta, data[2] - Ncount
        # data[3] - v_pxy*v_pxy, data[4] - v_pxz*v_pxz, data[5] - v_pyz*v_pyz

        data_info = data_info_read(mol_path)
        rho = data_info.get("rho", None)
        volume = data_info.get("vol_m3", None)

        # Реальное время лага в секундах
        lag_s = data[1] * self.dt_s

        # Среднее по трём компонентам
        c_bar2 = np.mean(data[3:6], axis=0)

        # Множитель Green-Kubo (без DT_FS, он внутри lag_s)
        convert = volume / (KB * self.T) * LMP_P_2_PA * MPA

        # Векторный кумулятивный интеграл
        integral = cumulative_trapezoid(c_bar2, x=lag_s, initial=0.0)

        viscosity = (convert * integral).tolist()
        return viscosity, rho, volume, convert

    def run(self) -> dict:
        logger.info(f"  Integrating {self.n_snapshots} SACF curves...")

        # Собираем матрицу: [n_t, n_snapshots]
        all_visc = []
        all_params = {"rho": [], "volume": [], "convert": []}
        for num in range(1, self.n_snapshots + 1):
            visc, rho, volume, convert = self._integrate_one(num)
            all_visc.append(visc)
            all_params["rho"].append(rho)
            all_params["volume"].append(volume)
            all_params["convert"].append(convert)
            if num % 10 == 0:
                logger.info(f"    [{num}/{self.n_snapshots}] integrated")

        self.rho = sum(all_params["rho"]) / len(all_params["rho"])
        self.volume = sum(all_params["volume"]) / len(all_params["volume"])
        self.convert = sum(all_params["convert"]) / len(all_params["convert"])
        '''
        matrix = np.array(all_visc).T   # [n_t, n_snapshots]

        # Time axis = TimeDelta = [0, 1, 2, ..., n_t-1]
        time_axis = list(range(len(matrix)))

        # result_all.txt
        result_all_path = self.output_dir / "result_all.txt"
        with open(result_all_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(time_axis):
                row = [t] + list(matrix[i])
                f.write(" ".join(str(x) for x in row) + "\n")

        # result.txt: [t, mean, std, rel_err]
        result_path = self.output_dir / "result.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(time_axis):
                vals = matrix[i]
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1))
                rel_err = std / mean if mean != 0 else 0.0
                f.write(f"{t}\t{mean:.8e}\t{std:.8e}\t{rel_err:.8e}\n")
        '''
        matrix = np.array(all_visc).T   # [n_t, n_snapshots]

        # Реальная ось времени в fs
        n_t = matrix.shape[0]
        time_fs = np.arange(n_t) * self.timestep_ps * 1000.0   # ps -> fs
        # ps * 1000 = fs

        # result_all.txt
        result_all_path = self.output_dir / "result_all.txt"
        with open(result_all_path, "w", encoding="utf-8") as f:
            for i in range(n_t):
                row = [time_fs[i]] + list(matrix[i])
                f.write(" ".join(str(x) for x in row) + "\n")

        # result.txt: [t_fs, mean, std, rel_err]
        result_path = self.output_dir / "result.txt"
        with open(result_path, "w", encoding="utf-8") as f:
            for i in range(n_t):
                vals = matrix[i]
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1))
                rel_err = std / mean if mean != 0 else 0.0
                f.write(f"{time_fs[i]}\t{mean:.8e}\t{std:.8e}\t{rel_err:.8e}\n")

        logger.info(f"  -> {result_path}")
        logger.info(f"  -> {result_all_path}")
        return {
            "result_path": result_path,
            "result_all_path": result_all_path,
            "time_fs": time_fs,
        }


# ============================================================
# Fitter: двойная экспонента
# ============================================================
class Fitter:
    """
    Фитит усреднённую η(t) двойной экспонентой:
        η(t) = A*α*τ₁*(1 - exp(-t/τ₁)) + A*(1-α)*τ₂*(1 - exp(-t/τ₂))
    Асимптота = вязкость.
    """

    def __init__(self, result_path: Path, rho: float):
        self.result_path = result_path
        self.rho = rho
        self.data = None
        self.sigmadata = None
        self.kinematic_convert = 1e-3

    @staticmethod
    def _exp_func(x, A, alpha, tau1, tau2):
        return (
            A * alpha * tau1 * (1.0 - np.exp(-x / tau1))
            + A * (1.0 - alpha) * tau2 * (1.0 - np.exp(-x / tau2))
        )

    @staticmethod
    def _sigma_func(x, A, b):
        return A * (x ** b)

    def _cutoff_left(self) -> int:
        # Исключаем первые 2 ps = 2000 fs из fitting
        fit_left_fs = 2000
        return int(np.searchsorted(self.data[:, 0], fit_left_fs))

    def _cutoff_right(self) -> int:
        sigma = 0.6
        lim = len(self.data)
        start = self._cutoff_left()

        rel_sigma = self.data[:, 3]
        lim_candidates = np.where(rel_sigma[start:] > sigma)[0]

        if len(lim_candidates) == 0:
            logger.warning(
                f"cutoff_right: no point with rel_sigma > {sigma} "
                f"after cutoff_left — using full range [{start}, {lim}]"
            )
            return lim

        end = start + int(lim_candidates[0])

        # Проверка: если окно слишком маленькое — fallback
        min_window = 10
        if end - start < min_window:
            logger.warning(
                f"cutoff_right: fit window too small [{start}, {end}] "
                f"({end - start} points, min={min_window}) — using full range [{start}, {lim}]"
            )
            return lim

        return end

    def _get_dispersion(self, start: int, end: int) -> None:
        xdata = self.data[start:end, 0]
        ydata = self.data[start:end, 2]
        popt, pcov = curve_fit(self._sigma_func, xdata, ydata)
        A = popt[0]
        self.sigmadata = self._sigma_func(xdata, *popt) / A

    def run(self) -> dict:
        self.data = np.loadtxt(str(self.result_path))

        start = self._cutoff_left()
        end = self._cutoff_right()
        logger.info(f"  fit cutoff: [{start}, {end}]")

        # 1. Fit dispersion -> self.sigmadata
        self._get_dispersion(start, end)

        # 2. Fit viscosity with weights
        xdata = self.data[start:end, 0]
        ydata = self.data[start:end, 1]

        popt, pcov = curve_fit(
            self._exp_func, xdata, ydata,
            sigma=self.sigmadata, absolute_sigma=True,
            maxfev=5000,
            bounds=([0., 0.8, 0., 50.], [100., 10., 1000., 1000.]),
        )
        popt_nw, pcov_nw = curve_fit(
            self._exp_func, xdata, ydata,
            maxfev=5000,
            bounds=([0., 0.9, 0., 50.], [100., 10., 1000., 1000.]),
        )

        A, alpha, tau1, tau2 = popt
        # Асимптота при t -> inf (аналитически)
        eta_inf = A * alpha * tau1 + A * (1.0 - alpha) * tau2

        kinematic = eta_inf / self.rho * self.kinematic_convert

        result = {
            "A": float(A),
            "alpha": float(alpha),
            "tau1": float(tau1),
            "tau2": float(tau2),
            "viscosity_mPa_s": float(eta_inf),
            "kinematic_visc_m2_s": float(kinematic),
            "cutoff_right": int(end),
            "A_nw": float(popt_nw[0]),
            "alpha_nw": float(popt_nw[1]),
            "tau1_nw": float(popt_nw[2]),
            "tau2_nw": float(popt_nw[3]),
        }

        logger.info(f"  eta = {eta_inf:.3f} mPa*s")
        logger.info(f"  kinematic = {kinematic:.4e} m^2/s")
        return result


# ============================================================
# Plotter
# ============================================================
class Plotter:

    def __init__(self, result_path: Path, output_path: Path, fit_params: dict,
                 title: str = ""):
        self.result_path = result_path
        self.output_path = output_path
        self.fit_params = fit_params
        self.title = title

    @staticmethod
    def _exp_func(x, A, alpha, tau1, tau2):
        return (
            A * alpha * tau1 * (1.0 - np.exp(-x / tau1))
            + A * (1.0 - alpha) * tau2 * (1.0 - np.exp(-x / tau2))
        )

    def run(self) -> None:
        data = np.loadtxt(str(self.result_path))

        fig, ax = plt.subplots(figsize=(8, 8))

        # Error bars + line
        ax.errorbar(
            data[:, 0], data[:, 1], yerr=data[:, 2],
            fmt="o", markersize=2, lw=1, color="red",
            label="Average", alpha=0.6, errorevery=20,
        )
        ax.plot(data[:, 0], data[:, 1], "-", lw=1.5, color="red", label="Average")

        # Fit
        A = self.fit_params["A"]
        alpha = self.fit_params["alpha"]
        tau1 = self.fit_params["tau1"]
        tau2 = self.fit_params["tau2"]
        eta_inf = self.fit_params["viscosity_mPa_s"]

        xfit = np.linspace(data[0, 0], data[-1, 0], 500)
        yfit = self._exp_func(xfit, A, alpha, tau1, tau2)
        ax.plot(xfit, yfit, "--", lw=2, color="blue",
                label=f"Fit (η∞ = {eta_inf:.3f} mPa·s)")

        # Asymptote
        ax.axhline(eta_inf, color="green", ls=":", lw=1, label="Asymptote")

        ax.set_xlabel("Time [fs]")
        ax.set_ylabel("Green-Kubo visc. integral [mPa·s]")
        ax.set_title(self.title or "Green-Kubo viscosity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(self.output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  -> {self.output_path}")


# ============================================================
# Main runner
# ============================================================
class ViscosityAnalyzer:

    def __init__(self, args):
        self.args = args

        self.workdir = Path(args.workdir)
        self.step_workdir = Path(args.step_workdir)
        self.prev_step_workdir = Path(args.prev_step_workdir)

        # Force-clear
        if args.force_clear:
            force_clear_step_workdir(
                self.step_workdir, self.workdir,
                expected_name="04_viscosity_analysis",
            )
        self.step_workdir.mkdir(parents=True, exist_ok=True)

        # Концентрации
        self.T = args.T
        self.n_snapshots = args.n_snapshots

        # Checkpoint
        from checkpoints import (
            ViscosityAnalysisCheckpoint,
            compute_viscosity_analysis_hash,
            compute_viscosity_analysis_input_hash,
        )
        self.checkpoint = ViscosityAnalysisCheckpoint(self.step_workdir / "checkpoint.json")

        prev_checkpoint_path = self.prev_step_workdir / "checkpoint.json"
        if not prev_checkpoint_path.exists():
            raise RuntimeError(
                f"Step 3 checkpoint not found: {prev_checkpoint_path}. "
                f"Run step 3 first."
            )

        with open(prev_checkpoint_path, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        self._prev_cases = prev_data.get("cases", {})
        self._prev_meta = prev_data.get("_meta", {})

        # timestep из метаданных шага 3
        prev_snapshot = self._prev_meta.get("params_snapshot", {})
        prev_timestep = prev_snapshot.get("timestep")
        if prev_timestep is None:
            raise RuntimeError(
                f"Cannot find timestep in step 3 params_snapshot: {prev_checkpoint_path}"
            )
        self.timestep_ps = float(prev_timestep)
        logger.info(f"  timestep from step 3 = {self.timestep_ps} ps")
        
        params_hash = compute_viscosity_analysis_hash(self._params_for_hash())
        input_hash = compute_viscosity_analysis_input_hash(prev_checkpoint_path)

        self.checkpoint.init_meta(params_hash, input_hash, self._params_for_hash())

        params_ok, input_ok = self.checkpoint.check_hashes(params_hash, input_hash)

        if not params_ok:
            logger.error("=" * 60)
            logger.error("PARAMS HASH MISMATCH")
            logger.error(f"  saved:   {self.checkpoint.get_params_hash()}")
            logger.error(f"  current: {params_hash}")
            logger.error("  Use --force-clear to recompute.")
            logger.error("=" * 60)
            raise RuntimeError("Params changed. Use --force-clear.")

        if not input_ok:
            logger.warning("=" * 60)
            logger.warning("INPUT HASH MISMATCH (step 3 data changed)")
            logger.warning(f"  saved:   {self.checkpoint.get_input_hash()}")
            logger.warning(f"  current: {input_hash}")
            logger.warning("  Old per-concentration results may be stale.")
            logger.warning("  Use --force-clear to recompute.")
            logger.warning("=" * 60)

        # Готовые концентрации (из чекпойнта шага 3)
        self.ready = self._load_ready_compositions(prev_checkpoint_path)

    # ------------------------------------------------------------------
    def _params_for_hash(self) -> dict:
        return {
            "T": self.T,
            "n_snapshots": self.n_snapshots,
        }

    def _load_ready_compositions(self, prev_checkpoint_path: Path) -> list:
        with open(prev_checkpoint_path, "r", encoding="utf-8") as f:
            prev_data = json.load(f)

        cases = prev_data.get("cases", {})

        # Фильтр по --compositions
        requested = None
        if self.args.compositions:
            requested = set()
            for x in self.args.compositions.split(","):
                c_ti = round(float(x.strip()), 4)
                requested.add(f"c_Ti_{c_ti:.2f}")

        ready = []
        skipped = []

        for case_name, entry in cases.items():
            if requested is not None and case_name not in requested:
                continue

            if entry.get("status") != "completed":
                skipped.append((case_name, "status != completed"))
                continue

            n_snap = entry.get("n_snapshots", 0)
            completed = entry.get("completed_viscs", [])
            if len(completed) != n_snap:
                skipped.append((case_name, f"{len(completed)}/{n_snap} viscs"))
                continue

            if len(completed) == 0:
                skipped.append((case_name, "no snapshots done"))
                continue

            output_dir = entry.get("output_dir")
            if not output_dir or not Path(output_dir).exists():
                skipped.append((case_name, "output_dir missing"))
                continue

            ready.append(case_name)

        ready.sort(key=lambda x: float(x.replace("c_Ti_", "")))

        logger.info("=" * 60)
        logger.info("Ready for viscosity analysis:")
        for name in ready:
            logger.info(f"  ✅ {name}")
        if skipped:
            logger.info("Skipped:")
            for name, reason in skipped:
                logger.info(f"  ⏭️  {name}: {reason}")
        logger.info("=" * 60)

        return ready

    # ------------------------------------------------------------------
    def _analyze_one(self, case_name: str):
        c_Ti = float(case_name.replace("c_Ti_", ""))

        input_dir = self.prev_step_workdir / case_name
        output_dir = self.step_workdir / case_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Смотрим, сколько снимков реально готово в шаге 3
        prev_entry = self._prev_cases.get(case_name, {})
        available = len(prev_entry.get("completed_viscs", []))

        # min(requested, available)
        n_use = min(self.n_snapshots, available)
        if self.n_snapshots > available:
            logger.warning(
                f"{case_name}: requested n_snapshots={self.n_snapshots}, "
                f"but only {available} available — using {n_use}"
            )

        logger.info("=" * 60)
        logger.info(f"{case_name}: viscosity analysis")
        logger.info(f"  input_dir:  {input_dir}")
        logger.info(f"  output_dir: {output_dir}")
        logger.info(f"  T = {self.T} K, n_snapshots_used = {n_use} (requested {self.n_snapshots})")
        logger.info("=" * 60)

        # 1. Интегрирование
        integrator = Integrator(
            input_dir=input_dir,
            output_dir=output_dir,
            T=self.T,
            n_snapshots=n_use,
            timestep_ps=self.timestep_ps,
        )
        integrator.run()

        # 2. Фит
        fitter = Fitter(
            result_path=output_dir / "result.txt",
            rho=integrator.rho,
        )
        fit_result = fitter.run()

        # 3. Плот
        plotter = Plotter(
            result_path=output_dir / "result.txt",
            output_path=output_dir / "visc_vs_time.png",
            fit_params=fit_result,
            title=f"Al-Ti, c_Ti = {c_Ti:.2f}, T = {self.T:.0f} K",
        )
        plotter.run()

        # 4. summary.json
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "case_name": case_name,
                "c_Ti": c_Ti,
                "T": self.T,
                "n_snapshots_requested": self.n_snapshots,
                "n_snapshots_used": n_use,
                "volume": integrator.volume,
                "rho_kg_m3": integrator.rho,
                "convert": integrator.convert,
                **fit_result,
            }, f, indent=2)
        logger.info(f"  -> {summary_path}")

        return {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "n_snapshots_used": n_use,
            "viscosity_mPa_s": fit_result["viscosity_mPa_s"],
            "kinematic_visc_m2_s": fit_result["kinematic_visc_m2_s"],
            "fit_params": fit_result,
        }

    # ------------------------------------------------------------------
    def _write_global_summary(self):
        """Сводка по всем посчитанным концентрациям."""
        rows = []
        for case_name in self.ready:
            entry = self.checkpoint._cases().get(case_name)
            if not entry or entry.get("status") != "completed":
                continue
            rows.append({
                "case_name": case_name,
                "c_Ti": entry["c_Ti"],
                "T": entry["T"],
                "viscosity_mPa_s": entry["viscosity_mPa_s"],
                "kinematic_visc_m2_s": entry["kinematic_visc_m2_s"],
            })

        rows.sort(key=lambda r: r["c_Ti"])

        global_path = self.step_workdir / "summary.json"
        with open(global_path, "w", encoding="utf-8") as f:
            json.dump({
                "system": self.args.system,
                "family": self.args.family,
                "variant": self.args.variant,
                "use_type": self.args.use_type,
                "dft_method": self.args.dft_method,
                "T": self.T,
                "n_concentrations": len(rows),
                "concentrations": rows,
            }, f, indent=2)
        logger.info(f"Global summary -> {global_path}")

    # ------------------------------------------------------------------
    def run(self):
        logger.info("=" * 60)
        logger.info("Viscosity analysis (Green-Kubo)")
        logger.info("=" * 60)
        logger.info(f"  T             = {self.T}")
        logger.info(f"  n_snapshots   = {self.n_snapshots}")
        logger.info(f"  continue_on_error = {self.args.continue_on_error}")
        logger.info("=" * 60)

        for case_name in self.ready:
            if self.checkpoint.is_completed(case_name, verify_paths=True):
                logger.info(f"{case_name}: SKIP (already analyzed)")
                continue

            c_Ti = float(case_name.replace("c_Ti_", ""))

            try:
                res = self._analyze_one(case_name)
                self.checkpoint.mark_completed(
                    case_name=case_name,
                    c_Ti=c_Ti,
                    input_dir=res["input_dir"],
                    output_dir=res["output_dir"],
                    n_snapshots_requested=self.n_snapshots,
                    n_snapshots_used=res["n_snapshots_used"],
                    viscosity_mPa_s=res["viscosity_mPa_s"],
                    kinematic_visc_m2_s=res["kinematic_visc_m2_s"],
                    fit_params=res["fit_params"],
                    T=self.T,
                )
            except Exception as e:
                logger.error(f"{case_name}: FAILED — {e}")
                self.checkpoint.mark_failed(case_name, c_Ti, str(e))
                if not self.args.continue_on_error:
                    raise

        # Глобальная сводка
        self._write_global_summary()

        s = self.checkpoint.summary()
        logger.info("=" * 60)
        logger.info(f"Checkpoint summary: {s['completed']}/{s['total']} completed, "
                    f"{s['failed']} failed")
        if s["failed_list"]:
            logger.warning(f"  Failed cases: {s['failed_list']}")
        logger.info("=" * 60)


# ============================================================
# Entry point
# ============================================================
def main():
    args = parse_args()
    try:
        runner = ViscosityAnalyzer(args)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 2
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())