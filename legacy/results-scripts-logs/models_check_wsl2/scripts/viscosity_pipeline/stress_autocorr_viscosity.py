#models_check/scripts/viscosity_pipeline/stress_autocorr_viscosity.py
"""
stress_autocorr_viscosity.py

Шаг 3 пайплайна viscosity: расчёт SACF (stress autocorrelation function).

Для каждой концентрации c_Ti:
    для i = 1..n_snapshots:
        - читает <prev_step_workdir>/c_Ti_X.XX/data.{i}
        - LAMMPS NVT 20000 шагов при T
        - fix ave/correlate для pxy, pxz, pyz
        - пишет sacf_{i}.data в visc{i}/
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from lammps import lammps

from checkpoints import StressAutocorrCheckpoint, compute_stress_autocorr_hash


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
# Argparse
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Stress autocorrelation (SACF) for viscosity pipeline",
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

    # --- D. Состав ---
    p.add_argument("--compositions", type=str, default="", help="Comma-separated list of c_Ti values; empty = auto-discover")

    # --- B. Термодинамика ---
    p.add_argument("--T",         type=float, default=1943.0)
    p.add_argument("--timestep",  type=float, default=0.001)
    p.add_argument("--Tdamp",     type=float, default=0.2)

    # --- C. SACF ---
    p.add_argument("--run-steps",        type=int, default=20000)
    p.add_argument("--corr-length",      type=int, default=5000)
    p.add_argument("--sample-interval",  type=int, default=1)
    p.add_argument("--n-snapshots",      type=int, default=50)

    # --- D. Состав ---
    p.add_argument("--elements", type=str, default="Al,Ti")
    p.add_argument("--masses",   type=str, default="26.98,47.88")

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
        raise RuntimeError(
            f"Refusing to remove {step_resolved}: not inside {workdir_resolved}"
        )

    if step_resolved.name != expected_name:
        raise RuntimeError(
            f"Refusing to remove {step_resolved}: "
            f"unexpected name (expected '{expected_name}')"
        )

    logger.warning(f"FORCE CLEAR: removing {step_resolved}")
    shutil.rmtree(step_resolved)


# ============================================================
# SACF-генератор
# ============================================================
class StressAutocorrRunner:

    def __init__(self, args):
        self.args = args

        # --- Пути ---
        self.workdir = Path(args.workdir)
        self.step_workdir = Path(args.step_workdir)
        self.prev_step_workdir = Path(args.prev_step_workdir)

        # --- Force-clear ---
        if args.force_clear:
            force_clear_step_workdir(
                self.step_workdir, self.workdir, expected_name="03_stress_autocorr"
            )
        self.step_workdir.mkdir(parents=True, exist_ok=True)

        # --- Модель ---
        self.model_full_path = Path(args.model_path) / args.model_name

        # --- Состав ---
        self.elements = [s.strip() for s in args.elements.split(",")]
        self.masses = [float(s) for s in args.masses.split(",")]
        if len(self.elements) != len(self.masses):
            raise ValueError(
                f"--elements has {len(self.elements)} items, "
                f"--masses has {len(self.masses)} items"
            )
        self.elements_str = " ".join(self.elements)

        # --- Термодинамика ---
        self.T = args.T
        self.timestep = args.timestep
        self.Tdamp = args.Tdamp

        # --- SACF ---
        self.run_steps = args.run_steps
        self.corr_length = args.corr_length
        self.sample_interval = args.sample_interval
        self.n_snapshots = args.n_snapshots

        # --- Концентрации ---
        self.compositions = self._build_compositions()

        # --- Checkpoint ---
        self.checkpoint = StressAutocorrCheckpoint(self.step_workdir / "checkpoint.json")
        params_hash = compute_stress_autocorr_hash(self._params_for_hash())
        self.checkpoint.init_meta(params_hash, self._params_for_hash())

        if not self.checkpoint.check_params_hash(params_hash):
            logger.error("=" * 60)
            logger.error("PARAMS HASH MISMATCH")
            logger.error(f"  saved:   {self.checkpoint.get_params_hash()}")
            logger.error(f"  current: {params_hash}")
            logger.error("  Checkpoint entries are STALE.")
            logger.error("  Use --force-clear to recompute everything.")
            logger.error("=" * 60)
            raise RuntimeError(
                "Params changed. Refusing to reuse old checkpoint. "
                "Use --force-clear to recompute."
            )

    # ------------------------------------------------------------------
    def _build_compositions(self):
        comps_str = self.args.compositions
        if comps_str:
            # Явный список
            comps = []
            for x in comps_str.split(","):
                c_ti = round(float(x.strip()), 4)
                case_name = f"c_Ti_{c_ti:.2f}"
                data_file = self.prev_step_workdir / case_name / "data.1"
                if not data_file.exists():
                    if self.args.continue_on_error:
                        logger.warning(
                            f"Skip {case_name}: step 2 output missing ({data_file})"
                        )
                        continue
                    raise FileNotFoundError(
                        f"--compositions set, but step 2 output missing: {data_file}"
                    )
                comps.append(c_ti)
            return comps

        # Auto-discover
        return self._discover_compositions()

    # ------------------------------------------------------------------
    def _discover_compositions(self):
        comps = []
        for d in sorted(self.prev_step_workdir.glob("c_Ti_*")):
            if not d.is_dir():
                continue
            # Проверяем, что есть хотя бы data.1
            if not (d / "data.1").exists():
                logger.warning(f"Skip {d}: no data.1")
                continue
            try:
                c_ti = float(d.name.replace("c_Ti_", ""))
            except ValueError:
                logger.warning(f"Skip {d}: cannot parse c_Ti")
                continue
            comps.append(c_ti)
        return comps

    def _params_for_hash(self) -> dict:
        return {
            "T": self.T,
            "timestep": self.timestep,
            "Tdamp": self.Tdamp,
            "run_steps": self.run_steps,
            "corr_length": self.corr_length,
            "sample_interval": self.sample_interval,
            #"n_snapshots": self.n_snapshots,
            "elements": self.elements,
            "masses": self.masses,
        }

    # ------------------------------------------------------------------
    def _run_one_snapshot(self, input_data: Path, visc_dir: Path, i: int):
        """Один SACF-прогон для снимка i."""
        visc_dir.mkdir(parents=True, exist_ok=True)

        mol_data = visc_dir / "mol.data"
        shutil.copy(input_data, mol_data)

        sacf_file = visc_dir / f"sacf_{i}.data"
        pressure_file = visc_dir / f"pressure_{i}.data"

        L = lammps()
        try:
            L.commands_string(f"""
                units metal
                atom_style atomic
                boundary p p p
                atom_modify map yes
                read_data {mol_data} nocoeff
            """)

            L.commands_string(f"""
                mass 1 {self.masses[0]}  # {self.elements[0]}
                mass 2 {self.masses[1]}  # {self.elements[1]}
            """)

            L.commands_string(f"""
                pair_style deepmd {self.model_full_path}
                pair_coeff * * {self.elements_str}
            """)

            L.commands_string(f"""
                timestep {self.timestep}
                fix NVT all nvt temp {self.T} {self.T} {self.Tdamp}
            """)

            L.commands_string(f"""
                variable p equal {self.corr_length}
                variable s equal {self.sample_interval}
                variable d equal {self.run_steps}
                variable pxx equal pxx
                variable pyy equal pyy
                variable pzz equal pzz
                variable pxy equal pxy
                variable pxz equal pxz
                variable pyz equal pyz
            """)

            # SACF для TDM (Green-Kubo через усреднённые корреляции)
            L.commands_string(f"""
                fix SS all ave/correlate $s $p $d &
                        v_pxy v_pxz v_pyz &
                        type auto file {sacf_file} ave running overwrite
            """)

            # Сырые временные ряды тензора давлений для KUTE/STACIE            
            L.commands_string(f"""
                fix PP all ave/time 1 1 1 &
                        v_pxx v_pyy v_pzz v_pxy v_pxz v_pyz &
                        file {pressure_file}
            """)

            L.commands_string(f"""
                thermo 100
                run {self.run_steps}
            """)
        finally:
            L.close()

        if not sacf_file.exists():
            raise RuntimeError(f"SACF file not created: {sacf_file}")

        return sacf_file

    # ------------------------------------------------------------------
    def _run_one_composition(self, c_Ti: float, start_from: int = 0):
        """
        start_from = 0  — начать с data.1
        start_from = N  — продолжить с data.(N+1)
        """
        case_name = f"c_Ti_{c_Ti:.2f}"

        input_dir = self.prev_step_workdir / case_name
        output_dir = self.step_workdir / case_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"{case_name}: SACF run")
        logger.info(f"  input_dir:  {input_dir}")
        logger.info(f"  output_dir: {output_dir}")
        logger.info(f"  T = {self.T} K, Tdamp = {self.Tdamp}")
        logger.info(f"  run_steps = {self.run_steps}, corr_length = {self.corr_length}")
        logger.info(f"  start_from = {start_from} ({'fresh' if start_from == 0 else f'extend from visc{start_from}'})")
        logger.info(f"  target n_snapshots = {self.n_snapshots}")
        logger.info("=" * 60)

        # Готовые из прошлых запусков + новые
        completed_viscs = list(range(1, start_from + 1))

        for i in range(start_from + 1, self.n_snapshots + 1):
            input_data = input_dir / f"data.{i}"
            if not input_data.exists():
                raise FileNotFoundError(
                    f"Input snapshot not found: {input_data}. "
                    f"Step 2 may not have enough snapshots: "
                    f"requested n_snapshots={self.n_snapshots}, "
                    f"but step 2 provides fewer."
                )

            visc_dir = output_dir / f"visc{i}"

            logger.info(f"  [{i}/{self.n_snapshots}] {input_data} -> {visc_dir}")
            self._run_one_snapshot(input_data, visc_dir, i)
            completed_viscs.append(i)
            logger.info(f"  [{i}/{self.n_snapshots}] done")

        return input_dir, output_dir, completed_viscs

    # ------------------------------------------------------------------
    def run(self):
        logger.info("=" * 60)
        logger.info("Stress autocorrelation (SACF) generation")
        logger.info("=" * 60)
        logger.info(f"  workdir           = {self.workdir}")
        logger.info(f"  step_workdir      = {self.step_workdir}")
        logger.info(f"  prev_step_workdir = {self.prev_step_workdir}")
        logger.info(f"  model             = {self.model_full_path}")
        logger.info(f"  elements          = {self.elements}")
        logger.info(f"  masses            = {self.masses}")
        logger.info(f"  T                 = {self.T}")
        logger.info(f"  timestep          = {self.timestep}")
        logger.info(f"  Tdamp             = {self.Tdamp}")
        logger.info(f"  run_steps         = {self.run_steps}")
        logger.info(f"  corr_length       = {self.corr_length}")
        logger.info(f"  sample_interval   = {self.sample_interval}")
        logger.info(f"  n_snapshots       = {self.n_snapshots}")
        logger.info(f"  compositions      = {self.compositions}")
        logger.info(f"  continue_on_error = {self.args.continue_on_error}")
        logger.info("=" * 60)

        for c_Ti in self.compositions:
            case_name = f"c_Ti_{c_Ti:.2f}"

            if self.checkpoint.is_completed(case_name, self.n_snapshots, verify_paths=True):
                logger.info(f"{case_name}: SKIP (checkpoint, verified)")
                continue

            # Определить, откуда продолжить
            entry = self.checkpoint._cases().get(case_name)
            if entry:
                completed = self.checkpoint._recover_completed_viscs(entry)
                max_done = max(completed) if completed else 0
                if max_done > 0:
                    logger.info(f"{case_name}: EXTEND from visc{max_done} to visc{self.n_snapshots}")
                    start_from = max_done
                else:
                    logger.info(f"{case_name}: FRESH start")
                    start_from = 0
            else:
                logger.info(f"{case_name}: FRESH start")
                start_from = 0

            try:
                input_dir, output_dir, completed_viscs = self._run_one_composition(
                    c_Ti, start_from=start_from
                )
                self.checkpoint.mark_completed(
                    case_name=case_name,
                    c_Ti=c_Ti,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    n_snapshots=self.n_snapshots,
                    completed_viscs=completed_viscs,
                    T=self.T,
                )
            except Exception as e:
                logger.error(f"{case_name}: FAILED — {e}")
                self.checkpoint.mark_failed(case_name, c_Ti, str(e))
                if not self.args.continue_on_error:
                    raise

        # --- Финальный отчёт ---
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
        runner = StressAutocorrRunner(args)
        runner.run()
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())