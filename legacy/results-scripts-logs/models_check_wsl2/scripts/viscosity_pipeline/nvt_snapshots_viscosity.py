#models_check/scripts/viscosity_pipeline/nvt_snapshots_viscosity.py
"""
nvt_snapshots_viscosity.py

Шаг 2 пайплайна viscosity: NVT-прогон с записью снимков.

Для каждой концентрации c_Ti из [0.0, 0.1, ..., 1.0]:
    - читает <prev_step_workdir>/c_Ti_X.XX/data.lammps
    - задаёт deepmd-потенциал
    - NPT-эквилибрация (equil_steps) → data.0
    - NVT-прогон, n_snapshots снимков по run_steps шагов → data.1 ... data.50
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from lammps import lammps
import numpy as np

from checkpoints import SnapshotsCheckpoint, compute_snapshots_hash


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
        description="NVT snapshots generation for viscosity pipeline",
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

    # --- D. Состав системы ---
    p.add_argument("--compositions", type=str, default="", help="Comma-separated list of c_Ti values; empty = auto-discover")

    # --- B. Термодинамика ---
    p.add_argument("--T",         type=float, default=1943.0)
    p.add_argument("--P",         type=float, default=1.0)
    p.add_argument("--timestep",  type=float, default=0.001)
    p.add_argument("--seed",      type=int,   default=1)

    # --- C. Протокол ---
    p.add_argument("--equil-steps", type=int, default=5000)
    p.add_argument("--run-steps",   type=int, default=1000)
    p.add_argument("--n-snapshots", type=int, default=50)

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
# Генератор снимков
# ============================================================
class NVTSnapshotsGenerator:

    def __init__(self, args):
        self.args = args

        # --- Пути ---
        self.workdir = Path(args.workdir)
        self.step_workdir = Path(args.step_workdir)
        self.prev_step_workdir = Path(args.prev_step_workdir)

        # --- Force-clear ---
        if args.force_clear:
            force_clear_step_workdir(
                self.step_workdir, self.workdir, expected_name="02_nvt_snapshots"
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
        self.P = args.P
        self.timestep = args.timestep
        self.seed = args.seed
        self.velocity_seed_base = args.seed      # для velocity create в NVT
        self.dump_every = 100                     # шаг дампа NPT

        # --- Протокол ---
        self.equil_steps = args.equil_steps
        self.run_steps = args.run_steps
        self.n_snapshots = args.n_snapshots

        # --- Концентрации ---
        self.compositions = self._build_compositions()

        # --- Checkpoint ---
        self.checkpoint = SnapshotsCheckpoint(self.step_workdir / "checkpoint.json")
        params_hash = compute_snapshots_hash(self._params_for_hash())
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
                data_file = self.prev_step_workdir / case_name / "data.lammps"
                if not data_file.exists():
                    if self.args.continue_on_error:
                        logger.warning(
                            f"Skip {case_name}: step 1 output missing ({data_file})"
                        )
                        continue
                    raise FileNotFoundError(
                        f"--compositions set, but step 1 output missing: {data_file}"
                    )
                comps.append(c_ti)

            return comps

        # Auto-discover
        return self._discover_compositions()

    # ------------------------------------------------------------------
    def _discover_compositions(self):
        """Найти папки c_Ti_X.XX в prev_step_workdir с data.lammps."""
        comps = []
        for d in sorted(self.prev_step_workdir.glob("c_Ti_*")):
            if not d.is_dir():
                continue
            data_file = d / "data.lammps"
            if not data_file.exists():
                logger.warning(f"Skip {d}: no data.lammps")
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
            "P": self.P,
            "timestep": self.timestep,
            "seed": self.seed,
            "equil_steps": self.equil_steps,
            "run_steps": self.run_steps,
            #"n_snapshots": self.n_snapshots,
            "elements": self.elements,
            "masses": self.masses,
        }

    # ------------------------------------------------------------------
    '''
    def _generate_one(self, c_Ti: float, start_from: int = 0):
        """
        start_from = 0  — полный протокол (NPT → data.0 → NVT → data.1..N)
        start_from = N  — продолжить с data.N → NVT → data.(N+1)..N
        """
        c_Al = 1.0 - c_Ti
        case_name = f"c_Ti_{c_Ti:.2f}"

        # Вход (шаг 1) — только для полного протокола
        input_dir = self.prev_step_workdir / case_name
        input_data = input_dir / "data.lammps"

        # Выход
        output_dir = self.step_workdir / case_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info(f"c_Ti = {c_Ti:.2f}   c_Al = {c_Al:.2f}")
        logger.info(f"  output: {output_dir}")
        logger.info(f"  start_from = {start_from} ({'fresh' if start_from == 0 else f'extend from data.{start_from}'})")
        logger.info(f"  T = {self.T} K, P = {self.P} bar")
        logger.info(f"  equil_steps = {self.equil_steps}")
        logger.info(f"  target n_snapshots = {self.n_snapshots}")
        logger.info("=" * 60)

        L = lammps()
        try:
            if start_from == 0:
                # === Полный протокол ===
                if not input_data.exists():
                    raise FileNotFoundError(f"Input data not found: {input_data}")

                L.commands_string(f"""
                    units metal
                    boundary p p p
                    atom_style atomic
                    atom_modify map yes
                    read_data {input_data} nocoeff
                """)
            else:
                # === Продолжение с data.{start_from} ===
                continue_data = output_dir / f"data.{start_from}"
                if not continue_data.exists():
                    raise FileNotFoundError(
                        f"Cannot extend: {continue_data} not found"
                    )

                L.commands_string(f"""
                    units metal
                    boundary p p p
                    atom_style atomic
                    atom_modify map yes
                    read_data {continue_data} nocoeff
                """)

            # --- Массы (одинаково для обоих режимов) ---
            L.commands_string(f"""
                mass 1 {self.masses[0]}  # {self.elements[0]}
                mass 2 {self.masses[1]}  # {self.elements[1]}
            """)

            # --- Потенциал ---
            L.commands_string(f"""
                pair_style deepmd {self.model_full_path}
                pair_coeff * * {self.elements_str}
            """)

            # --- Timestep / thermo ---
            L.commands_string(f"""
                timestep {self.timestep}
                thermo_style custom step temp press pe vol enthalpy density
                thermo 100
            """)

            if start_from == 0:
                # --- NPT-эквилибрация + data.0 ---
                L.commands_string(f"""
                    velocity all create {self.T} {self.seed}
                    fix int all npt temp {self.T} {self.T} 0.1 iso {self.P} {self.P} 0.5
                    run {self.equil_steps}
                    write_data {output_dir}/data.0
                    unfix int
                    fix int all nvt temp {self.T} {self.T} 0.1
                """)
                first_new = 1
            else:
                # --- Сразу NVT ---
                L.commands_string(f"""
                    fix int all nvt temp {self.T} {self.T} 0.1
                """)
                first_new = start_from + 1

            # --- Снимки ---
            for a in range(first_new, self.n_snapshots + 1):
                L.commands_string(f"""
                    run {self.run_steps}
                    write_data {output_dir}/data.{a}
                """)

        finally:
            L.close()

        first_snapshot = output_dir / "data.0"
        last_snapshot = output_dir / f"data.{self.n_snapshots}"

        logger.info(f"  -> {first_snapshot} ... {last_snapshot}")
        return input_data, output_dir, first_snapshot, last_snapshot
    '''
    def _generate_one(self, c_Ti: float, old_chosen_ts: list, completed: list):
        """
        Новый протокол:
          1. NPT + dump (если npt.lammpstrj ещё нет)
          2. Выбор случайных ts (или дополнение existing)
          3. NVT для новых ts (тот же LAMMPS-инстанс)

        old_chosen_ts — уже выбранные timestep'ы (из чекпойнта)
        completed     — список готовых 1-based номеров (SKIP)
        """
        c_Al = 1.0 - c_Ti
        case_name = f"c_Ti_{c_Ti:.2f}"

        input_dir = self.prev_step_workdir / case_name
        input_data = input_dir / "data.lammps"

        output_dir = self.step_workdir / case_name
        output_dir.mkdir(parents=True, exist_ok=True)

        npt_dump_file = output_dir / "npt.lammpstrj"

        logger.info("=" * 60)
        logger.info(f"c_Ti = {c_Ti:.2f}   c_Al = {c_Al:.2f}")
        logger.info(f"  output:         {output_dir}")
        logger.info(f"  npt_dump_file:  {npt_dump_file}")
        logger.info(f"  T = {self.T} K, P = {self.P} bar")
        logger.info(f"  equil_steps = {self.equil_steps}, dump_every = {self.dump_every}")
        logger.info(f"  run_steps = {self.run_steps}, n_snapshots = {self.n_snapshots}")
        logger.info(f"  old_chosen_ts = {len(old_chosen_ts)}")
        logger.info(f"  completed     = {len(completed)}")
        logger.info("=" * 60)

        L = lammps()
        try:
            # ==================
            # Этап 1: NPT + dump (если нет)
            # ==================
            if not npt_dump_file.exists():
                if not input_data.exists():
                    raise FileNotFoundError(f"Input data not found: {input_data}")

                logger.info("  NPT + dump (fresh)")
                L.commands_string(f"""
                    units metal
                    boundary p p p
                    atom_style atomic
                    atom_modify map yes
                    read_data {input_data} nocoeff
                    mass 1 {self.masses[0]}  # {self.elements[0]}
                    mass 2 {self.masses[1]}  # {self.elements[1]}
                    pair_style deepmd {self.model_full_path}
                    pair_coeff * * {self.elements_str}
                    timestep {self.timestep}
                    thermo 100

                    velocity all create {self.T} {self.seed}
                    fix npt all npt temp {self.T} {self.T} 0.1 iso {self.P} {self.P} 0.5
                    dump npt_dump all custom {self.dump_every} {npt_dump_file} id type x y z vx vy vz
                    dump_modify npt_dump sort id
                    run {self.equil_steps}
                    undump npt_dump
                    unfix npt
                """)
            else:
                if not input_data.exists():
                    raise FileNotFoundError(f"Input data not found: {input_data}")

                logger.info(f"  NPT dump exists, reusing: {npt_dump_file}")
                L.commands_string(f"""
                    units metal
                    boundary p p p
                    atom_style atomic
                    atom_modify map yes
                    read_data {input_data} nocoeff
                    mass 1 {self.masses[0]}  # {self.elements[0]}
                    mass 2 {self.masses[1]}  # {self.elements[1]}
                    pair_style deepmd {self.model_full_path}
                    pair_coeff * * {self.elements_str}
                    timestep {self.timestep}
                    thermo 100
                """)

            # ==================
            # Этап 2: выбор ts
            # ==================
            available_ts = list(range(self.dump_every, self.equil_steps + 1, self.dump_every))

            n_old = len(old_chosen_ts)
            n_new = self.n_snapshots - n_old

            if n_new > 0:
                candidates = [ts for ts in available_ts if ts not in old_chosen_ts]
                if len(candidates) < n_new:
                    raise RuntimeError(
                        f"Not enough ts: need {n_new}, have {len(candidates)} "
                        f"(equil_steps={self.equil_steps}, dump_every={self.dump_every})"
                    )
                rng = np.random.default_rng(self.seed)
                new_ts = rng.choice(candidates, size=n_new, replace=False)
                chosen_ts = sorted([int(x) for x in old_chosen_ts] + [int(x) for x in new_ts])
            else:
                chosen_ts = sorted([int(x) for x in old_chosen_ts])

            logger.info(f"  chosen_ts ({len(chosen_ts)}): {chosen_ts}")

            # ==================
            # Этап 3: NVT для новых
            # ==================
            for i, ts in enumerate(chosen_ts):
                idx = i + 1   # 1-based
                if idx in completed:
                    logger.info(f"  [{idx}/{len(chosen_ts)}] ts={ts} SKIP")
                    continue

                logger.info(f"  [{idx}/{len(chosen_ts)}] ts={ts} NVT ({self.run_steps} steps)")
                L.commands_string(f"""
                    read_dump {npt_dump_file} {ts} x y z vx vy vz box yes
                    velocity all create {self.T} {self.seed + i}
                    fix nvt_{idx} all nvt temp {self.T} {self.T} 0.1
                    run {self.run_steps}
                    write_data {output_dir}/data.{idx}
                    unfix nvt_{idx}
                """)

        finally:
            L.close()

        first_snapshot = output_dir / "data.1"
        last_snapshot = output_dir / f"data.{self.n_snapshots}"

        return input_data, output_dir, first_snapshot, last_snapshot, chosen_ts, npt_dump_file

    # ------------------------------------------------------------------
    '''
    def run(self):
        logger.info("=" * 60)
        logger.info("NVT snapshots generation")
        logger.info("=" * 60)
        logger.info(f"  workdir           = {self.workdir}")
        logger.info(f"  step_workdir      = {self.step_workdir}")
        logger.info(f"  prev_step_workdir = {self.prev_step_workdir}")
        logger.info(f"  model             = {self.model_full_path}")
        logger.info(f"  elements          = {self.elements}")
        logger.info(f"  masses            = {self.masses}")
        logger.info(f"  T                 = {self.T}")
        logger.info(f"  P                 = {self.P}")
        logger.info(f"  timestep          = {self.timestep}")
        logger.info(f"  seed              = {self.seed}")
        logger.info(f"  equil_steps       = {self.equil_steps}")
        logger.info(f"  run_steps         = {self.run_steps}")
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
            if entry and entry.get("status") == "completed":
                max_done = max(entry.get("completed_viscs") or
                               [entry.get("n_snapshots", 1) - 1])
                logger.info(f"{case_name}: EXTEND from data.{max_done} to data.{self.n_snapshots}")
                start_from = max_done
            else:
                logger.info(f"{case_name}: FRESH start")
                start_from = 0

            try:
                input_data, output_dir, first_snap, last_snap = self._generate_one(
                    c_Ti, start_from=start_from
                )
                self.checkpoint.mark_completed(
                    case_name=case_name,
                    c_Ti=c_Ti,
                    input_data=input_data,
                    output_dir=output_dir,
                    n_snapshots=self.n_snapshots + 1,   # data.0 + data.1..N
                    completed_viscs=list(range(1, self.n_snapshots + 1)),
                    first_snapshot=first_snap,
                    last_snapshot=last_snap,
                    T=self.T,
                    P=self.P,
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
    '''
    def run(self):
        logger.info("=" * 60)
        logger.info("NVT snapshots generation (NPT-dump + resample protocol)")
        logger.info("=" * 60)
        logger.info(f"  workdir           = {self.workdir}")
        logger.info(f"  step_workdir      = {self.step_workdir}")
        logger.info(f"  prev_step_workdir = {self.prev_step_workdir}")
        logger.info(f"  model             = {self.model_full_path}")
        logger.info(f"  elements          = {self.elements}")
        logger.info(f"  masses            = {self.masses}")
        logger.info(f"  T                 = {self.T}")
        logger.info(f"  P                 = {self.P}")
        logger.info(f"  timestep          = {self.timestep}")
        logger.info(f"  seed              = {self.seed}")
        logger.info(f"  equil_steps       = {self.equil_steps}")
        logger.info(f"  dump_every        = {self.dump_every}")
        logger.info(f"  run_steps         = {self.run_steps}")
        logger.info(f"  n_snapshots       = {self.n_snapshots}")
        logger.info(f"  compositions      = {self.compositions}")
        logger.info(f"  continue_on_error = {self.args.continue_on_error}")
        logger.info("=" * 60)

        for c_Ti in self.compositions:
            case_name = f"c_Ti_{c_Ti:.2f}"

            if self.checkpoint.is_completed(case_name, self.n_snapshots, verify_paths=True):
                logger.info(f"{case_name}: SKIP (checkpoint, verified)")
                continue

            entry = self.checkpoint._cases().get(case_name, {})
            completed = entry.get("completed_viscs", [])
            old_chosen_ts = entry.get("chosen_ts", [])

            logger.info(
                f"{case_name}: completed={len(completed)}, "
                f"old_chosen_ts={len(old_chosen_ts)}"
            )

            try:
                input_data, output_dir, first_snap, last_snap, chosen_ts, npt_dump = \
                    self._generate_one(c_Ti, old_chosen_ts, completed)

                all_completed = list(range(1, self.n_snapshots + 1))

                self.checkpoint.mark_completed(
                    case_name=case_name,
                    c_Ti=c_Ti,
                    input_data=input_data,
                    output_dir=output_dir,
                    n_snapshots=self.n_snapshots,
                    completed_viscs=all_completed,
                    chosen_ts=chosen_ts,
                    npt_dump_file=npt_dump,
                    first_snapshot=first_snap,
                    last_snapshot=last_snap,
                    T=self.T,
                    P=self.P,
                )
            except Exception as e:
                logger.error(f"{case_name}: FAILED — {e}")
                self.checkpoint.mark_failed(case_name, c_Ti, str(e))
                if not self.args.continue_on_error:
                    raise

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
        gen = NVTSnapshotsGenerator(args)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 2
    gen.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())