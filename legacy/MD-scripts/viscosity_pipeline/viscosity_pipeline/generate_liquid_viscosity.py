#!/usr/bin/env python
"""
generate_liquid_viscosity.py

Шаг 1 пайплайна viscosity: генерация жидких структур Al-Ti
через LAMMPS (SC-решётка + случайное украшение + minimize + NVT).
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from lammps import lammps

from checkpoints import GenerationCheckpoint, compute_generation_hash as compute_params_hash


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
        description="Generate liquid Al-Ti structures via LAMMPS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- A. Общие ---
    p.add_argument("--calculator",        type=str, required=True)
    p.add_argument("--model-path",        type=str, required=True)
    p.add_argument("--model-name",        type=str, required=True)
    p.add_argument("--system",            type=str, required=True)
    p.add_argument("--family",            type=str, required=True)
    p.add_argument("--variant",           type=str, required=True)
    p.add_argument("--use-type",          type=str, required=True)
    p.add_argument("--dft-method",        type=str, default="")
    p.add_argument("--device",            type=str, required=True)
    p.add_argument("--dtype",             type=str, required=True)
    p.add_argument("--gpu-name",          type=str, default="")
    p.add_argument("--workdir",           type=str, required=True)
    p.add_argument("--step-workdir",      type=str, required=True)
    p.add_argument("--force-clear",       action="store_true")
    p.add_argument("--continue-on-error", action="store_true")

    # --- B. Состав системы ---
    p.add_argument("--step",       type=float, default=0.1)
    p.add_argument("--nl",         type=int,   default=16)
    p.add_argument("--rho",        type=float, default=0.045)
    p.add_argument("--natmtypes",  type=int,   default=2)
    p.add_argument("--elements",   type=str,   default="Al,Ti")
    p.add_argument("--masses",     type=str,   default="26.98,47.88")
    p.add_argument("--radii",      type=str,   default="1.43,1.47")

    # --- C. Термодинамика / потенциал ---
    p.add_argument("--T",          type=float, default=2000.0)
    p.add_argument("--timestep",   type=float, default=0.001)
    p.add_argument("--enr-mie",    type=float, default=0.001)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--run-steps",  type=int,   default=100000)

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
# Генератор
# ============================================================
class LiquidStructureGenerator:

    def __init__(self, args):
        self.args = args

        # --- Пути ---
        self.workdir = Path(args.workdir)
        self.step_workdir = Path(args.step_workdir)

        # --- Force-clear ---
        if args.force_clear:
            force_clear_step_workdir(
                self.step_workdir, self.workdir, expected_name="01_generate_liquid"
            )
        self.step_workdir.mkdir(parents=True, exist_ok=True)

        # --- Состав ---
        self.elements = [s.strip() for s in args.elements.split(",")]
        self.masses = [float(s) for s in args.masses.split(",")]
        self.radii = [float(s) for s in args.radii.split(",")]

        if len(self.elements) != args.natmtypes:
            raise ValueError(f"--elements has {len(self.elements)} items, "
                             f"but --natmtypes={args.natmtypes}")
        if len(self.masses) != args.natmtypes:
            raise ValueError(f"--masses has {len(self.masses)} items, "
                             f"but --natmtypes={args.natmtypes}")
        if len(self.radii) != args.natmtypes:
            raise ValueError(f"--radii has {len(self.radii)} items, "
                             f"but --natmtypes={args.natmtypes}")

        # --- Система ---
        self.Nl = args.nl
        self.rho = args.rho
        self.Natmtypes = args.natmtypes
        self.T = args.T
        self.timestep = args.timestep
        self.enr_mie = args.enr_mie
        self.seed = args.seed
        self.run_steps = args.run_steps

        # --- Ntot / Lbox / abox ---
        self.Ntot = self.Nl ** 3
        self.Lbox = (self.Ntot / self.rho) ** (1.0 / 3.0)
        self.abox = self.Lbox / self.Nl

        # --- Концентрации ---
        self.compositions = self._build_compositions(args.step)

        # --- Checkpoint ---
        self.checkpoint = GenerationCheckpoint(self.step_workdir / "checkpoint.json")
        params_hash = compute_params_hash(self._params_for_hash())
        self.checkpoint.init_meta(params_hash)

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
    def _params_for_hash(self) -> dict:
        return {
            "nl": self.Nl,
            "rho": self.rho,
            "natmtypes": self.Natmtypes,
            "elements": self.elements,
            "masses": self.masses,
            "radii": self.radii,
            "T": self.T,
            "timestep": self.timestep,
            "enr_mie": self.enr_mie,
            "seed": self.seed,
            "run_steps": self.run_steps,            
        }

    def _build_compositions(self, step: float):
        comps = []
        n = int(round(1.0 / step))
        for k in range(n + 1):
            comps.append(round(k * step, 4))
        return comps

    # ------------------------------------------------------------------
    def _generate_one(self, c_Ti: float) -> Path:
        c_Al = 1.0 - c_Ti

        case_name = f"c_Ti_{c_Ti:.2f}"
        case_dir = self.step_workdir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        data_ini = case_dir / "data_ini.lammps"
        data_min = case_dir / "data_min.lammps"
        data_final = case_dir / "data.lammps"

        logger.info("=" * 60)
        logger.info(f"c_Ti = {c_Ti:.2f}   c_Al = {c_Al:.2f}")
        logger.info(f"  Ntot = {self.Ntot}, Lbox = {self.Lbox:.4f}, abox = {self.abox:.4f}")
        logger.info(f"  case_dir = {case_dir}")
        logger.info("=" * 60)

        L = lammps()
        try:
            L.commands_string(f"""
                units metal
                newton on
                dimension 3
                boundary p p p
                atom_style atomic
            """)

            L.commands_string(f"""
                lattice sc {self.abox}
                region mybox block 0 {self.Nl} 0 {self.Nl} 0 {self.Nl}
                create_box {self.Natmtypes} mybox
            """)

            mass_lines = "\n".join(
                f"mass {i+1} {m}  # {self.elements[i]}"
                for i, m in enumerate(self.masses)
            )
            L.commands_string(mass_lines)

            L.commands_string(f"""
                create_atoms 1 region mybox
                set type 1 type/ratio 2 {c_Ti} {self.seed}
            """)

            L.commands_string(f"""
                neighbor 1.9 multi
                neigh_modify every 1 delay 1 check yes
            """)

            L.command(f"pair_style mie/cut 6.0")
            for i in range(1, self.Natmtypes + 1):
                for j in range(i, self.Natmtypes + 1):
                    rc = self.radii[i - 1] + self.radii[j - 1]
                    sigma = rc * 0.98
                    L.command(
                        f"pair_coeff {i} {j} {self.enr_mie} {sigma} 50.0 49.0 {rc}"
                    )

            L.command("pair_modify shift yes")

            # write_data после pair_style — без WARNING про cutoff
            L.command(f"write_data {data_ini}")

            L.command("compute _rg all gyration")

            L.commands_string(f"""
                neigh_modify every 1 delay 0 check yes
                minimize 1.0e-4 1.0e-6 1000 1000
            """)
            L.command(f"write_data {data_min}")

            L.commands_string(f"""
                neighbor 0.3 bin
                neigh_modify delay 0
                velocity all create {self.T} {self.seed} dist gaussian mom yes
                fix tp all nvt temp {self.T} {self.T} 0.1
                timestep {self.timestep}
                thermo_style custom step pe ke temp press vol
                thermo 500
            """)
            L.command(f"run {self.run_steps}")
            L.command(f"write_data {data_final}")

        finally:
            L.close()

        logger.info(f"  -> {data_final}")
        return data_final

    # ------------------------------------------------------------------
    def run(self):
        logger.info("=" * 60)
        logger.info("Liquid structure generation")
        logger.info("=" * 60)
        logger.info(f"  workdir       = {self.workdir}")
        logger.info(f"  step_workdir  = {self.step_workdir}")
        logger.info(f"  elements      = {self.elements}")
        logger.info(f"  masses        = {self.masses}")
        logger.info(f"  radii         = {self.radii}")
        logger.info(f"  Nl            = {self.Nl}")
        logger.info(f"  rho           = {self.rho}")
        logger.info(f"  Ntot          = {self.Ntot}")
        logger.info(f"  T             = {self.T}")
        logger.info(f"  timestep      = {self.timestep}")
        logger.info(f"  enr_mie       = {self.enr_mie}")
        logger.info(f"  seed          = {self.seed}")
        logger.info(f"  run_steps     = {self.run_steps}")
        logger.info(f"  compositions  = {self.compositions}")
        logger.info(f"  continue_on_error = {self.args.continue_on_error}")
        logger.info("=" * 60)

        for c_Ti in self.compositions:
            case_name = f"c_Ti_{c_Ti:.2f}"

            if self.checkpoint.is_completed(case_name, verify_paths=True):
                logger.info(f"{case_name}: SKIP (checkpoint, verified)")
                continue

            try:
                data_final = self._generate_one(c_Ti)
                self.checkpoint.mark_completed(
                    case_name=case_name,
                    c_Ti=c_Ti,
                    data_path=data_final,
                    n_atoms=self.Ntot,
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
        gen = LiquidStructureGenerator(args)
    except RuntimeError as e:
        # Осмысленные ошибки конфигурации — без трейсбека
        logger.error(str(e))
        return 2
    gen.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())