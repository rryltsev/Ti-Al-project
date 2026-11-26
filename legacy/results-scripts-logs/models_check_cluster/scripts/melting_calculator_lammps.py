"""
Melting Calculator: Determine melting temperature via solid-liquid interface method.
"""

import sys
import json
import logging
import shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    CalculatorFactory,
    StructureFactory,
    LammpsTemperaturePointRunner,
    LammpsTemperatureScanManager,
    InterfaceAnalyzer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeltingCalculator:
    """
    Главный оркестратор расчёта температуры плавления.
    """

    def __init__(self, args):
        self.args = args
        self.system = args.system
        self.T_melt_estimate = float(args.T_melt_estimate)
        self.T_delta = float(args.T_delta)
        self.T_step = float(args.T_step)
        self.T_overheat_factor = float(args.T_overheat_factor)
        self.n_atoms_min = int(args.n_atoms_min)
        self.supercell = tuple(map(int, args.supercell.split(","))) if args.supercell else (1, 1, 3)

        # Output: <workdir>/<engine>/<lammps_mode>/<family>/<variant>/<use_type>/<system>[/<dft_method>]
        parts = [
            args.engine,
            args.lammps_mode,
            args.family,
            args.variant,
            args.use_type,
            args.system,
        ]
        if args.dft_method:
            parts.append(args.dft_method)

        self.output_base = Path(args.workdir) / Path(*parts)

        # Force clear
        if self.args.force_clear and self.output_base.exists():
            workdir_resolved = Path(self.args.workdir).resolve()
            target_resolved = self.output_base.resolve()
            if workdir_resolved not in target_resolved.parents:
                raise RuntimeError(
                    f"Refusing to remove {target_resolved}: not inside {workdir_resolved}"
                )
            logger.warning(f"FORCE CLEAR: removing contents of {self.output_base}")
            for item in self.output_base.iterdir():
                if item.name in (".cache", ".triton", ".numba_cache"):
                    logger.info(f"  Skipping cache dir: {item.name}")
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        self.output_base.mkdir(parents=True, exist_ok=True)

        # Temperature list (high to low)
        T_max = self.T_melt_estimate + self.T_delta
        T_min = self.T_melt_estimate - self.T_delta
        self.temperatures = list(np.arange(T_max, T_min - 1, -self.T_step))

        logger.info(f"MeltingCalculator initialized")
        logger.info(f"  System: {self.system}")
        logger.info(f"  Mode: {args.family}-{args.variant}-{args.use_type}")
        logger.info(f"  DFT method: {args.dft_method or '<none>'}")
        logger.info(f"  Calculator: {args.calculator}")
        logger.info(f"  Engine: {args.engine}")
        logger.info(f"  LAMMPS mode: {args.lammps_mode}")
        logger.info(f"  Device: {args.device}, dtype: {args.dtype}, GPU: {args.gpu_name}")
        logger.info(f"  Force clear: {args.force_clear}")
        logger.info(f"  Tm estimate: {self.T_melt_estimate} K")
        logger.info(f"  T overheat factor: {self.T_overheat_factor}")
        logger.info(f"  Temperature range: {self.temperatures}")
        logger.info(f"  Output: {self.output_base}")

        # Factories
        self._calculator = None
        self._structure_factory = StructureFactory()
        self._scan_manager = None

    @property
    def calculator(self):
        if self._calculator is None:
            factory = CalculatorFactory()
            factory.configure(
                model_path=self.args.model_path,
                model_name=self.args.model_name,
                device='cuda' if self.args.device != 'cpu' else 'cpu',
                dtype=self.args.dtype
            )
            if self.args.calculator == "grace":
                self._calculator = factory.calculator("grace")
            elif self.args.calculator == "deepmd":
                self._calculator = factory.calculator("deepmd")
            else:
                raise ValueError(f"Unknown calculator: {self.args.calculator}")
        return self._calculator

    @property
    def scan_manager(self):
        if self._scan_manager is None:
            point_runner = LammpsTemperaturePointRunner(
                mode=self.args.lammps_mode,
                calculator=self.args.calculator,
                structure_factory=self._structure_factory,
                calculator_obj=self.calculator if self.args.lammps_mode == "callback" else None,
                model_path=f"{self.args.model_path}/{self.args.model_name}",
                dt=self.args.md_timestep,
                Tdamp=self.args.Tdamp_mult * self.args.md_timestep,
                Pdamp=self.args.Pdamp_mult * self.args.Tdamp_mult * self.args.md_timestep,
                T_overheat_factor=self.T_overheat_factor,
                equil_steps=self.args.equil_steps,
                heat_steps=self.args.heat_steps,
                cool_steps=self.args.cool_steps,
                melt_steps=self.args.melt_steps,
                min_tol_energy=self.args.min_tol_energy,
                min_tol_force=self.args.min_tol_force,
                min_max_iter=self.args.min_max_iter,
                min_max_eval=self.args.min_max_eval,
                thermo=self.args.log_interval,
            )
            self._scan_manager = LammpsTemperatureScanManager(
                point_runner=point_runner,
            )
        return self._scan_manager

    def run(self):
        """Main execution."""
        logger.info("=" * 60)
        logger.info("MELTING POINT CALCULATOR")
        logger.info("=" * 60)
        logger.info(f"System:       {self.system}")
        logger.info(f"Tm estimate:  {self.T_melt_estimate} K")
        logger.info(f"T range:      {self.temperatures[-1]:.0f} - {self.temperatures[0]:.0f} K")
        logger.info(f"T step:       {self.T_step} K")
        logger.info(f"T overheat:   x{self.T_overheat_factor}")
        logger.info(f"Output:       {self.output_base}")
        logger.info("=" * 60)

        # Step 1: Create bulk structure + LAMMPS data
        logger.info("\n[1/3] Creating bulk structure...")
        result = self._structure_factory.write_lammps_data(
            self._structure_factory.get_structure(self.system, cell_type="conventional"),
            scaling_matrix=self.supercell,
            target_natoms=self.n_atoms_min,
            directory=str(self.output_base / "data")
        )
        logger.info(f"      Bulk: {result['formula']}, {result['natoms']} atoms")
        logger.info(f"      Data file: {result['data_path']}")

        # Step 2: Temperature scan (minimize + equilibrate + overheat + cool + melt per T)
        logger.info(f"\n[2/3] Temperature scan ({len(self.temperatures)} points)...")
        logger.info(f"      Per T: minimize -> equilibrate -> overheat -> cool -> melt")

        point_results = self.scan_manager.run_scan(
            temperatures=self.temperatures,
            data_path=result["data_path"],
            output_base=self.output_base,
        )
        logger.info(f"      Completed: {len(point_results)}/{len(self.temperatures)} points")

        # Step 3: Analysis (TODO)
        logger.info(f"\n[3/3] Analysis...")
        # TODO: InterfaceAnalyzer — определить Tm по point_results
        logger.info(f"      Analysis not implemented yet")

        logger.info("=" * 60)
        logger.info("MELTING POINT CALCULATOR DONE")
        logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Melting Point Calculator")

    # Calculator
    parser.add_argument("--calculator", required=True, choices=["grace", "deepmd"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--gpu-name", default="unknown")

    # Engine + LAMMPS mode
    parser.add_argument("--engine", default="lammps")
    parser.add_argument("--lammps-mode", default="native", choices=["callback", "native"])

    # Mode: <family>-<variant>-<use_type>
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--use-type", required=True, choices=["fit", "finetune", "pretrained"])
    parser.add_argument("--dft-method", default="", help="DFT method: pbe, scan, sol, or '' (flat JSON)")

    # System
    parser.add_argument("--system", required=True)
    parser.add_argument("--n-atoms-min", type=int, default=2000)
    parser.add_argument("--supercell", default=None, help="nx,ny,nz e.g. 2,2,6")
    parser.add_argument("--T-melt-estimate", required=True, type=float)
    parser.add_argument("--T-delta", type=float, default=200)
    parser.add_argument("--T-step", type=float, default=50)
    parser.add_argument("--T-overheat-factor", type=float, default=1.5)

    # Output
    parser.add_argument("--workdir", default="models_check/results/melting")
    parser.add_argument("--force-clear", action="store_true", default=False)

    # MD parameters
    parser.add_argument("--md-timestep", type=float, default=0.001)
    parser.add_argument("--Tdamp-mult", type=float, default=100)
    parser.add_argument("--Pdamp-mult", type=float, default=10)

    # Steps
    parser.add_argument("--equil-steps", type=int, default=10000)
    parser.add_argument("--heat-steps", type=int, default=25000)
    parser.add_argument("--cool-steps", type=int, default=15000)
    parser.add_argument("--melt-steps", type=int, default=25000)

    # Minimization
    parser.add_argument("--min-tol-energy", type=float, default=1e-10)
    parser.add_argument("--min-tol-force", type=float, default=1e-10)
    parser.add_argument("--min-max-iter", type=int, default=5000)
    parser.add_argument("--min-max-eval", type=int, default=10000)

    # Logging
    parser.add_argument("--log-interval", type=int, default=500)

    args = parser.parse_args()

    calc = MeltingCalculator(args)
    calc.run()


if __name__ == "__main__":
    main()