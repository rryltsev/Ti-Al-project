"""LammpsTemperatureScanManager: Orchestrate temperature scan with caching."""

import logging
from pathlib import Path
from utils.lammps_cash_manager import LammpsCacheManager
from .point_result_lammps import PointResultLammps

logger = logging.getLogger(__name__)


class LammpsTemperatureScanManager:
    """
    Оркестратор температурного скана с кешированием.
    Проходит от высокой T к низкой.

    Каждая точка (per T):
        minimize → equilibrate (at T) → overheat (at T * factor) → cool (T_melt → T) → melt (at T)
    """

    # Ordered list of stages for caching
    STAGE_FILES = ["equil", "overheat", "cool", "melt"]

    def __init__(self, point_runner):
        self.point_runner = point_runner

    def _get_params(self, T_target):
        return {
            "T_target": float(T_target),
            "T_overheat_factor": float(self.point_runner.T_overheat_factor),
            "equil_steps": self.point_runner.equil_steps,
            "heat_steps": self.point_runner.heat_steps,
            "cool_steps": self.point_runner.cool_steps,
            "melt_steps": self.point_runner.melt_steps,
            "mode": self.point_runner.mode,
            "calculator": self.point_runner.calculator,
            "dt": float(self.point_runner.dt),
            "Tdamp": float(self.point_runner.Tdamp),
            "Pdamp": float(self.point_runner.Pdamp),
            "model_path": self.point_runner.model_path,
        }

    def run_scan(self, temperatures, data_path, output_base):
        """
        Run temperature scan.

        Args:
            temperatures: list of temperatures (high → low)
            data_path: path to initial LAMMPS data file (from StructureFactory)
            output_base: base output directory

        Returns:
            List of PointResultLammps for each temperature.
        """

        logger.info(f"\n{'='*60}")
        logger.info(f"TEMPERATURE SCAN: {len(temperatures)} points")
        logger.info(f"  T_max = {temperatures[0]:.0f} K")
        logger.info(f"  T_min = {temperatures[-1]:.0f} K")
        logger.info(f"  Data:  {data_path}")
        logger.info(f"{'='*60}")

        results = []

        for T in temperatures:
            output_dir = output_base / f"T{int(T)}K"
            cache_dir = output_dir / "cache"
            dumps_dir = output_dir / "dumps"
            cache = LammpsCacheManager(cache_dir)

            params = self._get_params(T)
            run_dir, run_id, skip_stages = cache.get_run_dir(params, self.STAGE_FILES)

            # Full cache — skip everything
            if len(skip_stages) == len(self.STAGE_FILES):
                logger.info(f"\n  T = {T:.0f} K: full cache, skipping")

                cached_result = PointResultLammps(T)
                cached_result.equil_path = run_dir / "equil.lammps"
                cached_result.overheat_path = run_dir / "overheat.lammps"
                cached_result.cool_path = run_dir / "cool.lammps"
                cached_result.melt_path = run_dir / "melt.lammps"
                results.append(cached_result)
                continue

            # Determine input data path: last completed stage or original data
            if not skip_stages:                
                input_path = data_path
            else:
                last_stage = skip_stages[-1]
                input_path = run_dir / f"{last_stage}.lammps"

            logger.info(f"\n  T = {T:.0f} K: Run ID = {run_id}")
            logger.info(f"    Skip stages: {skip_stages if skip_stages else '<none>'}")
            logger.info(f"    Input:       {input_path}")

            point_result = self.point_runner.run(
                T_target=T,
                input_path=input_path,
                cache_dir=run_dir,
                dumps_dir=dumps_dir,
                cache=cache,
                run_id=run_id,
                params=params,
                skip_stages=skip_stages,
            )
            results.append(point_result)

        logger.info(f"\n{'='*60}")
        logger.info(f"SCAN COMPLETE: {len(results)}/{len(temperatures)} points done")
        logger.info(f"{'='*60}")

        return results