#models_check/scripts/density_thermal_scan.py
"""
Thermal Density Scanner: Generate, relax, equilibrate, and production MD
for solid and liquid phases at multiple temperatures using ML potentials.
"""

import sys
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime
from math import ceil

from ase.io import read
from ase.io import write
from pymatgen.core import Composition
from pymatgen.io.ase import AseAtomsAdaptor

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    CalculatorFactory,
    StructureFactory,
    LiquidStructureManager,
    MDRunner,
    CheckpointManager,
    get_bulk_modulus,
    TrajectoryAnalyzer,
)

import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThermalDensityScanner:
    def __init__(self, args):
        self.args = args

        # System info
        self.system = args.system
        self.n_atoms = args.n_atoms
        self.seed = args.seed

        # Parse phases from JSON
        phases_raw = json.loads(args.phases_json)["phases"]

        self.phases = {}
        for p in phases_raw:
            name = p["name"]
            ptype = p["type"]
            temps = [float(t.strip()) for t in p["temps"].split(',') if t.strip()]
            self.phases[name] = {
                "type": ptype,
                "temperatures": temps,
            }

        # Validate: at least one phase with non-empty temperatures
        if not any(ph["temperatures"] for ph in self.phases.values()):
            raise ValueError("At least one phase must have non-empty temperatures")        

        # Temperature factors (liquid only)
        self.melt_factor = float(args.melt_factor)
        self.equil_factor = float(args.equil_factor)

        # ============================================================
        # PATHS
        # ============================================================
        self.output_base = (
            Path(args.workdir)
            / args.family
            / args.variant
            / args.use_type
            / self.system
        )

        if args.dft_method:
            self.output_base = self.output_base / args.dft_method
            
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

        self.initial_dir = self.output_base / "initial_structures"
        self.initial_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.output_base / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.solid_dir = self.output_base / "solid"
        self.solid_dir.mkdir(parents=True, exist_ok=True)

        self.liquid_dir = self.output_base / "liquid"
        self.liquid_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"System: {self.system}")
        logger.info(f"Parsed {len(self.phases)} phases:")
        for name, ph in self.phases.items():
            logger.info(f"  {name} [{ph['type']}]: {ph['temperatures']}")
        logger.info(f"Melt factor:  {self.melt_factor}")
        logger.info(f"Equil factor: {self.equil_factor}")
        logger.info(f"Output: {self.output_base}")

        # Factories
        self._calculator = None
        self._structure_manager = None
        self._md_runner = None
        self._structure_factory = StructureFactory()

        # Checkpoint
        self.checkpoint = CheckpointManager(self.output_base / "checkpoint.json")

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
    def structure_manager(self):
        if self._structure_manager is None:
            self._structure_manager = LiquidStructureManager(
                output_dir=self.initial_dir,
                n_atoms=self.args.n_atoms,
                r_min=self.args.r_min,
                seed=self.args.seed,
            )
        return self._structure_manager

    @property
    def md_runner(self):
        if self._md_runner is None:
            self._md_runner = MDRunner(
                calculator=self.calculator,
                relax_fmax=self.args.relax_fmax,
                relax_steps=self.args.relax_steps,
                md_timestep=self.args.md_timestep,
                md_ttime=self.args.md_ttime,
                md_pressure=self.args.md_pressure,
                melt_steps=self.args.melt_steps,
                equil_steps=self.args.equil_steps,
                prod_steps=self.args.prod_steps,
                log_interval=self.args.log_interval,
            )
        return self._md_runner

    # ============================================================
    # STRUCTURE PREPARATION
    # ============================================================

    def _get_counts(self) -> Dict[str, int]:
        if "_" in self.system:
            prefix, system = self.system.split("_")
            comp = Composition(system)
        else:    
            comp = Composition(self.system)
        counts = comp.get_el_amt_dict()
        return {k: int(round(v * self.n_atoms)) for k, v in counts.items()}

    def _generate_liquid_structure(self, name) -> Path:
        """Generate and relax liquid structure once."""
        relaxed_cache = self.cache_dir / "liquid_relaxed.traj"

        if "_" in name:
            prefix, system = name.split("_")            
        else:    
            system = name

        if self.checkpoint.is_relax_done(f"{name}") and relaxed_cache.exists():
            logger.info(f"Loading liquid relaxed structure from cache: {relaxed_cache}")
            return relaxed_cache

        logger.info(f"\nGenerating liquid structure for {name} ({self.n_atoms} atoms)...")
        
        elements = self._get_counts().keys()

        if len(elements) == 1:
            struct_path = self.structure_manager.generate_pure(system)
        else:
            struct_path = self.structure_manager.generate_alloy(system)

        logger.info(f"  Structure generated: {struct_path}")

        atoms = read(str(struct_path))
        logger.info(f"Relaxing liquid structure...")
        self.md_runner.relax(atoms)

        write(str(relaxed_cache), atoms, format='traj')
        self.checkpoint.mark_relax_done(f"{name}", str(relaxed_cache))
        logger.info(f"  Relaxed liquid structure saved: {relaxed_cache}")
        return relaxed_cache

    def _generate_solid_structure(self, name) -> Path:
        """Generate and relax solid (crystalline) structure once."""
        relaxed_cache = self.cache_dir / "solid_relaxed.traj"

        if self.checkpoint.is_relax_done(f"{name}") and relaxed_cache.exists():
            logger.info(f"Loading solid relaxed structure from cache: {relaxed_cache}")
            return relaxed_cache

        logger.info(f"\nGenerating solid structure for {name}...")

        # 1. Conventional cell from pymatgen
        structure = self._structure_factory.get_structure(name, cell_type="conventional")
        n_cell = structure.num_sites
        logger.info(f"  Conventional cell: {n_cell} atoms")

        # 2. Compute supercell size to reach ~n_atoms (never less)
        ratio = self.n_atoms / n_cell
        n_repeat = ceil(ratio ** (1.0 / 3.0))
        logger.info(f"  Target n_atoms: {self.n_atoms}, ratio = {ratio:.2f}, repeat = {n_repeat}")

        # 3. Make supercell
        structure.make_supercell([n_repeat, n_repeat, n_repeat])
        n_super = structure.num_sites
        logger.info(f"  Supercell: {n_repeat}x{n_repeat}x{n_repeat} → {n_super} atoms")

        if n_super != self.n_atoms:
            logger.warning(
                f"  Supercell has {n_super} atoms, target was {self.n_atoms} "
                f"(difference: {n_super - self.n_atoms})"
            )

        # 4. pymatgen → ASE
        atoms = AseAtomsAdaptor.get_atoms(structure)
        logger.info(f"  Converted to ASE: {len(atoms)} atoms")

        # 5. Relax
        logger.info(f"Relaxing solid structure...")
        self.md_runner.relax(atoms)

        # 6. Save
        write(str(relaxed_cache), atoms, format='traj')
        self.checkpoint.mark_relax_done(f"{name}", str(relaxed_cache))
        logger.info(f"  Relaxed solid structure saved: {relaxed_cache}")
        return relaxed_cache

    # ============================================================
    # RUN ONE TEMPERATURE
    # ============================================================

    def _run_solid_temperature(self, base_atoms, T_target: float, name: str) -> Dict:
        logger.info("\n" + "="*60)
        logger.info(f"SOLID | T = {T_target} K")
        logger.info("="*60)
        
        if self.checkpoint.is_completed(f"{name}", T_target):
            logger.info(f"  Already completed! Loading from checkpoint...")
            return self.checkpoint.load_result(f"{name}", T_target)

        start_at = datetime.now().isoformat()
        temp_dir = self.solid_dir / f"T_{int(T_target)}K"
        temp_dir.mkdir(parents=True, exist_ok=True)

        counts = self._get_counts()
        if "_" in name:
            prefix, system = name.split("_")
        else:
            system = name

        B = get_bulk_modulus(system, counts)
        logger.info(f"  Bulk modulus = {B:.1f} GPa")

        atoms = base_atoms.copy()
        atoms.calc = self.calculator

        # ---- TIMER START (before MD) ----
        t_start = time.perf_counter()

        # Solid protocol: NO MELT
        logger.info(f"\nStep 1: Equilibrating at {T_target:.0f} K...")
        self.md_runner.equilibrate(atoms, T_target, B)

        traj_path = temp_dir / "trajectory.traj"
        logger.info(f"\nStep 2: Production at {T_target:.0f} K...")
        self.md_runner.production(atoms, T_target, B, traj_path)

        # ---- TIMER STOP ----
        t_end = time.perf_counter()
        wall_time_s = t_end - t_start

        total_steps = self.args.equil_steps + self.args.prod_steps
        steps_per_second = total_steps / wall_time_s if wall_time_s > 0 else 0.0

        logger.info(f"  Wall time:  {wall_time_s:.2f} s")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Speed:      {steps_per_second:.1f} steps/s")

        # ---- Analyze trajectory ----
        result = TrajectoryAnalyzer.average_from_trajectory(traj_path, ["density"])

        # ---- Merge metadata ----
        result.update({
            "phase": "solid",
            "system": name,
            "T_target_K": T_target,
            "bulk_modulus_GPa": B,
            "n_atoms": len(atoms),
            "counts": counts,
            "trajectory_path": str(traj_path),
            "calculator": self.args.calculator,
            "model_type": "-".join([self.args.family, self.args.variant, self.args.use_type]),
            "wall_time_s": round(wall_time_s, 2),
            "total_steps": total_steps,
            "steps_per_second": round(steps_per_second, 1),
            "start_at": start_at,
            "completed_at": datetime.now().isoformat(),
        })

        # ---- Log results ----
        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS: SOLID T = {T_target} K")
        logger.info(f"{'='*60}")
        logger.info(f"  ρ      = {result['density_g_cm3']:.4f} g/cm³")
        logger.info(f"  ρ      = {result['density_kg_m3']:.2f} kg/m³")
        logger.info(f"  ρ      = {result['density_atoms_A3']:.6f} atoms/Å³")
        logger.info(f"  V      = {result['V_average_A3']:.3f} Å³")
        logger.info(f"  T_avg  = {result['T_average_K']:.1f} K")
        logger.info(f"  Frames = {result['n_frames']}")
        logger.info(f"{'='*60}")

        # ---- Save metadata.json ----
        metadata_path = temp_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # ---- Checkpoint ----
        self.checkpoint.mark_completed(f"{name}", T_target, result)

        logger.info(f"\n  Completed SOLID T = {T_target} K")
        return result

    def _run_liquid_temperature(self, base_atoms, T_target: float, name: str) -> Dict:
        T_melt = T_target * self.melt_factor
        T_equil = T_target * self.equil_factor
        T_prod = T_target * self.equil_factor

        logger.info("\n" + "="*60)
        logger.info(f"LIQUID | T = {T_target} K")
        logger.info("="*60)
        logger.info(f"  T_melt  = {T_melt:.0f} K ({self.melt_factor}x)")
        logger.info(f"  T_equil = {T_equil:.0f} K ({self.equil_factor}x)")
        logger.info(f"  T_prod  = {T_prod:.0f} K")        

        if self.checkpoint.is_completed(f"{name}", T_target):
            logger.info(f"  Already completed! Loading from checkpoint...")
            return self.checkpoint.load_result(f"{name}", T_target)

        start_at = datetime.now().isoformat()
        temp_dir = self.liquid_dir / f"T_{int(T_target)}K"
        temp_dir.mkdir(parents=True, exist_ok=True)

        counts = self._get_counts()

        if "_" in name:
            prefix, system = name.split("_")
        else:
            system = name
        B = get_bulk_modulus(system, counts)
        logger.info(f"  Bulk modulus = {B:.1f} GPa")

        atoms = base_atoms.copy()
        atoms.calc = self.calculator

        # ---- TIMER START (before MD) ----
        t_start = time.perf_counter()

        # Liquid protocol: MELT -> EQUILIBRATE -> PROD
        logger.info(f"\nStep 1: Melting at {T_melt:.0f} K...")
        self.md_runner.melt(atoms, T_melt, B)

        logger.info(f"\nStep 2: Equilibrating at {T_equil:.0f} K...")
        self.md_runner.equilibrate(atoms, T_equil, B)

        traj_path = temp_dir / "trajectory.traj"
        logger.info(f"\nStep 3: Production at {T_prod:.0f} K...")
        self.md_runner.production(atoms, T_prod, B, traj_path)

        # ---- TIMER STOP ----
        t_end = time.perf_counter()
        wall_time_s = t_end - t_start

        total_steps = self.args.melt_steps + self.args.equil_steps + self.args.prod_steps
        steps_per_second = total_steps / wall_time_s if wall_time_s > 0 else 0.0

        logger.info(f"  Wall time:  {wall_time_s:.2f} s")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Speed:      {steps_per_second:.1f} steps/s")

        # ---- Analyze trajectory ----
        result = TrajectoryAnalyzer.average_from_trajectory(traj_path, ["density"])

        # ---- Merge metadata ----
        result.update({
            "phase": "liquid",
            "system": name,
            "T_target_K": T_target,
            "T_melt_K": T_melt,
            "T_equil_K": T_equil,
            "T_prod_K": T_prod,
            "bulk_modulus_GPa": B,
            "n_atoms": len(atoms),
            "counts": counts,
            "trajectory_path": str(traj_path),
            "calculator": self.args.calculator,
            "model_type": "-".join([self.args.family, self.args.variant, self.args.use_type]),
            "wall_time_s": round(wall_time_s, 2),
            "total_steps": total_steps,
            "steps_per_second": round(steps_per_second, 1),
            "start_at": start_at,
            "completed_at": datetime.now().isoformat(),
        })

        # ---- Log results ----
        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS: LIQUID T = {T_target} K")
        logger.info(f"{'='*60}")
        logger.info(f"  ρ      = {result['density_g_cm3']:.4f} g/cm³")
        logger.info(f"  ρ      = {result['density_kg_m3']:.2f} kg/m³")
        logger.info(f"  ρ      = {result['density_atoms_A3']:.6f} atoms/Å³")
        logger.info(f"  V      = {result['V_average_A3']:.3f} Å³")
        logger.info(f"  T_avg  = {result['T_average_K']:.1f} K")
        logger.info(f"  Frames = {result['n_frames']}")
        logger.info(f"{'='*60}")

        # ---- Save metadata.json ----
        metadata_path = temp_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # ---- Checkpoint ----
        self.checkpoint.mark_completed(f"{name}", T_target, result)

        logger.info(f"\n  Completed LIQUID T = {T_target} K")
        return result

    # ============================================================
    # RUN
    # ============================================================

    def run(self):
        logger.info("=" * 60)
        logger.info("THERMAL DENSITY SCANNER")
        logger.info("=" * 60)
        logger.info(f"Calculator:   {self.args.calculator}")
        logger.info(f"Mode:         {self.args.family}-{self.args.variant}-{self.args.use_type}")
        logger.info(f"System:       {self.system}")
        logger.info(f"Device:       {self.args.device}, dtype: {self.args.dtype}, GPU: {self.args.gpu_name}")
        logger.info(f"Parsed {len(self.phases)} phases:")
        for name, ph in self.phases.items():
            logger.info(f"  {name} [{ph['type']}]: {ph['temperatures']}")
        logger.info(f"N atoms:      {self.n_atoms}")
        logger.info(f"Output:       {self.output_base}")
        logger.info("=" * 60)

        summary_path = self.output_base / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            logger.info(f"Loaded existing summary: {len(all_results)} entries")
        else:
            all_results = {"solid": {}, "liquid": {}}

        # Ensure sub-dicts
        all_results.setdefault("solid", {})
        all_results.setdefault("liquid", {})

        for name, ph in self.phases.items():
            # --- SOLID ---
            if ph['type'] == "solid":
                solid_path = self._generate_solid_structure(name)
                solid_atoms = read(str(solid_path))
                #solid_atoms.calc = self.calculator

                for T in ph['temperatures']:
                    key = str(T)
                    if key in all_results["solid"]:
                        logger.info(f"Skipping SOLID T = {T} K (already in summary)")
                        continue
                    try:
                        result = self._run_solid_temperature(solid_atoms, T, name)
                        if result:
                            all_results["solid"][key] = result
                            with open(summary_path, 'w', encoding='utf-8') as f:
                                json.dump(all_results, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        logger.error(f"Failed SOLID T = {T} K: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                logger.info("No solid temperatures. Skipping solid protocol.")

            # --- LIQUID ---
            if ph['type'] == "liquid":
                liquid_path = self._generate_liquid_structure(name)
                liquid_atoms = read(str(liquid_path))
                #liquid_atoms.calc = self.calculator

                for T in ph['temperatures']:
                    key = str(T)
                    if key in all_results["liquid"]:
                        logger.info(f"Skipping LIQUID T = {T} K (already in summary)")
                        continue
                    try:
                        result = self._run_liquid_temperature(liquid_atoms, T, name)
                        if result:
                            all_results["liquid"][key] = result
                            with open(summary_path, 'w', encoding='utf-8') as f:
                                json.dump(all_results, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        logger.error(f"Failed LIQUID T = {T} K: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                logger.info("No liquid temperatures. Skipping liquid protocol.")

        logger.info(f"\n{'='*60}")
        logger.info(f"ALL DONE! solid={len(all_results['solid'])}, liquid={len(all_results['liquid'])}")
        logger.info(f"Results: {summary_path}")
        logger.info(f"{'='*60}")

        # Finalization (stub)
        self._finalize()


    # ============================================================
    # FINALIZATION (STUB)
    # ============================================================

    def _finalize(self):
        """
        Collect density scan results from summary.json.
        Writes density_scan.json, density_scan.csv, density_scan.png.
        """
        summary_path = self.output_base / "summary.json"
        if not summary_path.exists():
            logger.warning(f"summary.json not found: {summary_path}. Skipping finalization.")
            return

        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        logger.info(f"\n{'='*60}")
        logger.info("FINALIZING DENSITY SCAN")
        logger.info(f"{'='*60}")

        points = []
        missing = []

        for name, ph in self.phases.items():
            # --- Solid ---
            for T in ph['temperatures']:
                key = str(T)
                data = summary.get(ph["type"], {}).get(key)
                if data is None:
                    missing.append({"phase": ph["type"], "T_target_K": T})
                    continue
                points.append({
                    "phase": ph["type"],
                    "system": self.system,
                    "T_target_K": data["T_target_K"],
                    "density_g_cm3": data["density_g_cm3"],
                    "n_atoms": data["n_atoms"],
                    "calculator": data["calculator"],
                    "model_type": data["model_type"],
                    "total_steps": data["total_steps"],
                    "wall_time_s": data["wall_time_s"],
                    "steps_per_second": data["steps_per_second"],
                })
            '''
            # --- Liquid ---
            for T in ph['temperatures']:
                key = str(T)
                data = summary.get("liquid", {}).get(key)
                if data is None:
                    missing.append({"phase": "liquid", "T_target_K": T})
                    continue
                points.append({
                    "phase": "liquid",
                    "system": self.system,
                    "T_target_K": data["T_target_K"],
                    "density_g_cm3": data["density_g_cm3"],
                    "n_atoms": data["n_atoms"],
                    "calculator": data["calculator"],
                    "model_type": data["model_type"],
                    "total_steps": data["total_steps"],
                    "wall_time_s": data["wall_time_s"],
                    "steps_per_second": data["steps_per_second"],
                })
            '''
        # Sort: solid first (ascending T), then liquid (ascending T)
        points.sort(key=lambda p: (0 if p["phase"] == "solid" else 1, p["T_target_K"]))

        status = "complete" if not missing else "partial"

        # --- Write JSON ---
        final = {
            "metadata": {
                "calculator": self.args.calculator,
                "model_type": "-".join([self.args.family, self.args.variant, self.args.use_type]),
                "system": self.system,
                "seed": self.args.seed,
                "r_min": self.args.r_min,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
            "points": points,
            "n_points": len(points),
            "status": status,
            "missing": missing,
        }

        json_path = self.output_base / "density_scan.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {json_path}")

        # --- Write CSV (pandas, ';') ---
        csv_path = self.output_base / "density_scan.csv"
        df = pd.DataFrame(points)
        df = df[[
            "phase", "system", "T_target_K", "density_g_cm3",
            "n_atoms", "calculator", "model_type",
            "total_steps", "wall_time_s", "steps_per_second",
        ]]
        df.to_csv(csv_path, sep=';', index=False, encoding='utf-8')
        logger.info(f"Saved: {csv_path}")

        # --- Plot ---
        if not points:
            logger.error("No points. Skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(7, 5))

        solid_pts = [p for p in points if p["phase"] == "solid"]
        liquid_pts = [p for p in points if p["phase"] == "liquid"]

        if solid_pts:
            ax.plot(
                [p["T_target_K"] for p in solid_pts],
                [p["density_g_cm3"] for p in solid_pts],
                'o', color='black', markersize=7, label='solid',
            )
        if liquid_pts:
            ax.plot(
                [p["T_target_K"] for p in liquid_pts],
                [p["density_g_cm3"] for p in liquid_pts],
                's', color='crimson', markersize=7, label='liquid',
            )

        ax.set_xlabel("T, K")
        ax.set_ylabel(r"$\rho$, g/cm$^3$")
        ax.set_title(
            f"{final['metadata']['model_type']} | {self.args.system}"
        )
        ax.grid(True, alpha=0.3)
        if solid_pts or liquid_pts:
            ax.legend()

        png_path = self.output_base / "density_scan.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {png_path}")

        logger.info(f"{'='*60}")
        logger.info(f"Density scan: status={status}, points={len(points)}")
        if missing:
            logger.info(f"Missing: {missing}")
        logger.info(f"{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Thermal Density Scanner")

    # Calculator
    parser.add_argument("--calculator", required=True, choices=["grace", "deepmd"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--gpu-name", default="unknown")

    # Mode: <family>-<variant>-<use_type>
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--use-type", required=True, choices=["fit", "finetune", "pretrained"])
    parser.add_argument("--dft-method", default="", help="DFT method: pbe, scan, sol, or '' (flat JSON)")

    # System
    parser.add_argument("--system", required=True, help="System name: Ti47Al47Zr6")
    parser.add_argument("--phases-json", required=True, help='JSON string: {"phases": [{"name": ..., "type": ..., "temps": ...}, ...]}')
    parser.add_argument("--n-atoms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--r-min", type=float, default=2.0)

    # Temperature factors (liquid only)
    parser.add_argument("--melt-factor", type=float, default=1.5)
    parser.add_argument("--equil-factor", type=float, default=1.0)

    # Output
    parser.add_argument("--workdir", default="models_check/results/density")
    parser.add_argument("--force-clear", action="store_true", default=False)

    # MD parameters
    parser.add_argument("--relax-fmax", type=float, default=0.001)
    parser.add_argument("--relax-steps", type=int, default=10000)
    parser.add_argument("--md-timestep", type=float, default=1.0)
    parser.add_argument("--md-ttime", type=float, default=25.0)
    parser.add_argument("--md-pressure", type=float, default=1.0)
    parser.add_argument("--melt-steps", type=int, default=10000)
    parser.add_argument("--equil-steps", type=int, default=30000)
    parser.add_argument("--prod-steps", type=int, default=30000)
    parser.add_argument("--log-interval", type=int, default=100)

    args = parser.parse_args()

    scanner = ThermalDensityScanner(args)
    scanner.run()


if __name__ == "__main__":
    main()