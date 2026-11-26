#models_check/utils/md_runner.py
"""MDRunner: Relaxation and MD stages for liquid calculations."""

import logging
from pathlib import Path
from typing import Optional
import numpy as np
from ase import Atoms, units
from ase.io import write, read
from ase.md.melchionna import MelchionnaNPT as NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

logger = logging.getLogger(__name__)


class MDRunner:
    """
    Runs relaxation and MD stages: Relax, Melt, Equilibrate, Production.
    
    Usage:
        runner = MDRunner(calculator)
        runner.relax(atoms)
        runner.melt(atoms, T=3000, B=120)
        runner.equilibrate(atoms, T=2000, B=120)
        runner.production(atoms, T=2000, B=120, traj_path="output.traj")
    """
    
    def __init__(
        self,
        calculator,
        relax_fmax: float = 0.001,
        relax_steps: int = 10000,
        md_timestep: float = 1.0,      # fs
        md_ttime: float = 25.0,         # fs
        md_pressure: float = 1.0,       # bar
        melt_steps: int = 10000,
        equil_steps: int = 30000,
        prod_steps: int = 30000,
        log_interval: int = 100,
    ):
        self.calculator = calculator
        self.relax_fmax = relax_fmax
        self.relax_steps = relax_steps
        self.md_timestep = md_timestep
        self.md_ttime = md_ttime
        self.md_pressure = md_pressure
        self.melt_steps = melt_steps
        self.equil_steps = equil_steps
        self.prod_steps = prod_steps
        self.log_interval = log_interval
    
    @staticmethod
    def _upper_triangular_cell(atoms: Atoms) -> None:
        """Convert cell to upper triangular form."""
        rcell, Q = atoms.cell.standard_form("upper")
        atoms.set_cell(rcell, scale_atoms=False)
        atoms.positions = atoms.positions @ Q.T
    
    def relax(self, atoms: Atoms) -> None:
        """Relax atomic positions with variable cell (isotropic)."""
        logger.info("  --- RELAX ---")
        
        self._upper_triangular_cell(atoms)
        atoms.wrap()
        atoms.calc = self.calculator
        
        cell_filter = FrechetCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0])
        dyn = FIRE(cell_filter, loginterval=self.log_interval, logfile=None)
        dyn.run(fmax=self.relax_fmax, steps=self.relax_steps)
        
        atoms.wrap()
        e_pot = atoms.get_potential_energy() / len(atoms)
        logger.info(f"    Relax done, E_pot = {e_pot:.6f} eV/atom")
    
    def _run_npt(
        self,
        atoms: Atoms,
        stage_name: str,
        temperature: float,
        bulk_modulus: float,
        num_steps: int,
        traj_path: Optional[Path] = None,
    ) -> None:
        """Run NPT MD stage."""
        logger.info(f"  --- {stage_name.upper()} ---")
        logger.info(f"    T = {temperature:.1f} K, P = {self.md_pressure} bar, "
                   f"steps = {num_steps}")
        
        self._upper_triangular_cell(atoms)
        atoms.wrap()
        
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, force_temp=True)
        Stationary(atoms)
        
        pfactor = (self.md_ttime ** 2) * bulk_modulus * units.GPa * (units.fs ** 2)
        
        dyn = NPT(
            atoms,
            timestep=self.md_timestep * units.fs,
            temperature_K=temperature,
            externalstress=self.md_pressure * units.bar,
            ttime=self.md_ttime * units.fs,
            pfactor=pfactor,
            loginterval=self.log_interval,
            logfile=None,
            trajectory=str(traj_path) if traj_path else None,
        )
        dyn.set_fraction_traceless(0)
        dyn.run(num_steps)
        
        if traj_path:
            logger.info(f"    Trajectory saved: {traj_path}")
    
    def melt(self, atoms: Atoms, temperature: float, bulk_modulus: float) -> None:
        """Melt stage at high temperature."""
        self._run_npt(atoms, "MELT", temperature, bulk_modulus, self.melt_steps)
    
    def equilibrate(self, atoms: Atoms, temperature: float, bulk_modulus: float) -> None:
        """Equilibration stage at target temperature."""
        self._run_npt(atoms, "EQUIL", temperature, bulk_modulus, self.equil_steps)
    
    def production(
        self,
        atoms: Atoms,
        temperature: float,
        bulk_modulus: float,
        traj_path: Path,
    ) -> None:
        """Production stage with trajectory output."""
        self._run_npt(atoms, "PROD", temperature, bulk_modulus, self.prod_steps, traj_path)