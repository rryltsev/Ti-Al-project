"""TrajectoryAnalyzer: Average properties from MD trajectory."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from ase.io import read
from ase import units

logger = logging.getLogger(__name__)


# Unit conversion constants
AMU_PER_A3_TO_G_CM3 = 1.66053906660
AMU_PER_A3_TO_KG_M3 = 1.66053906660e3


class TrajectoryAnalyzer:
    """Analyze MD trajectory and compute average properties."""

    KNOWN_PROPERTIES = {"energy", "density"}

    @staticmethod
    def average_from_trajectory(
        traj_path: Path,
        properties: Optional[List[str]] = None,
    ) -> Dict:
        """
        Average properties over all frames in trajectory.

        Args:
            traj_path: path to .traj file
            properties: list of property groups to compute.
                        None → ["energy"] (backward compatibility)
                        ["energy"] → energies only
                        ["density"] → densities only
                        ["energy", "density"] → both

        Returns:
            Dict with averaged values. Always includes "T_average_K" and "n_frames".
        """
        # --- Validate properties ---
        if properties is None:
            properties = ["energy"]

        unknown = set(properties) - TrajectoryAnalyzer.KNOWN_PROPERTIES
        if unknown:
            raise ValueError(
                f"Unknown properties: {sorted(unknown)}. "
                f"Known: {sorted(TrajectoryAnalyzer.KNOWN_PROPERTIES)}"
            )

        # --- Read trajectory ---
        frames = read(str(traj_path), index=':')
        n_frames = len(frames)

        if n_frames == 0:
            raise ValueError(f"Empty trajectory: {traj_path}")

        # --- Accumulators ---
        epot_list = []
        etot_list = []
        temp_list = []
        enthalpy_list = []

        density_g_cm3_list = []
        density_kg_m3_list = []
        density_atoms_A3_list = []
        vol_list = []

        mass_total_amu = None

        # --- Loop over frames ---
        for atoms in frames:
            n_atoms = len(atoms)

            # Energy & temperature (always computed, cheap)
            epot = atoms.get_potential_energy() / n_atoms
            ekin = atoms.get_kinetic_energy() / n_atoms
            etot = epot + ekin
            temp = atoms.get_temperature()

            # Stress for enthalpy
            stress = atoms.get_stress(include_ideal_gas=True) / units.GPa
            stress_avg = stress[:3].mean()
            vol = atoms.get_volume() / n_atoms
            enthalpy = (etot + vol * stress_avg * 6.242e-3) * 96.485  # eV/atom → kJ/mol

            epot_list.append(epot)
            etot_list.append(etot)
            temp_list.append(temp)
            enthalpy_list.append(enthalpy)

            # Density (only if requested — avoid useless get_volume calls)
            if "density" in properties:
                vol_A3 = atoms.get_volume()
                mass_amu = atoms.get_masses().sum()
                if mass_total_amu is None:
                    mass_total_amu = mass_amu

                rho_amu_A3 = mass_amu / vol_A3
                density_g_cm3_list.append(rho_amu_A3 * AMU_PER_A3_TO_G_CM3)
                density_kg_m3_list.append(rho_amu_A3 * AMU_PER_A3_TO_KG_M3)
                density_atoms_A3_list.append(n_atoms / vol_A3)
                vol_list.append(vol_A3)

        # --- Build result ---
        result = {
            "T_average_K": round(float(np.mean(temp_list)), 1),
            "n_frames": n_frames,
        }

        if "energy" in properties:
            result.update({
                "E_pot_eV_per_atom": round(float(np.mean(epot_list)), 6),
                "E_tot_eV_per_atom": round(float(np.mean(etot_list)), 6),
                "H_kJ_per_mol": round(float(np.mean(enthalpy_list)), 2),
            })

        if "density" in properties:
            result.update({
                "density_g_cm3": round(float(np.mean(density_g_cm3_list)), 4),
                "density_kg_m3": round(float(np.mean(density_kg_m3_list)), 2),
                "density_atoms_A3": round(float(np.mean(density_atoms_A3_list)), 6),
                "V_average_A3": round(float(np.mean(vol_list)), 3),
                "mass_total_amu": round(float(mass_total_amu), 4),
            })

        return result