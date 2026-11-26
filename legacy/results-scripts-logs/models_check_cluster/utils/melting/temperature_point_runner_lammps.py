"""TemperaturePointRunner: Single temperature point using LAMMPS."""

import logging
import os
from pathlib import Path
from lammps import lammps
from .point_result_lammps import PointResultLammps

logger = logging.getLogger(__name__)


class LammpsTemperaturePointRunner:
    """
    Расчёт одной температуры.

    Порядок стадий (per T):
        1. minimize    (relax, T=0)
        2. equil       (fix 1 all    npt T -> T)
        3. overheat    (fix 1 liquid npt T_melt -> T_melt)
        4. cool        (fix 1 liquid npt T_melt -> T)
        5. melt        (fix 1 all    npt T -> T, production)

    где T_melt = T * T_overheat_factor.

    Поддерживает callback и native режимы.
    """

    def __init__(
        self,
        mode: str,
        calculator: str,
        structure_factory,
        calculator_obj=None,
        model_path: str = None,
        dt: float = 0.001,
        Tdamp: float = 0.1,
        Pdamp: float = 1.0,
        T_overheat_factor: float = 1.5,
        equil_steps: int = 10000,
        heat_steps: int = 25000,
        cool_steps: int = 15000,
        melt_steps: int = 25000,
        min_tol_energy: float = 1e-10,
        min_tol_force: float = 1e-10,
        min_max_iter: int = 5000,
        min_max_eval: int = 10000,
        thermo: int = 500,
    ):
        if mode == "native" and calculator not in ["grace", "deepmd"]:
            raise ValueError(f"Native mode requires calculator='grace' or 'deepmd', got '{calculator}'")
        if mode == "callback" and calculator_obj is None:
            raise ValueError("Callback mode requires calculator_obj")

        self.mode = mode
        self.calculator = calculator
        self.structure_factory = structure_factory
        self.calculator_obj = calculator_obj
        self.model_path = model_path
        self.dt = dt
        self.Tdamp = Tdamp
        self.Pdamp = Pdamp
        self.T_overheat_factor = T_overheat_factor
        self.equil_steps = equil_steps
        self.heat_steps = heat_steps
        self.cool_steps = cool_steps
        self.melt_steps = melt_steps
        self.min_tol_energy = min_tol_energy
        self.min_tol_force = min_tol_force
        self.min_max_iter = min_max_iter
        self.min_max_eval = min_max_eval
        self.thermo = thermo

        # Cache element for native mode
        self._element = None

    # ------------------------------------------------------------------
    # LAMMPS setup
    # ------------------------------------------------------------------

    def _get_element(self, data_path):
        """Extract element string from LAMMPS data file for native mode."""
        if self._element is not None:
            return self._element
        result = self.structure_factory.read_lammps_data(str(data_path))
        specorder = result["specorder"]
        self._element = specorder[0] if len(specorder) == 1 else " ".join(specorder)
        return self._element

    def _init_pair(self, L, data_path):
        """Initialize pair style and fix external if needed."""
        if self.mode == "callback":
            from utils import ExternalASEForce

            L.commands_string("""
            pair_style zero 6.0
            pair_coeff * *
            """)
            L.command("fix ext all external pf/callback 1 1")

            result = self.structure_factory.read_lammps_data(str(data_path))
            atoms = result["atoms"]
            external = ExternalASEForce(atoms, self.calculator_obj, fix_id="ext")
            external.register(L)
        else:
            element = self._get_element(data_path)
            if self.calculator == "grace":
                L.commands_string(f"""
                pair_style grace
                pair_coeff * * {self.model_path} {element}
                """)
            elif self.calculator == "deepmd":
                L.commands_string(f"""
                pair_style deepmd {self.model_path}
                pair_coeff * * {element}
                """)

    def _run_common(self, L, minimizer=False):
        """Common LAMMPS setup: variables, groups, computes, thermo."""
        L.commands_string(f"""
        variable dt equal {self.dt}
        timestep ${{dt}}
        variable Tdamp equal {self.Tdamp}
        variable Pdamp equal {self.Pdamp}

        variable z2 equal zhi
        variable z1 equal zlo
        variable lz equal ${{z2}}-${{z1}}
        variable z_cen equal ${{z1}}+${{lz}}/2
        region liquid block INF INF INF INF INF ${{z_cen}} units box
        group liquid region liquid
        group solid subtract all liquid

        compute 2 all pe/atom
        compute LiquidTemp liquid temp
        compute SolidTemp solid temp
        """)
        if minimizer:
            L.commands_string(f"""
            min_style cg
            minimize {self.min_tol_energy} {self.min_tol_force} {self.min_max_iter} {self.min_max_eval}
            """)
        L.commands_string(f"""
        thermo {self.thermo}
        thermo_style custom step etotal pe ke press pxx pyy pzz temp c_LiquidTemp c_SolidTemp
        """)

    def _new_lammps(self, data_path, minimizer=False):
        """Create new LAMMPS instance with common setup."""
        L = lammps()
        map_command = "atom_modify map yes" if self.calculator == "deepmd" else ""
        L.commands_string(f"""
        units metal
        boundary p p p
        atom_style atomic
        {map_command}
        read_data {data_path} nocoeff
        """)
        self._init_pair(L, data_path)
        L.commands_string("""
        neighbor 2.0 bin
        neigh_modify delay 10 check yes
        """)
        self._run_common(L, minimizer=minimizer)

        return L

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, T_target, input_path, cache_dir, dumps_dir, cache, run_id, params, skip_stages):
        """
        Run single temperature point.

        Args:
            T_target: target temperature (K)
            input_path: path to LAMMPS data file to start from
            cache_dir: directory for stage outputs
            dumps_dir: directory for trajectory dumps
            cache: LammpsCacheManager instance
            run_id: run identifier
            params: run parameters dict
            skip_stages: list of already-completed stages (prefix of STAGE_FILES)

        Returns:
            PointResultLammps
        """
        dumps_dir = Path(dumps_dir)
        dumps_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Stage output paths
        equil_path = cache_dir / "equil.lammps"
        overheat_path = cache_dir / "overheat.lammps"
        cool_path = cache_dir / "cool.lammps"
        melt_path = cache_dir / "melt.lammps"

        # T_melt computed per T
        T_melt = T_target * self.T_overheat_factor

        result = PointResultLammps(T_target)

        logger.info(f"\n{'='*50}")
        logger.info(f"TEMPERATURE POINT: T = {T_target:.0f} K")
        logger.info(f"{'='*50}")
        logger.info(f"  Mode:        {self.mode}")
        logger.info(f"  T_melt:      {T_melt:.0f} K (x{self.T_overheat_factor})")
        logger.info(f"  Equil:       {self.equil_steps} steps")
        logger.info(f"  Heat:        {self.heat_steps} steps")
        logger.info(f"  Cool:        {self.cool_steps} steps")
        logger.info(f"  Melt:        {self.melt_steps} steps")
        logger.info(f"  Skip stages: {skip_stages if skip_stages else '<none>'}")
        logger.info(f"  Input:       {input_path}")
        logger.info(f"{'='*50}")

        data_path = input_path

        # ----------------------------------------------------------
        # Stage 1: MINIMIZE (always first, never cached separately —
        #          runs together with equil on first entry)
        # ----------------------------------------------------------
        # We run minimize + equil in the same LAMMPS instance when
        # equil is not yet done.
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # Stage 2: EQUIL
        # ----------------------------------------------------------
        if "equil" not in skip_stages:
            L = self._new_lammps(data_path, minimizer=True)

            logger.info(f"  --- EQUIL: {T_target:.0f} K ---")
            L.commands_string(f"""
            reset_timestep 0
            velocity all create {T_target} 4928459 dist gaussian
            fix 1 all npt temp {T_target} {T_target} {self.Tdamp} aniso 0 0 {self.Pdamp}
            dump dnve all custom {self.thermo} {dumps_dir}/{run_id}_Step1_Equi.xyz id type x y z c_2
            run {self.equil_steps}
            unfix 1
            undump dnve
            write_data {equil_path}
            """)
            L.close()
            result.equil_path = equil_path
            cache.save(run_id, params, "equil")
            data_path = equil_path
        else:
            logger.info(f"  --- EQUIL: skipped (cached) ---")
            result.equil_path = equil_path

        # ----------------------------------------------------------
        # Stage 3: OVERHEAT (liquid only)
        # ----------------------------------------------------------
        if "overheat" not in skip_stages:
            L = self._new_lammps(data_path)

            logger.info(f"  --- OVERHEAT: {T_melt:.0f} K (liquid) ---")
            L.commands_string(f"""
            reset_timestep 0
            fix 1 liquid npt temp {T_target} {T_melt} {self.Tdamp} aniso 0 0 {self.Pdamp}
            dump dnve all custom {self.thermo} {dumps_dir}/{run_id}_Step2_Heat.xyz id type x y z c_2
            run {self.heat_steps}
            unfix 1
            undump dnve
            write_data {overheat_path}
            """)
            L.close()
            result.overheat_path = overheat_path
            cache.save(run_id, params, "overheat")
            data_path = overheat_path
        else:
            logger.info(f"  --- OVERHEAT: skipped (cached) ---")
            result.overheat_path = overheat_path

        # ----------------------------------------------------------
        # Stage 4: COOL (liquid only)
        # ----------------------------------------------------------
        if "cool" not in skip_stages:
            L = self._new_lammps(data_path)

            logger.info(f"  --- COOL: {T_melt:.0f} K -> {T_target:.0f} K (liquid) ---")
            L.commands_string(f"""
            reset_timestep 0
            fix 1 liquid npt temp {T_melt} {T_target} {self.Tdamp} aniso 0 0 {self.Pdamp}
            dump dnve all custom {self.thermo} {dumps_dir}/{run_id}_Step3_Cool.xyz id type x y z c_2
            run {self.cool_steps}
            unfix 1
            undump dnve
            write_data {cool_path}
            """)
            L.close()
            result.cool_path = cool_path
            cache.save(run_id, params, "cool")
            data_path = cool_path
        else:
            logger.info(f"  --- COOL: skipped (cached) ---")
            result.cool_path = cool_path

        # ----------------------------------------------------------
        # Stage 5: MELT (all, production)
        # ----------------------------------------------------------
        if "melt" not in skip_stages:
            L = self._new_lammps(data_path)

            logger.info(f"  --- MELT: {T_target:.0f} K (all) ---")
            L.commands_string(f"""
            reset_timestep 0
            fix 1 all npt temp {T_target} {T_target} {self.Tdamp} aniso 0 0 {self.Pdamp}
            dump dnve all custom {self.thermo} {dumps_dir}/{run_id}_Step4_Melt.xyz id type x y z c_2
            run {self.melt_steps}
            unfix 1
            undump dnve
            write_data {melt_path}
            """)
            L.close()
            result.melt_path = melt_path
            cache.save(run_id, params, "melt")
        else:
            logger.info(f"  --- MELT: skipped (cached) ---")
            result.melt_path = melt_path

        logger.info(f"  T = {T_target:.0f} K done")
        return result