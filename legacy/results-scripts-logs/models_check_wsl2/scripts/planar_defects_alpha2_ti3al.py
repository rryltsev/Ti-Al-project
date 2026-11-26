"""
Planar defect calculator for α₂-Ti₃Al (DO₁₉) using ML potentials.
Calculates generalized stacking fault energy (γ-lines) on {0001} basal plane
following Zhao et al., Phys. Rev. B 110, 184115 (2024).

Uses crystallographic directions [01-10] and [-2110] from the article,
correctly expressed through slab basis via slab_scale_factor matrix.
"""

import sys
import os
import json
import math
import logging
from pathlib import Path
from typing import Optional, Dict, List
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymatgen.core import Structure, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from ase.constraints import FixCartesian
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator
from ase.spacegroup import crystal
from ase.build import surface
from ase import Atoms
from utils import CalculatorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GammaTiAlPlanarDefectCalculator:
    """Calculator for planar defects (stacking faults) in γ-TiAl following Zhao et al. (2024)."""
    
    # DFT reference values from Zhao et al., Phys. Rev. B 110, 184115 (2024) - Table VII
    DFT_REF = {
        "SISF": 104,   # mJ/m²
        "APB": 257,    # mJ/m²
        "CSF": 366,    # mJ/m²
    }
    
    # α₂-Ti₃Al (DO₁₉) lattice parameters
    A_ALPHA2 = 5.77  # Å
    C_ALPHA2 = 4.62  # Å
    
    def __init__(self, args):
        self.args = args
        
        # Output directory
        self.output_dir = Path(args.workdir) / args.family / args.variant / args.use_type / args.system
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {self.output_dir}")
        
        # Save command
        with open(self.output_dir / "command.txt", "w") as f:
            f.write(" ".join(sys.argv))
        
        # Parse defect types
        if args.defect_types == "all":
            self.defect_types = ["0001_01-10", "0001_-2110", "0001_combined"]
        else:
            self.defect_types = [d.strip() for d in args.defect_types.split(",")]
        logger.info(f"Defect types: {self.defect_types}")
        
        self._calculator = None
        self.n_layers = args.n_layers
        self.vacuum = args.vacuum
        self.repeat_inplane = tuple(map(int, args.repeat_inplane.split(',')))
        self.fmax = args.fmax
        self.max_steps = args.max_steps
        self.shift_step = args.shift_step
        self.max_fraction = args.max_fraction
        self.eV_to_J = 1.602176634e-19
        
        # Will be set during run()
        self.slab_gen = None
        
    @property
    def calculator(self) -> Calculator:
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
            elif self.args.calculator == "mace":
                self._calculator = factory.calculator("mace")
            elif self.args.calculator == "deepmd":
                self._calculator = factory.calculator("deepmd")
            else:
                raise ValueError(f"Unknown calculator: {self.args.calculator}")
        return self._calculator   

    def find_slip_plane_between_layers(self, atoms, tol=0.01, require_even_split=True):       
        """
        Находит плоскость скольжения в гамма-TiAl, используя дробные координаты 
        для компенсации тетрагонального скоса ячейки по оси Z.
        """
        # =========================================================
        # WORKING IN FRACTIONAL COORDINATES
        # =========================================================
        # get_scaled_positions() возвращает координаты от 0 до 1.
        # В дробном пространстве плоскость слэба всегда идеально плоская.
        frac_coords = atoms.get_scaled_positions()
        z_frac = frac_coords[:, 2]

        # =========================================================
        # SORT ATOMS BY FRACTIONAL Z
        # =========================================================
        order = np.argsort(z_frac)

        # =========================================================
        # BUILD LAYERS
        # =========================================================
        layers = []
        current = [order[0]]
        z_frac_ref = z_frac[order[0]]

        for idx in order[1:]:
            # tol=0.01 в дробных координатах — это примерно 0.5 Å для ячейки 53 Å
            if abs(z_frac[idx] - z_frac_ref) < tol:
                current.append(idx)
            else:
                layers.append(current)
                current = [idx]
                z_frac_ref = z_frac[idx]
        layers.append(current)

        n_layers = len(layers)
        if n_layers < 2:
            raise RuntimeError(
                f"Too few layers detected: {n_layers}. Check fractional tol."
            )

        # =========================================================
        # LAYER ANALYSIS (Converted to Cartesian for meaningful logs)
        # =========================================================
        logger.info("\n" + "=" * 80)
        logger.info("LAYER ANALYSIS (FRACTIONAL GATHERING -> CARTESIAN LOGS)")
        logger.info("=" * 80)

        layer_info = []

        for lid, inds in enumerate(layers):
            inds = np.array(inds)
            
            # Считаем геометрические параметры в реальных декартовых Ангстремах
            zz_cart = atoms.positions[inds, 2]
            z_mean = np.mean(zz_cart)
            spread = np.ptp(zz_cart) # Теперь spread покажет реальную гофрированность/скос слоя

            species = np.array(atoms.symbols)[inds]
            uniq, counts = np.unique(
                species,
                return_counts=True
            )

            chem = {
                str(u): int(c)
                for u, c in zip(uniq, counts)
            }

            layer_info.append({
                "id": lid,
                "inds": inds,
                "z_mean": z_mean,
                "spread": spread,
                "n_atoms": len(inds),
                "chem": chem,
            })

            logger.info(
                f"Layer {lid:3d} | "
                f"atoms={len(inds):3d} | "
                f"z={z_mean:10.6f} Å | "
                f"spread={spread:8.6f} Å | "
                f"chem={chem}"
            )

        # =========================================================
        # CENTRAL CUT
        # =========================================================
        mid = n_layers // 2

        lower = layer_info[mid - 1]
        upper = layer_info[mid]

        cut_z = 0.5 * (
            lower["z_mean"] + upper["z_mean"]
        )

        logger.info("\n" + "=" * 80)
        logger.info("SELECTED SLIP PLANE")
        logger.info("=" * 80)

        logger.info(
            f"Cut between layers "
            f"{mid - 1} and {mid}"
        )

        logger.info(
            f"Lower layer z = "
            f"{lower['z_mean']:.6f} Å"
        )

        logger.info(
            f"Upper layer z = "
            f"{upper['z_mean']:.6f} Å"
        )

        logger.info(
            f"Slip plane z = "
            f"{cut_z:.6f} Å"
        )

        # =========================================================
        # BUILD TOP MASK
        # =========================================================
        top_mask = np.zeros(
            len(atoms),
            dtype=bool
        )

        for layer in layer_info[mid:]:
            top_mask[layer["inds"]] = True

        n_top = np.sum(top_mask)
        n_bottom = len(atoms) - n_top

        logger.info(
            f"Bottom atoms = {n_bottom}"
        )
        logger.info(
            f"Top atoms    = {n_top}"
        )

        if require_even_split:
            if n_top != n_bottom:
                logger.warning(
                    f"Top/bottom halves are not equal! Top: {n_top}, Bottom: {n_bottom}"
                )

        logger.info("=" * 80)

        # =========================================================
        # FINAL SANITY CHECK
        # =========================================================
        for layer in layer_info:
            inds = layer["inds"]
            mask_values = top_mask[inds]

            if not (
                np.all(mask_values)
                or np.all(~mask_values)
            ):
                raise RuntimeError(
                    f"Layer {layer['id']} was split "
                    f"across the fault plane"
                )

        return top_mask
    
    def crystallographic_to_shift(self, d_bulk, slab):
        """
        Convert crystallographic direction to shift vector in slab.
        Uses formula: d_slab = (S⁻¹)ᵀ @ d_bulk
        where S = slab_scale_factor matrix.
        """
        #S = np.array(self.slab_gen.slab_scale_factor, dtype=float)
        S = np.array(self.slab_gen.slab_scale_factor, dtype=float)
        S_inv_T = np.linalg.inv(S).T
        
        d_bulk_arr = np.array(d_bulk, dtype=float)
        d_slab = S_inv_T @ d_bulk_arr
        
        k1, k2 = d_slab[0], d_slab[1]
        
        s1 = slab.lattice.matrix[0].copy()
        s2 = slab.lattice.matrix[1].copy()
        shift = k1 * s1 + k2 * s2
        shift[2] = 0.0
        
        logger.info(f"  Direction [{d_bulk[0]} {d_bulk[1]} {d_bulk[2]}]")
        logger.info(f"    d_slab = [{d_slab[0]:.6f}, {d_slab[1]:.6f}, {d_slab[2]:.6f}]")
        logger.info(f"    k1={k1:.6f}, k2={k2:.6f}")
        logger.info(f"    shift = [{shift[0]:.4f}, {shift[1]:.4f}, {shift[2]:.4f}], |shift| = {np.linalg.norm(shift):.4f} Å")
        
        return shift 

    def crystallographic_to_shift_upd(self, d_bulk: List[float], slab) -> np.ndarray:
        
        S = np.array(self.slab_gen.slab_scale_factor, dtype=float)
        d_bulk_arr = np.array(d_bulk, dtype=float)
        
        coeff = np.linalg.solve(S.T, d_bulk_arr)
        coeff = np.asarray(coeff, dtype=float)
        
        if not np.isclose(coeff[2], 0.0, atol=1e-10):
            raise ValueError(
                f"Direction [{d_bulk[0]} {d_bulk[1]} {d_bulk[2]}] is not in the slab plane: "
                f"normal coefficient = {coeff[2]:.6f}"
            )
        
        k1, k2 = coeff[0], coeff[1]
        
        s1 = slab.lattice.matrix[0].copy()
        s2 = slab.lattice.matrix[1].copy()
        shift = k1 * s1 + k2 * s2
        shift[2] = 0.0
        
        logger.info(f"  Direction [{d_bulk[0]} {d_bulk[1]} {d_bulk[2]}]")
        logger.info(f"    coeff = [{coeff[0]:.6f}, {coeff[1]:.6f}, {coeff[2]:.6f}]")
        logger.info(f"    k1={k1:.6f}, k2={k2:.6f}")
        logger.info(f"    shift = [{shift[0]:.4f}, {shift[1]:.4f}, {shift[2]:.4f}], |shift| = {np.linalg.norm(shift):.4f} Å")
        
        return shift
    
    def setup_target_planes(self, slab):
        """Define slip directions for α₂-Ti₃Al {0001} from the article."""
        planes = []
        
        # Direction 1: Pure [01-10] shift (full vector = a = 5.77 Å)
        # 4-index [01-10] → 3-index [0, 1, -1]
        if "0001_01-10" in self.defect_types:
            #shift = self.crystallographic_to_shift([1, -1, 0], slab)
            shift = self.crystallographic_to_shift_upd([1, 2, 0], slab)
            planes.append({
                "name": "0001_01-10_direction",
                "shift_vector": shift,
                "special_points": {1/3: "SISF", 0.5: "APB", 5/6: "CSF"},
                "description": "Pure [01-10] shift (Fig. 6a,b) - SISF/APB/CSF"
            })
        
        # Direction 2: Pure [-2110] shift (full vector = a = 5.77 Å)
        # 4-index [-2110] → 3-index [-2, 1, 1]
        if "0001_-2110" in self.defect_types:
            #shift = self.crystallographic_to_shift([0, 1, -1], slab)
            shift = self.crystallographic_to_shift_upd([-1, 0, 0], slab)
            planes.append({
                "name": "0001_-2110_direction",
                "shift_vector": shift,
                "special_points": {},
                "description": "Pure [-2110] shift (Fig. 6c) - γ-line only"
            })
        
        # Direction 3: Combined u*[01-10] + v*[-2110]
        if "0001_combined" in self.defect_types:
            #shift_v = self.crystallographic_to_shift([-1, 1, 0], slab)
            #shift_u = self.crystallographic_to_shift([0, 1, -1], slab)
            shift_v = self.crystallographic_to_shift_upd([1, 2, 0], slab)
            shift_u = self.crystallographic_to_shift_upd([-1, 0, 0], slab)
            planes.append({
                "name": "0001_combined_path",
                "shift_fixed": shift_u,
                "shift_var": shift_v,
                "special_points": {},
                "description": "Combined u*[01-10] + v*[-2110] (Fig. 6d)"
            })
        
        return planes
    
    def get_bulk_structure(self):
        """Create α₂-Ti₃Al (DO₁₉) bulk structure."""
        a = self.A_ALPHA2
        c = self.C_ALPHA2
        
        logger.info(f"Generating α₂-Ti₃Al (DO19) structure with a={a:.4f} Å, c={c:.4f} Å")
        
        bulk = Structure.from_spacegroup(
            "P63/mmc",
            Lattice.hexagonal(a, c),
            ["Al", "Ti"],
            [
                [1/3, 2/3, 1/4],   # Al - Wyckoff 2c
                [5/6, 2/3, 1/4]    # Ti - Wyckoff 6h
            ]
        )
        
        logger.info(f"Generated bulk: {bulk.composition}, {len(bulk)} atoms")
        return bulk

    def generate_slab(self):
        """Generate slab for {111} plane."""
        bulk = self.get_bulk_structure()
        
        self.slab_gen = SlabGenerator(
            bulk,
            (0, 0, 1),
            min_slab_size=self.n_layers,
            min_vacuum_size=self.vacuum,
            center_slab=True,
            primitive=True,          # Обязательно True! Уберет тетрагональный перекос векторов в плоскости XY
            in_unit_planes=True,
            lll_reduce=True,         # Включаем LLL-редукцию для получения максимально красивых и чистых углов ячейки
            max_normal_search=10
        )
        
        # Генерируем слэбы с жестким требованием СИММЕТРИИ поверхностей
        slabs = self.slab_gen.get_slabs(
            symmetrize=True,         # Обязательно Enforce equivalency of slab surfaces! Верх и низ слэба станут одинаковыми
            filter_out_sym_slabs=False # Выключаем агрессивный фильтр, чтобы увидеть ВСЕ химические варианты срезов
        )

        print(f"Всего генератор выдал вариантов среза: {len(slabs)}")
        for i, s in enumerate(slabs):
            print(f"Срез №{i}: состав {s.composition}, симметричен: {s.is_symmetric}")


        slab = slabs[0]
        
        logger.info(f"Generated slab: {slab.composition}, {len(slab)} atoms")
        logger.info(f"Slab size: {len(slab)} atoms, cell: {slab.lattice.abc}")
        logger.info(f"slab_scale_factor:\n{self.slab_gen.slab_scale_factor}")
        
        return slab, bulk
    
    def relax_structure(self, atoms, fix_xy=True, skip_relax=False):
        """Relax atomic positions."""
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator
        
        if fix_xy:
            atoms_copy.set_constraint(FixCartesian(range(len(atoms_copy)), mask=(True, True, False)))
        if skip_relax:
            return atoms_copy
        else:    
            dyn = FIRE(atoms_copy, logfile=None, trajectory=None)
            dyn.run(fmax=self.fmax, steps=self.max_steps)        
            return atoms_copy
    
    def calculate_gamma_line(self, plane_info, atoms, A_m2, output_subdir):
        """Calculate γ-line for a given slip direction."""
        plane_name = plane_info["name"]
        logger.info(f"\n{'='*50}")
        logger.info(f"Calculating γ-line: {plane_name}")
        logger.info(f"{'='*50}")
        
        # Step 2: Find slip plane        
        top_mask = self.find_slip_plane_between_layers(atoms)

        # Step 1: Relax ideal slab WITHOUT constraints
        logger.info("\nRelaxing ideal slab (E0, no constraints)...")
        atoms_relaxed = self.relax_structure(atoms, fix_xy=False, skip_relax=False)
        E0 = atoms_relaxed.get_potential_energy()
        logger.info(f"E0 = {E0:.6f} eV")

        # Step 3: Calculate γ for each shift fraction
        shift_fractions = np.arange(0, self.max_fraction + self.shift_step/2, self.shift_step)
        results = []
        
        logger.info(f"\nCalculating γ for {len(shift_fractions)} shifts...")
        
        for frac in shift_fractions:
            atoms_def = atoms_relaxed.copy()
            pos = atoms_def.positions.copy()
            
            if "shift_vector" in plane_info:
                shift = frac * plane_info["shift_vector"]
            else: 
                """               
                if frac < 0.18:
                    u = 0
                    v = frac * 1.0                    
                elif 0.18 < frac < 0.24:
                    u = frac * 1.0 
                    v = 0.18 + (frac - 0.18) * 1.0   
                elif 0.24 < frac < 0.42:
                    u = 0.15 
                    v = 0.24 + (frac -0.24) * 1.0        
                else:
                    u = 0.15 + (frac - 0.15) * 1.0 
                    v = 0.42 + (frac - 0.42) * 1.0   
                """
                if frac < 0.5:
                    u = 0.18 
                    v = frac * 1       
                else:
                    u = 0.18
                    v = 0.5 + (frac - 0.5) * 1
                
                shift = u * plane_info["shift_fixed"] + v * plane_info["shift_var"] / 2
            
            # Apply shift
            pos[top_mask] += shift
            atoms_def.set_positions(pos)
                        
            # Step 5: Relax WITH FixCartesian (only Z free)
            atoms_def = self.relax_structure(atoms_def, fix_xy=True, skip_relax=False)
            E = atoms_def.get_potential_energy()
            
            gamma_mJm2 = (E - E0) * self.eV_to_J / A_m2 * 1e3
            
            logger.info(f"  frac={frac:.2f} | E={E:.6f} eV | γ={gamma_mJm2:.1f} mJ/m²")
            results.append({"shift_fraction": frac, "gamma_mJm2": gamma_mJm2})
        
        df = pd.DataFrame(results)
        
        # Save CSV
        csv_path = output_subdir / "gamma_line.csv"
        df.to_csv(csv_path, index=False)
        
        # Plot
        self.plot_gamma_line(df, plane_info, output_subdir)
    
        return df
    
    def plot_gamma_line(self, df, plane_info, output_subdir):
        """Plot γ-line and save figure."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df["shift_fraction"], df["gamma_mJm2"], "o-", color="firebrick", lw=2, markersize=4)
        
        y_max = df["gamma_mJm2"].max()
        if y_max > 0:
            x_max = df.loc[df["gamma_mJm2"].idxmax(), "shift_fraction"]
            ax.annotate(f"Peak: {y_max:.0f} mJ/m²", xy=(x_max, y_max),
                       xytext=(x_max, y_max * 1.05), ha='center', fontsize=10, fontweight='bold')
        
        if "special_points" in plane_info:
            colors = {"SISF": "blue", "APB": "green", "CSF": "orange"}
            for frac_target, name in plane_info["special_points"].items():
                idx = (df["shift_fraction"] - frac_target).abs().idxmin()
                nearest_frac = df.loc[idx, "shift_fraction"]
                gamma_val = df.loc[idx, "gamma_mJm2"]
                ax.axvline(x=frac_target, color=colors.get(name, "gray"), linestyle="--", alpha=0.5)
                ax.annotate(f"{name}\n{gamma_val:.0f} mJ/m²", xy=(nearest_frac, gamma_val),
                           xytext=(frac_target, gamma_val * 0.85), ha='center', fontsize=9,
                           color=colors.get(name, "gray"), fontweight='bold')
        
        ax.set_title(f"γ-line: {plane_info['name']}\n{plane_info['description']}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Shift fraction", fontsize=12)
        ax.set_ylabel("γ (mJ/m²)", fontsize=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        if y_max > 0:
            ax.set_ylim(0, y_max * 1.2)
        
        plt.tight_layout()
        fig_path = output_subdir / "gamma_line.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
    
    def extract_special_points(self, df, plane_info):
        """Extract special point energies from γ-line."""
        if "special_points" not in plane_info:
            return {}
        special_points = {}
        for frac_target, name in plane_info["special_points"].items():
            idx = (df["shift_fraction"] - frac_target).abs().idxmin()
            gamma_val = df.loc[idx, "gamma_mJm2"]
            actual_frac = df.loc[idx, "shift_fraction"]
            special_points[name] = {
                "target_fraction": frac_target,
                "actual_fraction": actual_frac,
                "gamma_mJm2": gamma_val,
                "dft_reference_mJm2": self.DFT_REF.get(name),
                "error_pct": (gamma_val - self.DFT_REF[name]) / self.DFT_REF[name] * 100 
                             if name in self.DFT_REF else None
            }
            logger.info(f"  {name}: γ={gamma_val:.1f} mJ/m² (DFT: {self.DFT_REF.get(name, 'N/A')} mJ/m², "
                       f"error: {special_points[name]['error_pct']:.1f}%)")
        return special_points
    
    def compare_with_dft(self, all_special_points):
        """Generate comparison table."""
        comparison = []
        for plane_name, points in all_special_points.items():
            for defect_name, data in points.items():
                comparison.append({
                    "plane": plane_name, "defect": defect_name,
                    "calculated_mJm2": round(data["gamma_mJm2"], 1),
                    "dft_reference_mJm2": data["dft_reference_mJm2"],
                    "error_pct": round(data["error_pct"], 1) if data["error_pct"] is not None else None
                })
        df_comparison = pd.DataFrame(comparison)
        comp_path = self.output_dir / "comparison_table.json"
        with open(comp_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        df_comparison.to_csv(self.output_dir / "comparison_table.csv", index=False)
        logger.info("\n" + "="*70)
        logger.info("COMPARISON WITH DFT (Zhao et al., 2024 - Table VII)")
        logger.info("="*70)
        logger.info(f"\n{df_comparison.to_string(index=False)}")
        logger.info("="*70)
        return df_comparison
    
    def run(self):
        """Main execution."""
        logger.info("="*60)
        logger.info("γ-TiAl PLANAR DEFECT CALCULATOR (correct directions via slab_scale_factor)")
        logger.info("="*60)
        logger.info(f"Calculator: {self.args.calculator} | Model: {self.args.family}/{self.args.variant}/{self.args.use_type} | System: {self.args.system}")
        logger.info(f"Defect types: {self.defect_types} | Shift step: {self.shift_step}")
        
        # Generate slab
        logger.info("\nGenerating slab for {111} plane...")
        slab, bulk = self.generate_slab()
        
        # Setup target planes with CORRECT method
        logger.info("\nComputing shift vectors via slab_scale_factor...")
        planes = self.setup_target_planes(slab)        

        # Make supercell
        slab.make_supercell([self.repeat_inplane[0], self.repeat_inplane[1], 1])
        logger.info(f"Supercell size: {len(slab)} atoms, cell: {slab.lattice.abc}")
        
        # Convert to ASE
        atoms = AseAtomsAdaptor.get_atoms(slab)
        atoms.pbc = [True, True, False]
        atoms.center(axis=2)

        # Area
        A = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
        A_m2 = A * 1e-20
        logger.info(f"Slip plane area: {A:.2f} Å²")
        
        all_special_points = {}
        
        for plane_info in planes:
            output_subdir = self.output_dir / plane_info["name"]
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            df = self.calculate_gamma_line(plane_info, atoms.copy(), A_m2, output_subdir)
            
            special_points = self.extract_special_points(df, plane_info)
            if special_points:
                all_special_points[plane_info["name"]] = special_points
                with open(output_subdir / "special_points.json", 'w') as f:
                    json.dump(special_points, f, indent=2)
        
        if all_special_points:
            self.compare_with_dft(all_special_points)
        
        logger.info("\n" + "="*60)
        logger.info("CALCULATION COMPLETED SUCCESSFULLY")
        logger.info(f"Results: {self.output_dir}")
        logger.info("="*60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate planar defect energies in γ-TiAl (correct directions)")
    parser.add_argument("--calculator", required=True, choices=["grace", "mace", "deepmd"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--family", required=True, choices=["dpa4", "dpa4c", "grace"])
    parser.add_argument("--variant", required=True, choices=["nano", "mini", "neo", "air", "plus", "2L", "3L"])
    parser.add_argument("--use-type", required=True, choices=["pretrained", "fit", "finetune", "finetune_half"])
    parser.add_argument("--system", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--workdir", default="models_check")
    parser.add_argument("--defect-types", default="all",
                        help="Comma-separated: 0001_01-10,0001_-2110,0001_combined or 'all'")
    parser.add_argument("--shift-step", type=float, default=0.05)
    parser.add_argument("--max-fraction", type=float, default=1.0)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--vacuum", type=float, default=12.0)
    parser.add_argument("--repeat-inplane", default="3,3")
    parser.add_argument("--fmax", type=float, default=0.001)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    
    calculator = GammaTiAlPlanarDefectCalculator(args)
    calculator.run()


if __name__ == "__main__":
    main()