"""
Point Defect Calculator: Surface energies & Vacancy formation energies
for Ti-Al systems using ML potentials.

Calculates:
  - Relaxed surface energies σ (J/m²)
  - Vacancy formation energies Evf (eV)

Following Zhao et al., Phys. Rev. B 110, 184115 (2024) - Tables V, VI
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pymatgen.io.ase import AseAtomsAdaptor
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from ase.calculators.calculator import Calculator

from utils import CalculatorFactory, StructureFactory, SlabFactory, LayerAnalyzer

from pymatgen.analysis.defects.generators import VacancyGenerator
from pymatgen.analysis.defects.core import Vacancy

import time
import socket
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _to_python(obj):
    """Recursively convert numpy types to Python builtins for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class PointDefectCalculator:
    """
    Calculator for point defects: surface energies and vacancy formation energies.
    
    Usage:
        calc = PointDefectCalculator(args)
        calc.run()
    """
    
    TASKS = {
        "fcc_Al": {
            "surfaces": [
                {"miller": (1, 0, 0), "label": "σ{100}"},
                {"miller": (1, 1, 0), "label": "σ{110}"},
                {"miller": (1, 1, 1), "label": "σ{111}"},
            ],
            "vacancies": [
                {"element": "Al", "label": "Evf-Al", "miller": (1, 0, 0)},
            ],
        },
        "alpha_Ti": {
            "surfaces": [
                {"miller": (0, 0, 1), "label": "σ_basal"},
                {"miller": (1, 0, 0), "label": "σ_prism"},
                {"miller": (1, 0, 1), "label": "σ_pyr.I"},
                {"miller": (1, 1, 0), "label": "σ_pyr.II"},
            ],
            "vacancies": [
                {"element": "Ti", "label": "Evf-Ti", "miller": (0, 0, 1)},
            ],
        },
        "gamma_AlTi": {
            "surfaces": [
                {"miller": (1, 1, 1), "label": "σ{111}"},
            ],
            "vacancies": [
                {"element": "Ti", "label": "Evf-Ti", "miller": (1, 1, 1)},
                {"element": "Al", "label": "Evf-Al", "miller": (1, 1, 1)},
            ],
        },
        "alpha2_AlTi3": {
            "surfaces": [
                {"miller": (0, 0, 1), "label": "σ{0001}"},
            ],
            "vacancies": [
                {"element": "Ti", "label": "Evf-Ti", "miller": (0, 0, 1)},
                {"element": "Al", "label": "Evf-Al", "miller": (0, 0, 1)},
            ],
        },
        "D022_Al3Ti": {
            "surfaces": [
                {"miller": (0, 0, 1), "label": "σ{001}"},
            ],
            "vacancies": [
                {"element": "Ti", "label": "Evf-Ti", "miller": (0, 0, 1)},
                {"element": "Al", "label": "Evf-Al", "miller": (0, 0, 1)},
            ],
        },
    }
    
    # Conversion: eV/Å² → J/m²
    EV_A2_TO_JM2 = 16.0217656
    
    def __init__(self, args):
        self.args = args
        
        # Parse systems
        if args.systems == "all":
            self.systems = list(self.TASKS.keys())
        else:
            self.systems = [s.strip() for s in args.systems.split(",")]
        
        logger.info(f"Systems: {self.systems}")
        
        # Output directory
        self.output_base = Path(args.workdir) / args.family / args.variant / args.use_type
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Slab parameters
        self.n_layers = args.n_layers
        self.vacuum = args.vacuum
        self.repeat_inplane = tuple(map(int, args.repeat_inplane.split(',')))
        
        # Relaxation
        self.fmax = args.fmax
        self.max_steps = args.max_steps
        
        # Fixed fraction for surface/vacancy calculations
        self.fixed_fraction = args.fixed_fraction
        
        # Factories
        self._calculator = None
        self._structure_factory = StructureFactory()
        self._slab_factory = SlabFactory()

        self.relax_events: List[Dict] = []

        self.metrics = {
            "calculator": args.calculator,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "family": args.family,
            "variant": args.variant,
            "use_type": args.use_type,
            "systems": self.systems,
            "device": args.device,
            "dtype": args.dtype,
            "gpu_name": args.gpu_name,
            "fmax_target": float(args.fmax),
            "max_steps_limit": int(args.max_steps),
            "n_layers": int(args.n_layers),
            "vacuum": float(args.vacuum),
            "repeat_inplane": list(self.repeat_inplane),
            "fixed_fraction": float(args.fixed_fraction),
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
        }
    
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
    
    def _record_relaxation(
        self,
        label: str,
        type_: str,
        system: Optional[str],
        n_atoms: int,
        n_steps: int,
        converged: bool,
        time_sec: float,
    ) -> None:
        event = {
            "label": label,
            "type": type_,
            "system": system,
            "n_atoms": int(n_atoms),
            "n_steps": int(n_steps),
            "time_sec": float(time_sec),
            "time_per_step_sec": (float(time_sec) / int(n_steps)) if n_steps > 0 else None,
            "converged": bool(converged),
        }
        self.relax_events.append(event)
        logger.info(
            f"[timing] {label}: {time_sec:.2f} s, {n_steps} steps, "
            f"{n_atoms} atoms, converged={converged}"
        )
    
    def calc_bulk_energy(
        self,
        bulk,
        label: str = "bulk",
        system: Optional[str] = None,
    ) -> float:
        """Calculate energy per atom in bulk."""
        
        bulk_ase = AseAtomsAdaptor.get_atoms(bulk)
        bulk_ase.calc = self.calculator

        dyn = FIRE(bulk_ase, logfile=None, trajectory=None)

        t0 = time.perf_counter()
        converged = dyn.run(fmax=self.fmax, steps=self.max_steps)
        t1 = time.perf_counter()

        self._record_relaxation(
            label=label,
            type_="bulk",
            system=system,
            n_atoms=len(bulk_ase),
            n_steps=int(dyn.nsteps),
            converged=bool(converged),
            time_sec=float(t1 - t0),
        )

        e_total = bulk_ase.get_potential_energy()
        e_per_atom = e_total / len(bulk_ase)

        logger.info(f"Bulk E_total = {e_total:.6f} eV, E/atom = {e_per_atom:.6f} eV")
        return e_per_atom
    
    def relax_slab(
        self,
        atoms,
        fixed_indices: Optional[np.ndarray] = None,
        label: str = "slab",
        type_: str = "slab",
        system: Optional[str] = None,
    ) -> float:
        """Relax slab (or bulk) and return energy."""
        atoms_copy = atoms.copy()
        atoms_copy.calc = self.calculator

        if fixed_indices is not None and len(fixed_indices) > 0:
            atoms_copy.set_constraint(FixAtoms(indices=fixed_indices.tolist()))

        dyn = FIRE(atoms_copy, logfile=None, trajectory=None)

        t0 = time.perf_counter()
        converged = dyn.run(fmax=self.fmax, steps=self.max_steps)
        t1 = time.perf_counter()

        self._record_relaxation(
            label=label,
            type_=type_,
            system=system,
            n_atoms=len(atoms_copy),
            n_steps=int(dyn.nsteps),
            converged=bool(converged),
            time_sec=float(t1 - t0),
        )

        return atoms_copy.get_potential_energy()
    
    def calculate_surface_energy(
        self,
        system: str,
        miller: tuple,
        label: str,
        bulk,
        e_per_atom: float,
        output_dir: Path,
    ) -> Dict:
        """Calculate relaxed surface energy for one plane."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Surface: {system} {label} (miller={miller})")
        logger.info(f"{'='*50}")
        
        # Generate slab
        slab_pmg, _ = self._slab_factory.create(
            bulk, miller_index=miller,
            n_layers=self.n_layers, vacuum=self.vacuum,
        )
        slab_pmg.make_supercell([self.repeat_inplane[0], self.repeat_inplane[1], 1])
        
        # Convert to ASE
        atoms = AseAtomsAdaptor.get_atoms(slab_pmg)
        atoms.pbc = [True, True, False]
        atoms.center(axis=2)
        
        logger.info(f"Slab: {len(atoms)} atoms")
        
        # Surface area (one side)
        A = np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
        logger.info(f"Surface area: {A:.4f} Å²")
        
        # Get fixed bottom layers mask
        analyzer = LayerAnalyzer()
        result = analyzer.get_mask(
            atoms, mode="select",
            fraction=self.fixed_fraction, part="bottom",
            slice_pos="bottom"
        )
        fixed_indices = np.where(result.mask)[0]
        
        logger.info(f"Fixed {len(fixed_indices)} atoms (bottom {self.fixed_fraction*100:.0f}%)")
        
        # Relax slab
        #E_slab = self.relax_slab(atoms, fixed_indices)

        E_slab = self.relax_slab(
            atoms, fixed_indices,
            label=f"surface/{label}",
            type_="surface",
            system=system,
        )
        
        # Calculate σ
        n_atoms = len(atoms)
        sigma_ev_a2 = (E_slab - n_atoms * e_per_atom) / (2 * A)
        sigma_jm2 = sigma_ev_a2 * self.EV_A2_TO_JM2
        
        logger.info(f"E_slab = {E_slab:.6f} eV")
        logger.info(f"σ = {sigma_jm2:.4f} J/m²")
        
        return {
            "property": label,
            "type": "surface_energy",
            "miller": list(miller),
            "value": round(sigma_jm2, 4),
            "unit": "J/m2",
            "E_slab_eV": round(E_slab, 6),
            "n_atoms": n_atoms,
            "surface_area_A2": round(A, 4),
        }  

    @staticmethod
    def _create_defect_at_site(defect, site):
        """Создает копию дефекта с конкретным сайтом и multiplicity=1."""
        return defect.__class__(
            structure=defect.structure,
            site=site,
            multiplicity=1,
            equivalent_sites=[site],
        )

    def calculate_vacancy_energy(
        self,
        system: str,
        element: str,
        label: str,
        miller: tuple,         # оставляем для совместимости, но не используем
        bulk,
        e_per_atom: float,
        output_dir: Path,
    ) -> Dict:
        """Calculate vacancy formation energy in BULK supercell."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Vacancy: {system} {label} (element={element})")
        logger.info(f"{'='*50}")
        
        # Make bulk supercell
        bulk_perfect = bulk.copy()        
        bulk_defect = bulk.copy()        
        
        # 1. Получаем все вакансии через генератор
        elements = list(bulk_defect.symbol_set)

        gen = VacancyGenerator()
        all_defects = gen.get_defects(bulk_defect, rm_species=elements)

        # 2. Группируем дефекты по элементу
        defects_map = {}
        for defect in all_defects:
            el = str(defect.site.specie)
            defects_map.setdefault(el, []).append(defect)

        defects = defects_map[element]
        defect = defects[0]        
        site = defect.equivalent_sites[0]
        vacancy = self._create_defect_at_site(defect, site)

        vacancy_structure = vacancy.get_supercell_structure()
        
        sc_mat = np.round(vacancy_structure.lattice.matrix @ np.linalg.inv(bulk_defect.lattice.matrix)).astype(int)

        bulk_perfect.make_supercell([sc_mat[0,0], sc_mat[1,1], sc_mat[2,2]])

        logger.info(f"ratio: {[sc_mat[0,0], sc_mat[1,1], sc_mat[2,2]]}; vacancy_structure: {len(vacancy_structure)} bulk_perfect: {len(bulk_perfect)}")

        # Convert to ASE
        atoms_perfect = AseAtomsAdaptor.get_atoms(bulk_perfect)
        atoms_perfect.pbc = [True, True, True]
        atoms_vacancy = AseAtomsAdaptor.get_atoms(vacancy_structure)
        atoms_vacancy.pbc = [True, True, True]
                
        # Relax perfect bulk (NO CONSTRAINTS)        
        E_perfect = self.relax_slab(
            atoms_perfect,
            label=f"vacancy_bulk/{label}/perfect",
            type_="vacancy_bulk",
            system=system,
        )
                
        # Relax bulk with vacancy (NO CONSTRAINTS)               
        E_vac = self.relax_slab(
            atoms_vacancy,
            label=f"vacancy_bulk/{label}/vac",
            type_="vacancy_bulk",
            system=system,
        )
        
        # Evf
        if element == "Ti":
            mu = self.mu_Ti0
        else:
            mu = self.mu_Al0
        Evf = E_vac - E_perfect + mu        
        
        logger.info(f"E_perfect = {E_perfect:.6f} eV")
        logger.info(f"E_vac     = {E_vac:.6f} eV")
        logger.info(f"Evf       = {Evf:.4f} eV")
        
        return {
            "property": label,
            "type": "vacancy_formation_energy",
            "element": element,
            "value": round(Evf, 4),
            "unit": "eV",
            "E_perfect_eV": round(E_perfect, 6),
            "E_vac_eV": round(E_vac, 6),            
        }

    def run_system(self, system: str) -> Dict:
        """Run all calculations for one system."""
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING: {system}")
        logger.info(f"{'='*60}")
        
        output_dir = self.output_base / system 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bulk
        bulk = self._structure_factory.get_structure(system, cell_type="conventional")
        
        # Bulk energy per atom
        logger.info("\nCalculating bulk energy...")
        #e_per_atom = self.calc_bulk_energy(bulk)
        e_per_atom = self.calc_bulk_energy(bulk, label=f"bulk/{system}", system=system)
        results = {
            "system": system,
            "calculator": self.args.calculator,
            "family": self.args.family,
            "variant": self.args.variant,
            "use_type": self.args.use_type,
            "e_per_atom_eV": round(e_per_atom, 6),
            "properties": [],
        }
        
        tasks = self.TASKS.get(system, {})
        
        # Surface energies
        for surf_cfg in tasks.get("surfaces", []):
            try:
                prop = self.calculate_surface_energy(
                    system, surf_cfg["miller"], surf_cfg["label"],
                    bulk, e_per_atom, output_dir,
                )
                results["properties"].append(prop)
            except Exception as e:
                logger.error(f"Failed surface {surf_cfg['label']}: {e}")
        
        # Vacancy energies
        for vac_cfg in tasks.get("vacancies", []):
            try:
                prop = self.calculate_vacancy_energy(
                    system, vac_cfg["element"], vac_cfg["label"], vac_cfg["miller"],
                    bulk, e_per_atom, output_dir,
                )
                results["properties"].append(prop)
            except Exception as e:
                logger.error(f"Failed vacancy {vac_cfg['label']}: {e}")
        
        # Save results
        json_path = output_dir / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to: {json_path}")
        
        return results

    def _save_metrics(self, total_wall_time_sec: float) -> None:
        events = self.relax_events

        n_relax = len(events)
        total_time = sum(e["time_sec"] for e in events)
        total_steps = sum(e["n_steps"] for e in events)
        total_atom_steps = sum(e["n_atoms"] * e["n_steps"] for e in events)
        n_converged = sum(1 for e in events if e["converged"])

        summary = {
            "calculator": self.args.calculator,
            "model_name": self.args.model_name,
            "model_path": self.args.model_path,
            "family": self.args.family,
            "variant": self.args.variant,
            "use_type": self.args.use_type,
            "systems": self.systems,
            "device": self.args.device,
            "dtype": self.args.dtype,
            "gpu_name": self.args.gpu_name,
            "total_wall_time_sec": float(total_wall_time_sec),
            "total_relax_time_sec": float(total_time),
            "n_relaxations": int(n_relax),
            "total_steps": int(total_steps),
            "total_atom_steps": int(total_atom_steps),
            "time_per_step_avg_sec": float(total_time / total_steps) if total_steps > 0 else None,
            "time_per_atom_step_avg_sec": float(total_time / total_atom_steps) if total_atom_steps > 0 else None,
            "n_converged": int(n_converged),
            "n_not_converged": int(n_relax - n_converged),
            "converged_ratio": float(n_converged / n_relax) if n_relax > 0 else None,
        }

        self.metrics["relaxations"] = events
        self.metrics["summary"] = summary

        metrics_file = self.output_base / "metrics.json"
        clean = _to_python(self.metrics)
        with open(metrics_file, "w") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)

        logger.info(f"Metrics saved to: {metrics_file}")
    
    def run(self):
        """Main execution."""
        t_start = time.perf_counter()
        logger.info("=" * 60)
        logger.info("POINT DEFECT CALCULATOR")
        logger.info("=" * 60)
        logger.info(f"Calculator: {self.args.calculator}")
        logger.info(f"Model: {self.args.family}/{self.args.variant}/{self.args.use_type}")
        logger.info(f"Systems: {self.systems}")
        logger.info(f"Fixed fraction: {self.fixed_fraction}")
        logger.info("=" * 60)

        # Считаем химические потенциалы чистых металлов ОДИН раз
        logger.info("\nCalculating chemical potentials for pure metals...")

        # fcc_Al
        bulk_Al = self._structure_factory.get_structure("fcc_Al", cell_type="conventional")
        self.mu_Al0 = self.calc_bulk_energy(bulk_Al, label="mu_Al0", system="fcc_Al")

        # alpha_Ti
        bulk_Ti = self._structure_factory.get_structure("alpha_Ti", cell_type="conventional")
        self.mu_Ti0 = self.calc_bulk_energy(bulk_Ti, label="mu_Ti0", system="alpha_Ti")

        logger.info(f"μ_Al⁰ = {self.mu_Al0:.6f} eV/atom")
        logger.info(f"μ_Ti⁰ = {self.mu_Ti0:.6f} eV/atom")
        
        for system in self.systems:
            if system not in self.TASKS:
                logger.warning(f"Skipping unknown system: {system}")
                continue
            self.run_system(system)

        t_end = time.perf_counter()
        self._save_metrics(total_wall_time_sec=float(t_end - t_start))
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL CALCULATIONS COMPLETED")
        logger.info(f"Results: {self.output_base}")
        logger.info("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Point defect calculator")
    parser.add_argument("--calculator", required=True, choices=["grace", "mace", "deepmd"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--family", required=True, choices=["dpa4", "dpa4c", "grace"])
    parser.add_argument("--variant", required=True, choices=["nano", "mini", "neo", "air", "plus", "2L", "3L"])
    parser.add_argument("--use-type", required=True, choices=["pretrained", "fit", "finetune", "finetune_half"])
    parser.add_argument("--systems", default="all", help="Comma-separated or 'all'")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--gpu-name", default="CPU", help="Human-readable GPU name")
    parser.add_argument("--workdir", default="models_check/results")
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--vacuum", type=float, default=12.0)
    parser.add_argument("--repeat-inplane", default="3,3,3")
    parser.add_argument("--fixed-fraction", type=float, default=0.25)
    parser.add_argument("--fmax", type=float, default=0.001)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    
    calculator = PointDefectCalculator(args)
    calculator.run()


if __name__ == "__main__":
    main()