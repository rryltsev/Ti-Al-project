import argparse
import sys
from pathlib import Path

# Добавляем models_check в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from ase.io import write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
from ase.phonons import Phonons
import ase.units as units
import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor

from utils import CalculatorFactory

import json
import os
import socket
import time
from datetime import datetime
from math import prod


class PhononCalculator:
    def __init__(self, args):
        self.args = args
        
        # Формируем путь: results/phonons/<family>/<variant>/<use_type>
        parts = [            
            args.family,
            args.variant,
            args.use_type,
            args.system,
        ]
        if args.dft_method:
            parts.append(args.dft_method)

        self.output_dir = Path(args.workdir) / Path(*parts)        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
        with open(self.output_dir / "command.txt", "w") as f:
            f.write(" ".join(sys.argv))
        
        self.atoms = None
        self.calculator = None
        self.ph = None

        self.metrics = {
            "calculator": args.calculator,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "family": args.family,
            "variant": args.variant,
            "use_type": args.use_type,
            "system": args.system,
            "n_atoms": None,
            "n_displacements": None,
            "n_atoms_supercell": None,
            "use_mean_minimum_images": None,
            "supercell": args.supercell,
            "device": args.device,
            "dtype": args.dtype,
            "gpu_name": args.gpu_name,
            "time_relax_sec": None,
            "n_steps_relax": None,
            "relax_converged": None,
            "time_phonons_sec": None,
            "n_displacements": None,
            "timestamp": datetime.now().isoformat(),
            "hostname": socket.gethostname(),
        }

    def save_metrics(self):
        metrics_file = self.output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to {metrics_file}")
        
    def load_structure(self):
        print(f"Loading structure from {self.args.cif}")
        
        parser = CifParser(self.args.cif)
        structure = parser.parse_structures(primitive=False)[0]
        structure.remove_oxidation_states()
        
        self.atoms = AseAtomsAdaptor().get_atoms(structure)
        print(f"Loaded {len(self.atoms)} atoms")
        self.metrics["n_atoms"] = len(self.atoms)
        
    def setup_calculator(self):
        print(f"Setting up calculator: {self.args.calculator}")
        
        factory = CalculatorFactory()
        factory.configure(
            model_path=self.args.model_path,
            model_name=self.args.model_name if self.args.model_name else None,
            device=self.args.device,
            dtype=self.args.dtype
        )
        
        self.calculator = factory.calculator(self.args.calculator)
        self.atoms.calc = self.calculator
        print("Calculator ready")
        
    def relax_structure(self):
        if self.args.skip_relax:
            print("Skipping relaxation")
            return
            
        print("Starting relaxation...")
        
        self.atoms.set_constraint(FixSymmetry(self.atoms))
        
        cf = FrechetCellFilter(
            self.atoms, 
            scalar_pressure=float(self.args.pressure * units.GPa),
            hydrostatic_strain=True
        )

        dyn = BFGS(cf, logfile=str(self.output_dir / "relaxation_log.dat"))

        t0 = time.perf_counter()
        converged = dyn.run(fmax=self.args.fmax, steps=self.args.max_steps)
        t1 = time.perf_counter()

        self.metrics["time_relax_sec"] = float(t1 - t0)
        self.metrics["n_steps_relax"] = int(dyn.nsteps)
        self.metrics["relax_converged"] = bool(converged)
        self.metrics["fmax_target"] = float(self.args.fmax)
        self.metrics["max_steps_limit"] = int(self.args.max_steps)

        print(f"Relaxation finished: {dyn.nsteps} steps, {t1 - t0:.2f} sec, "
            f"converged={converged}")

        write(self.output_dir / "relaxed_structure.cif", self.atoms)
        
    def calculate_phonons(self):
        print("Starting phonon calculation...")
        
        self.atoms.constraints = []
        
        supercell = tuple(map(int, self.args.supercell.split(',')))
        
        self.ph = Phonons(
            self.atoms,
            self.calculator,
            supercell=supercell,
            delta=self.args.delta,
        )

        n_indices = len(self.ph.indices)
        n_disp = 1 + 6 * n_indices
        n_atoms_supercell = len(self.atoms) * prod(supercell)

        t0 = time.perf_counter()
        self.ph.run()
        t1 = time.perf_counter()

        self.metrics["time_phonons_sec"] = float(t1 - t0)
        self.metrics["n_displacements"] = int(n_disp)
        self.metrics["n_atoms_supercell"] = n_atoms_supercell

        print(f"Phonons done: {t1 - t0:.2f} sec, "
            f"displacements={n_disp}, atoms_per_call={n_atoms_supercell}")

        self.ph.read(acoustic=True)
        self.ph.clean()
        
    def compute_dos(self):
        print("Computing DOS...")
        
        kpts = tuple(map(int, self.args.kpts_dos.split(',')))
        
        dos = self.ph.get_dos(kpts=kpts).sample_grid(
            npts=self.args.dos_npts, 
            width=self.args.dos_width
        )
        
        energies = dos.get_energies()
        weights = dos.get_weights()
        
        if self.args.use_thz:
            conv = 241.799
            energies_conv = energies * conv
            energy_unit = "THz"
        else:
            energies_conv = energies
            energy_unit = "eV"
        
        dos_file = self.output_dir / "phonon_dos.dat"
        with open(dos_file, 'w') as f:
            f.write(f"# Energy ({energy_unit})    DOS\n")
            for e, w in zip(energies_conv, weights):
                f.write(f"{e:.6f}    {w:.6f}\n")
        
        plt.figure(figsize=(8, 6))
        plt.plot(energies_conv, weights, 'b-', lw=2)
        plt.fill_between(energies_conv, weights, alpha=0.3)
        plt.xlabel(f'Energy ({energy_unit})')
        plt.ylabel('DOS')
        plt.title('Phonon DOS')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "phonon_dos.png", dpi=300)
        plt.close()
        
        print(f"DOS saved to {dos_file}")

    def parse_bandpath_with_permutations(self, bandpath_str, x_permutation):
        """
        Обрабатывает путь с повторяющимися X для FCC.
        
        Args:
            bandpath_str: строка пути (например "GXXGL")
            x_permutation: тип перестановки ("type1" или "type2")
        
        Returns:
            (список координат, список меток) или (None, None)
        """
        # Проверяем условия для специальной обработки
        if self.args.system != "fcc_Al":
            return None, None
        if x_permutation is None:
            return None, None
        if "XX" not in bandpath_str:
            if x_permutation:
                print(f"Warning: --x-permutation={x_permutation} specified but no consecutive 'XX' in band path '{bandpath_str}', ignoring")
            return None, None
        
        print(f"Processing FCC band path with X permutation: {bandpath_str}, type={x_permutation}")
        
        # Получаем координаты специальных точек из ASE
        special_points = self.atoms.cell.get_bravais_lattice().get_special_points()
        print(f"Special points from ASE: {special_points}")
        
        # Определяем перестановки
        def permute_type1(x):
            return (x[1], x[2], x[0])
        
        def permute_type2(x):
            return (x[2], x[0], x[1])

        def permute_type3(x):
            return (x[0], x[1], -x[2])
        
        permute_func = permute_type1 if x_permutation == "type1" else permute_type2 if x_permutation == "type2" else permute_type3
        
        chars = list(bandpath_str)
        result_coords = []
        result_labels = []
        prev_char = None
        
        x_original = special_points.get('X')
        if x_original is None:
            print("Warning: 'X' point not found in special points, using standard path")
            return None, None
        
        for i, char in enumerate(chars):
            if char not in special_points and char != 'X':
                print(f"Warning: Unknown special point '{char}', using standard path")
                return None, None
            
            if char != prev_char:
                if char == 'X':
                    result_coords.append(tuple(x_original))
                else:
                    result_coords.append(tuple(special_points[char]))
                result_labels.append(char)
            else:
                if char == 'X':
                    permuted = permute_func(x_original)
                    result_coords.append(permuted)
                    result_labels.append('X')
                    print(f"  Applied {x_permutation} to X at position {i}: {x_original} -> {permuted}")
                else:
                    result_coords.append(result_coords[-1])
                    result_labels.append(char)
                    print(f"  Duplicated {char} at position {i}")
            
            prev_char = char
        
        print(f"  Generated {len(result_coords)} k-points")
        print(f"  Coordinates: {result_coords}")
        print(f"  Labels: {result_labels}")
        return result_coords, result_labels
        
    def compute_bands(self):
        print("Computing band structure...")
        
        custom_path, custom_labels = self.parse_bandpath_with_permutations(
            self.args.band_path, 
            self.args.x_permutation
        )
        
        if custom_path is not None:
            bandpath = self.atoms.cell.bandpath(custom_path, npoints=self.args.band_npoints)
            print(f"Using custom band path with {len(custom_path)} k-points")
            
            all_kpts = bandpath.kpts
            
            distances = []
            for i in range(1, len(all_kpts)):
                diff = all_kpts[i] - all_kpts[i-1]
                diff_cart = np.dot(diff, self.atoms.cell.reciprocal())
                dist = np.linalg.norm(diff_cart)
                distances.append(dist)
            
            x_coords = np.cumsum([0] + distances)
            
            special_positions = []
            special_labels = []
            start_idx = 0
            
            for i, special_kpt in enumerate(custom_path):
                min_dist = float('inf')
                closest_idx = start_idx
                for j in range(start_idx, len(all_kpts)):
                    diff_kpt = np.array(special_kpt) - all_kpts[j]
                    dist_kpt = np.linalg.norm(diff_kpt)
                    if dist_kpt < min_dist:
                        min_dist = dist_kpt
                        closest_idx = j
                special_positions.append(x_coords[closest_idx])
                special_labels.append(custom_labels[i])
                start_idx = closest_idx + 1
                print(f"Special point {custom_labels[i]} with coord {special_kpt} -> closest at index {closest_idx}, x={x_coords[closest_idx]:.6f}, distance={min_dist:.6f}")
            
            print(f"Special positions: {special_positions}")
            print(f"Special labels: {special_labels}")    
            
        else:
            bandpath = self.atoms.cell.bandpath(
                self.args.band_path, 
                npoints=self.args.band_npoints
            )
            print(f"Using standard band path: {self.args.band_path}")
            x_coords, special_positions, special_labels = bandpath.get_linear_kpoint_axis()
            all_kpts = bandpath.kpts
        
        bs = self.ph.get_band_structure(bandpath)
        bands = bs.energies
        bands = bands.real
        
        if bands.ndim == 3:
            bands = np.squeeze(bands)
        if bands.shape[0] == len(x_coords):
            bands = bands.T
        
        conv = 241.799 if self.args.use_thz else 1
        energy_unit = "THz" if self.args.use_thz else "eV"
        
        bands_file = self.output_dir / "phonon_bands.dat"
        with open(bands_file, 'w') as f:
            header = "# q_path"
            for ib in range(bands.shape[0]):
                header += f"   band_{ib+1:02d}({energy_unit})"
            f.write(header + "\n")
            
            for iq in range(len(x_coords)):
                line = f"{x_coords[iq]:12.6f}"
                for ib in range(bands.shape[0]):
                    val = bands[ib, iq].real * conv
                    line += f"  {val:12.6f}"
                f.write(line + "\n")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i in range(bands.shape[0]):
            ax.plot(x_coords, bands[i, :] * conv, 'b-', lw=1)
        
        for x in special_positions:
            ax.axvline(x=x, color='k', linestyle='--', alpha=0.5)
        
        ax.set_xticks(special_positions)
        ax.set_xticklabels(special_labels)
        ax.set_xlim(x_coords[0], x_coords[-1])
        ax.set_ylabel(f'Energy ({energy_unit})')
        ax.set_xlabel('Wave vector')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "phonon_bands.png", dpi=300)
        plt.savefig(self.output_dir / "phonon_bands.pdf")
        plt.close()
        
        print(f"Band structure saved to {bands_file}")
            
    def run(self):
        print("="*50)
        print("PHONON CALCULATION WORKFLOW")
        print("="*50)
        
        self.load_structure()
        self.setup_calculator()
        self.relax_structure()
        self.calculate_phonons()
        self.compute_dos()
        self.compute_bands()
        self.save_metrics()
        
        print("="*50)
        print(f"CALCULATION COMPLETED")
        print(f"Results saved in: {self.output_dir}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--calculator", required=True, choices=["mace", "grace", "deepmd"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--cif", required=True)
    parser.add_argument("--system", required=True)
    
    # Новые аргументы для иерархии
    parser.add_argument("--family", required=True, choices=["dpa4", "dpa4c", "grace"])
    parser.add_argument("--variant", required=True, choices=["air", "mini", "nano", "neo", "plus", "2L", "3L"])
    parser.add_argument("--use-type", required=True, choices=["pretrained", "fit", "finetune", "finetune_half"])
    parser.add_argument("--dft-method", default="", help="DFT method: pbe, scan, sol, or '' (flat JSON)")

    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])

    parser.add_argument("--gpu-name", default="CPU", help="Human-readable GPU name")
    
    parser.add_argument("--workdir", default="/home/katkoff/deepmd_check_pipline/models_check/results/phonons", 
                        help="Working directory for results")
    
    parser.add_argument("--skip-relax", action="store_true", help="Skip relaxation")
    parser.add_argument("--fmax", type=float, default=0.0001)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--pressure", type=float, default=0.0)
    
    parser.add_argument("--supercell", default="6,6,4")
    parser.add_argument("--delta", type=float, default=0.01)
    
    parser.add_argument("--kpts-dos", default="50,50,50")
    parser.add_argument("--dos-width", type=float, default=2e-4)
    parser.add_argument("--dos-npts", type=int, default=1000)
    
    parser.add_argument("--band-path", default="GKMGA")
    parser.add_argument("--band-npoints", type=int, default=200)
    parser.add_argument("--x-permutation", default=None, choices=["type1", "type2", "type3"],
                        help="Permutation type for consecutive X points in FCC (type1 or type2)")
    
    parser.add_argument("--use-thz", action="store_true", default=True)
    parser.add_argument("--no-use-thz", dest="use_thz", action="store_false")
    
    args = parser.parse_args()
    
    phonon_calc = PhononCalculator(args)
    phonon_calc.run()


if __name__ == "__main__":
    main()