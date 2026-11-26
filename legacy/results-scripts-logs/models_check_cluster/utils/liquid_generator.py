# liquid_generator.py
import numpy as np
from numba import njit
from ase import Atoms
from pymatgen.core import Composition
from typing import Union


@njit
def lcg_random(seed):
    """Линейный конгруэнтный генератор с изменяемым состоянием"""
    a = 1103515245
    c = 12345
    m = 2**31
    seed[0] = (a * seed[0] + c) % m
    return seed[0] / m


@njit
def check_overlap_numba(new_pos, positions, count, L, r_min_sq):
    """Numba-ускоренная проверка перекрытий"""
    for j in range(count):
        dx = new_pos[0] - positions[j, 0]
        dy = new_pos[1] - positions[j, 1]
        dz = new_pos[2] - positions[j, 2]
        
        dx -= L * np.round(dx / L)
        dy -= L * np.round(dy / L)
        dz -= L * np.round(dz / L)
        
        if dx*dx + dy*dy + dz*dz < r_min_sq:
            return True
    return False


@njit
def generate_positions_numba(natoms, L, r_min, maxtry, seed_val):
    """Полностью numba-ускоренная версия"""
    r_min_sq = r_min**2
    positions = np.zeros((natoms, 3))
    
    seed = np.array([seed_val], dtype=np.int64)
    
    for i in range(natoms):
        placed = False
        
        for attempt in range(maxtry):
            rx = lcg_random(seed)
            ry = lcg_random(seed)
            rz = lcg_random(seed)
            
            new_pos = np.array([rx * L, ry * L, rz * L])
            
            if i == 0 or not check_overlap_numba(new_pos, positions, i, L, r_min_sq):
                positions[i] = new_pos
                placed = True
                break
        
        if not placed:
            return None
    
    return positions


def generate_positions(natoms, L, r_min, maxtry, seed=None):
    """Интерфейсная функция"""
    if seed is None:
        seed = np.random.randint(0, 2**31)
    return generate_positions_numba(natoms, L, r_min, maxtry, int(seed))


def create_atoms_random(elements, r_min=2.0, maxtry_per_atom=250, 
                        initial_packing=0.5, seed=None, max_enlarge=50):
    """
    Полный алгоритм с увеличением L.
    """
    total_atoms = sum(elements.values())
    symbols = []
    for elem, count in elements.items():
        symbols.extend([elem] * count)
    
    radius = r_min / 2.0
    volume_per_atom = (4.0/3.0) * np.pi * radius**3 / initial_packing
    total_volume = total_atoms * volume_per_atom
    L_initial = total_volume**(1.0/3.0)
    
    enlarge_factor = 1.0
    
    for enlarge_step in range(max_enlarge):
        current_L = L_initial * enlarge_factor
        
        positions = generate_positions(
            total_atoms, current_L, r_min, maxtry_per_atom, seed
        )
        
        if positions is not None:
            return symbols, positions, current_L
        
        enlarge_factor *= 1.1
    
    raise RuntimeError(f"Не удалось разместить атомы после {max_enlarge} попыток")


class LiquidStructureGenerator:
    """
    Генератор случайных жидких структур.
    """
    
    def __init__(self, r_min: float = 2.0, maxtry_per_atom: int = 250, 
                 seed: Union[int, None] = None):
        self.r_min = r_min
        self.maxtry_per_atom = maxtry_per_atom
        self.seed = seed
    
    def generate(self, composition_str: str, total_atoms: int = 200) -> tuple:
        """
        Генерирует случайную жидкую структуру.
        
        Returns:
            tuple: (atoms, counts_dict) - ASE Atoms объект и словарь с количеством атомов
        """
        comp = Composition(composition_str)
        
        fractions = {str(el): comp.get_atomic_fraction(el) for el in comp.elements}
        
        total_frac = sum(fractions.values())
        if abs(total_frac - 1.0) > 1e-6:
            for el in fractions:
                fractions[el] /= total_frac
        
        counts = {}
        remaining = total_atoms
        
        elements = list(fractions.keys())
        for i, el in enumerate(elements[:-1]):
            counts[el] = int(round(total_atoms * fractions[el]))
            remaining -= counts[el]
        
        counts[elements[-1]] = remaining
        
        if any(c < 0 for c in counts.values()):
            raise ValueError(f"Ошибка распределения атомов: {counts}")
        
        elements_dict = {el: counts[el] for el in counts if counts[el] > 0}
        
        symbols, positions, L = create_atoms_random(
            elements=elements_dict,
            r_min=self.r_min,
            maxtry_per_atom=self.maxtry_per_atom,
            seed=self.seed
        )
        
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=[L, L, L],
            pbc=True,
        )
        atoms.wrap()
        
        return atoms, elements_dict