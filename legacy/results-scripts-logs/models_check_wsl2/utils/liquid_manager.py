#models_check/utils/liquid_manager.py
"""
LiquidStructureManager: Generate and manage liquid structures for MD calculations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from pymatgen.core import Composition

from utils.liquid_generator import LiquidStructureGenerator

logger = logging.getLogger(__name__)


class LiquidStructureManager:
    """
    Менеджер жидких структур: генерирует чистые элементы и сплавы.
    
    Usage:
        manager = LiquidStructureManager(output_dir, n_atoms=200)
        paths = manager.process(["TiAl", "Ti3Al", "Al3Ti"])
        # paths = {
        #     "pure": {"Ti": Path, "Al": Path},
        #     "alloys": {"TiAl": Path, "Ti3Al": Path, "Al3Ti": Path}
        # }
    """
    
    def __init__(
        self,
        output_dir: Path,
        n_atoms: int = 200,
        r_min: float = 2.0,
        maxtry_per_atom: int = 250,
        seed: Optional[int] = None,
    ):
        """
        Args:
            output_dir: Куда сохранять POSCAR'ы
            n_atoms: Число атомов в каждой структуре
            r_min: Минимальное расстояние между атомами (Å)
            maxtry_per_atom: Максимальное число попыток размещения атома
            seed: Seed для воспроизводимости
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_atoms = n_atoms
        self.r_min = r_min
        self.maxtry_per_atom = maxtry_per_atom
        self.seed = seed
        
        self.generator = LiquidStructureGenerator(
            r_min=r_min,
            maxtry_per_atom=maxtry_per_atom,
            seed=seed,
        )
        
        logger.info(f"LiquidStructureManager initialized")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  n_atoms: {n_atoms}")
        logger.info(f"  r_min: {r_min} Å")
    
    def _extract_pure_elements(self, formulas: List[str]) -> Set[str]:
        """Извлекает уникальные элементы из списка формул."""
        elements = set()
        for formula in formulas:
            comp = Composition(formula)
            for el in comp.elements:
                elements.add(str(el))
        return elements
    
    def generate_pure(self, element: str) -> Path:
        """
        Генерирует жидкую структуру чистого элемента.
        
        Args:
            element: Химический символ (напр. "Ti", "Al")
        
        Returns:
            Путь к сохранённому POSCAR
        """
        filepath = self.output_dir / f"{element}.vasp"
        
        # Проверяем, не существует ли уже
        if filepath.exists():
            logger.info(f"Pure {element} already exists: {filepath}")
            return filepath
        
        logger.info(f"Generating liquid {element} ({self.n_atoms} atoms)...")
        
        atoms, counts = self.generator.generate(element, total_atoms=self.n_atoms)
        atoms.write(str(filepath), format='vasp', direct=True, vasp5=True)
        
        logger.info(f"  Saved: {filepath} (composition: {counts})")
        return filepath
    
    def generate_alloy(self, formula: str) -> Path:
        """
        Генерирует жидкую структуру сплава.
        
        Args:
            formula: Химическая формула (напр. "TiAl", "Ti3Al")
        
        Returns:
            Путь к сохранённому POSCAR
        """
        # Очищаем имя файла
        safe_name = formula.replace(" ", "_")
        filepath = self.output_dir / f"{safe_name}.vasp"
        
        if filepath.exists():
            logger.info(f"Alloy {formula} already exists: {filepath}")
            return filepath
        
        logger.info(f"Generating liquid {formula} ({self.n_atoms} atoms)...")
        
        atoms, counts = self.generator.generate(formula, total_atoms=self.n_atoms)
        atoms.write(str(filepath), format='vasp', direct=True, vasp5=True)
        
        logger.info(f"  Saved: {filepath} (composition: {counts})")
        return filepath
    
    def process(self, formulas: List[str]) -> Dict[str, Dict[str, Path]]:
        """
        Обрабатывает список формул: генерирует чистые элементы и сплавы.
        
        Args:
            formulas: Список формул (напр. ["TiAl", "Ti3Al", "Al3Ti"])
        
        Returns:
            {
                "pure": {"Ti": Path, "Al": Path},
                "alloys": {"TiAl": Path, "Ti3Al": Path, "Al3Ti": Path}
            }
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {len(formulas)} formulas")
        logger.info(f"{'='*50}")
        
        # Шаг 1: Извлекаем чистые элементы
        pure_elements = self._extract_pure_elements(formulas)
        logger.info(f"Pure elements found: {pure_elements}")
        
        # Шаг 2: Определяем сплавы (всё что не чистый элемент)
        alloys = [f for f in formulas if f not in pure_elements]
        logger.info(f"Alloys: {alloys}")
        
        # Шаг 3: Генерируем чистые
        pure_paths = {}
        for el in sorted(pure_elements):
            pure_paths[el] = self.generate_pure(el)
        
        # Шаг 4: Генерируем сплавы
        alloy_paths = {}
        for formula in alloys:
            alloy_paths[formula] = self.generate_alloy(formula)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Generation complete!")
        logger.info(f"  Pure elements: {len(pure_paths)}")
        logger.info(f"  Alloys: {len(alloy_paths)}")
        logger.info(f"{'='*50}")
        
        return {
            "pure": pure_paths,
            "alloys": alloy_paths,
        }


# ======================== ТЕСТ ========================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    output = Path("/tmp/test_liquid_structures")
    
    manager = LiquidStructureManager(
        output_dir=output,
        n_atoms=100,
        seed=42,
    )
    
    formulas = ["TiAl", "Ti3Al", "Al3Ti"]
    paths = manager.process(formulas)
    
    print(f"\nResults:")
    print(f"  Pure: {paths['pure']}")
    print(f"  Alloys: {paths['alloys']}")