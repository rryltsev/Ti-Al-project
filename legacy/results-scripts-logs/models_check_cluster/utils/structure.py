"""
StructureFactory: Unified structure generator for ML potential calculations.
Loads structure configurations from JSON file.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Literal, Union, List, Dict, Any
from pymatgen.core import Structure, Lattice
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io.lammpsdata import write_lammps_data, read_lammps_data
    

logger = logging.getLogger(__name__)

class StructureFactory:
    """
    Фабрика для создания bulk-структур из JSON-конфига.
    
    Usage:
        factory = StructureFactory()
        factory.configure(system="gamma_AlTi")
        bulk_conv = factory.get_bulk(cell_type="conventional")
        bulk_prim = factory.get_bulk(cell_type="primitive")
    """
    
    # Путь к JSON-конфигу относительно этого файла
    DEFAULT_CONFIG_PATH = Path(__file__).parent / "structures_config.json"
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Инициализация фабрики.
        
        Args:
            config_path: Путь к JSON-файлу с конфигурациями структур.
                        Если None, используется DEFAULT_CONFIG_PATH.
        """
        self._config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._structures: Dict[str, Dict[str, Any]] = {}
        self._system: Optional[str] = None
        self._config: Optional[Dict[str, Any]] = None
        
        # Загружаем конфиг
        self._load_config()
    
    def _load_config(self) -> None:
        """Загрузить конфигурации структур из JSON-файла."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Structures config not found: {self._config_path}"
            )
        
        with open(self._config_path, 'r') as f:
            self._structures = json.load(f)
        
        logger.info(
            f"Loaded {len(self._structures)} structures from {self._config_path}"
        )
    
    def reload_config(self, config_path: Optional[Path] = None) -> 'StructureFactory':
        """
        Перезагрузить конфиг из файла.
        
        Args:
            config_path: Новый путь к конфигу (если None, используется текущий)
        """
        if config_path is not None:
            self._config_path = config_path
        
        self._load_config()
        self._system = None
        self._config = None
        return self
    
    def configure(self, system: str) -> 'StructureFactory':
        """
        Настройка на конкретную систему.
        
        Args:
            system: Имя системы (ключ в JSON-конфиге)
        
        Returns:
            self для цепочки вызовов
        
        Raises:
            ValueError: Если система не найдена в конфиге
        """
        if system not in self._structures:
            available = list(self._structures.keys())
            raise ValueError(
                f"Unknown system: '{system}'. Available: {available}"
            )
        
        self._system = system
        self._config = self._structures[system]
        
        logger.info(f"Configured for: {system}")
        return self
    
    @property
    def system(self) -> Optional[str]:
        """Текущая система."""
        return self._system
    
    @property
    def lattice_params(self) -> Dict[str, float]:
        """Параметры решётки текущей системы."""
        if self._config is None:
            raise ValueError("Factory not configured. Call .configure() first")
        return self._config["lattice_params"].copy()
    
    def _create_lattice(self, config: Dict[str, Any]) -> Lattice:
        """Создать объект Lattice из конфигурации."""
        lt = config["lattice_type"]
        params = config["lattice_params"]
        
        if lt == "cubic":
            return Lattice.cubic(params["a"])
        elif lt == "hexagonal":
            return Lattice.hexagonal(params["a"], params["c"])
        elif lt == "tetragonal":
            return Lattice.tetragonal(params["a"], params["c"])
        else:
            raise ValueError(f"Unknown lattice type: {lt}")
    
    def _validate(self, bulk: Structure, config: Dict[str, Any]) -> None:
        """
        Валидация созданной структуры.
        
        Args:
            bulk: Созданная структура
            config: Конфигурация из JSON
        """
        # Проверка числа атомов
        expected_n = config.get("expected_n_atoms")
        if expected_n is not None and len(bulk) != expected_n:
            logger.warning(
                f"Expected {expected_n} atoms, got {len(bulk)}. "
                f"This may be normal for primitive cells."
            )
        
        # Проверка состава
        expected_comp = config.get("expected_composition")
        if expected_comp is not None:
            actual_comp = str(bulk.composition)
            if actual_comp.replace(" ", "") != expected_comp.replace(" ", ""):
                logger.warning(
                    f"Expected composition '{expected_comp}', got '{actual_comp}'"
                )
        
        # Проверка уникальных Z-координат (для слоистых структур)
        z_coords = np.unique(np.round([s.frac_coords[2] for s in bulk], decimals=6))
        if len(z_coords) < 2:
            logger.debug(
                f"Only {len(z_coords)} distinct Z-layers (may be normal for simple cells)"
            )
        
        logger.debug(f"Validation passed for {self._system}")
    
    def get_bulk(
        self, 
        cell_type: Literal["conventional", "primitive"] = "conventional"
    ) -> Structure:
        """
        Создать bulk-структуру.
        
        Args:
            cell_type: Тип ячейки
                - "conventional": конвенциональная (как в JSON-конфиге)
                - "primitive": примитивная (минимальный объём)
        
        Returns:
            pymatgen Structure
        
        Raises:
            ValueError: Если фабрика не настроена
        """
        if self._config is None:
            raise ValueError(
                "Factory not configured. Call .configure(system) first"
            )
        
        config = self._config
        
        logger.info(
            f"Creating {cell_type} cell for '{self._system}' "
            f"({config['spacegroup']}, {config['lattice_type']})"
        )
        
        lattice = self._create_lattice(config)
        
        bulk = Structure.from_spacegroup(
            config["spacegroup"],
            lattice,
            config["species"],
            config["coords"]
        )
        
        logger.info(f"Generated bulk: {bulk.composition}, {len(bulk)} atoms")
        
        # Валидация конвенциональной ячейки
        if cell_type == "conventional":
            self._validate(bulk, config)
        
        # Примитивизация если нужно
        if cell_type == "primitive":
            logger.info("Converting to primitive cell...")
            primitive_bulk = bulk.get_primitive_structure()
            logger.info(
                f"Primitive cell: {primitive_bulk.composition}, "
                f"{len(primitive_bulk)} atoms"
            )
            return primitive_bulk
        
        return bulk
    
    def list_systems(self) -> List[str]:
        """Список доступных систем в конфиге."""
        return list(self._structures.keys())
    
    def get_info(self) -> Dict[str, Any]:
        """Информация о текущей конфигурации."""
        if self._config is None:
            return {"status": "not configured"}
        
        return {
            "system": self._system,
            "spacegroup": self._config["spacegroup"],
            "lattice_type": self._config["lattice_type"],
            "lattice_params": self._config["lattice_params"],
            "species": self._config["species"],
            "expected_composition": self._config.get("expected_composition"),
        }
    
    def get_structure(
        self, 
        system: str, 
        cell_type: Literal["conventional", "primitive"] = "conventional"
    ) -> Structure:
        """
        Быстрый метод: настроить и сразу получить структуру.
        
        Args:
            system: Имя системы
            cell_type: Тип ячейки
        
        Returns:
            pymatgen Structure
        """
        return self.configure(system).get_bulk(cell_type)

    def _sort_atoms_by_symbol(self, atoms: Atoms) -> Atoms:
        """Сортировка для стабильного порядка типов"""
        symbols = np.array(atoms.get_chemical_symbols())
        order = np.argsort(symbols)
        return atoms[order]

    def make_supercell(self, 
                       structure: Structure, 
                       scaling_matrix: Union[int, List[int], tuple, np.ndarray]) -> Structure:
        """Просто создаёт supercell из переданной структуры"""
        return structure.make_supercell(scaling_matrix)


    def get_ase_atoms(self, structure: Structure) -> Atoms:
        """Просто конвертер pymatgen Structure → ASE Atoms"""
        atoms = AseAtomsAdaptor.get_atoms(structure)
        # Сортируем для стабильности типов в LAMMPS
        atoms = self._sort_atoms_by_symbol(atoms)
        return atoms


    def get_ase_atoms_for_natoms(self, 
                                 structure: Structure,
                                 scaling_matrix: Union[int, List[int], tuple, np.ndarray],
                                 target_natoms: int) -> Atoms:
        """
        Создаёт supercell с заданным scaling_matrix и возвращает ASE Atoms.
        target_natoms используется только для лога/проверки.
        """
        # Determine supercell
        n_bulk = len(structure)        
        n = max(1, int(np.ceil((target_natoms / (3 * n_bulk)) ** (1/3))))
        nx, ny, nz = n, n, n
        supercell = np.array([nx, ny, nz]) * np.array(scaling_matrix)
        n_total = np.prod(supercell) * n_bulk
        logger.info(f"      Supercell: {supercell} → {n_total} atoms (≥ {target_natoms})")

        supercell_struct = self.make_supercell(structure, supercell)
        atoms = self.get_ase_atoms(supercell_struct)
        
        logger.info(f"Created supercell {supercell} → {len(atoms)} atoms (target was {target_natoms})")
        return atoms


    def write_lammps_data(self,
                          structure: Structure,
                          scaling_matrix: Union[int, List[int], tuple, np.ndarray],
                          target_natoms: int,
                          filename: Optional[str] = None,
                          directory: str = "lammps_structures",
                          specorder: Optional[List[str]] = None) -> Dict[str, Any]:
        """Записывает LAMMPS data файл"""
        atoms = self.get_ase_atoms_for_natoms(structure, scaling_matrix, target_natoms)

        if filename is None:
            formula = atoms.get_chemical_formula(empirical=True)
            n = len(atoms)
            filename = f"{formula}_{n}atoms.data"

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / filename

        if specorder is None:
            specorder = sorted(set(atoms.get_chemical_symbols()))

        write_lammps_data(
            str(filepath),
            atoms,
            specorder=specorder,
            units="metal",
            atom_style="atomic",
            masses=True,
            force_skew=True
        )

        type_map = {i+1: sym for i, sym in enumerate(specorder)}

        result = {
            "data_path": filepath,
            "atoms": atoms,
            "natoms": len(atoms),
            "formula": atoms.get_chemical_formula(empirical=True),
            "type_map": type_map,
            "symbol_to_type": {sym: i+1 for i, sym in enumerate(specorder)},
            "specorder": specorder
        }

        logger.info(f"LAMMPS data written → {filepath} ({len(atoms)} atoms)")
        return result

    def read_lammps_data(self, filepath, sort_by_id=True, 
                     units="metal", atom_style="atomic") -> Dict[str, Any]:
        """Читает LAMMPS data-файл и возвращает ASE Atoms + метаинформацию."""
        
        atoms = read_lammps_data(
            filepath, 
            sort_by_id=sort_by_id, 
            units=units, 
            atom_style=atom_style
        )
        
        specorder = sorted(set(atoms.get_chemical_symbols()))
        type_map = {i+1: sym for i, sym in enumerate(specorder)}
        
        logger.info(f"LAMMPS data read → {filepath} ({len(atoms)} atoms)")
        
        return {
            "data_path": Path(filepath),
            "atoms": atoms,
            "natoms": len(atoms),
            "formula": atoms.get_chemical_formula(empirical=True),
            "type_map": type_map,
            "symbol_to_type": {sym: i+1 for i, sym in enumerate(specorder)},
            "specorder": specorder
        }


# ======================== ТЕСТ ========================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    factory = StructureFactory()
    
    logger.info(f"Available systems: {factory.list_systems()}")
    
    for system in factory.list_systems():
        logger.info(f"{'='*60}")
        logger.info(f"System: {system}")
        logger.info(f"{'='*60}")
        
        try:
            factory.configure(system)
            info = factory.get_info()
            logger.info(f"  Spacegroup: {info['spacegroup']}")
            logger.info(f"  Lattice: {info['lattice_type']} {info['lattice_params']}")
            logger.info(f"  Species: {info['species']}")
            
            # Конвенциональная
            bulk_conv = factory.get_bulk(cell_type="conventional")
            logger.info(f"  Conventional: {bulk_conv.composition}, {len(bulk_conv)} atoms")
            
            # Примитивная
            bulk_prim = factory.get_bulk(cell_type="primitive")
            logger.info(f"  Primitive:    {bulk_prim.composition}, {len(bulk_prim)} atoms")

            factory.write_lammps_data(bulk_conv, (1, 1, 3), 2000, directory="/home/katkoff/fintune_pipline_grace/vasp_data")

        except Exception as e:
            logger.error(f"  ERROR: {e}")
    
    logger.info(f"{'='*60}")
    logger.info("All systems OK!")
    logger.info(f"{'='*60}")