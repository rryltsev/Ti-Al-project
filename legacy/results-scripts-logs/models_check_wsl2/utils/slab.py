# utils/slab.py

import logging
import numpy as np
from typing import Tuple, Optional, Literal
from pymatgen.core import Structure
from pymatgen.core.surface import SlabGenerator

logger = logging.getLogger(__name__)


class SlabFactory:
    """
    Фабрика для генерации слэбов из bulk-структур.
    
    Usage:
        slab_factory = SlabFactory()
        slab, bulk = slab_factory.create(
            bulk_structure, 
            miller_index=(1,1,1), 
            n_layers=12, 
            vacuum=12.0
        )
    """
    
    def __init__(self):
        """Инициализация фабрики слэбов."""
        self._slab_gen: Optional[SlabGenerator] = None
    
    def create(
        self,
        bulk: Structure,
        miller_index: Tuple[int, int, int] = (1, 1, 1),
        n_layers: int = 12,
        vacuum: float = 12.0,
        center_slab: bool = True,
        primitive: bool = True,
        in_unit_planes: bool = True,
        lll_reduce: bool = True,
        max_normal_search: int = 10,
        symmetrize: bool = True,
        filter_out_sym_slabs: bool = False,
        slab_index: int = 0,
    ) -> Tuple[Structure, Structure]:
        """
        Создать слэб из bulk-структуры.
        
        Args:
            bulk: Исходная bulk-структура (pymatgen Structure)
            miller_index: Индексы Миллера плоскости среза
            n_layers: Число слоёв
            vacuum: Размер вакуума в Å
            center_slab: Центрировать слэб
            primitive: Использовать примитивную ячейку в плоскости
            in_unit_planes: Задавать размер в единицах hkl-плоскостей
            lll_reduce: LLL-редукция для чистых углов
            max_normal_search: Максимальный поиск нормали
            symmetrize: Симметризовать верх/низ слэба
            slab_index: Индекс слэба из списка (0 = первый)
        
        Returns:
            (slab, bulk): Кортеж (слэб, исходный bulk)
        """
        logger.info(
            f"Generating slab: miller={miller_index}, "
            f"layers={n_layers}, vacuum={vacuum} Å"
        )
        
        self._slab_gen = SlabGenerator(
            bulk,
            miller_index,
            min_slab_size=n_layers,
            min_vacuum_size=vacuum,
            center_slab=center_slab,
            primitive=primitive,
            in_unit_planes=in_unit_planes,
            lll_reduce=lll_reduce,
            max_normal_search=max_normal_search
        )
        
        slabs = self._slab_gen.get_slabs(
            symmetrize=symmetrize,
            filter_out_sym_slabs=filter_out_sym_slabs
        )
        
        logger.info(f"Generated {len(slabs)} slab variant(s)")
        for i, s in enumerate(slabs):
            logger.debug(
                f"  Slab #{i}: {s.composition}, "
                f"{len(s)} atoms, symmetric: {s.is_symmetric}"
            )
        
        slab = slabs[slab_index]
        
        logger.info(f"Selected slab #{slab_index}: {slab.composition}, {len(slab)} atoms")
        logger.info(f"Slab cell: {slab.lattice.abc}")
        logger.info(f"slab_scale_factor:\n{self._slab_gen.slab_scale_factor}")
        
        return slab, bulk
    
    @property
    def slab_gen(self) -> Optional[SlabGenerator]:
        """Последний использованный SlabGenerator (для доступа к scale_factor)."""
        return self._slab_gen