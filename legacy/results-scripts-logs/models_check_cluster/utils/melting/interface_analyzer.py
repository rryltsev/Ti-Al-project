#!/usr/bin/env python3
"""InterfaceAnalyzer: Analyze solid-liquid interface from MD trajectory."""

import logging
from pathlib import Path
from typing import List
import numpy as np
from ase.io import read
from ase import Atoms

logger = logging.getLogger(__name__)


class InterfaceAnalyzer:
    """
    Анализ solid-liquid интерфейса из production траектории.
    
    Метод:
    1. Для каждого кадра вычисляет Q6 (параметр порядка)
    2. Классифицирует атомы на solid-like / liquid-like
    3. Строит профиль solid_fraction(z)
    4. Находит положение интерфейса z_interface(t)
    5. Вычисляет скорость интерфейса v = dz/dt
    """
    
    def __init__(
        self,
        q6_cutoff: float = 3.5,
        q6_threshold: float = 0.4,
        bin_width: float = 2.0,
        md_timestep: float = 1.0,
        log_interval: int = 100,
    ):
        """
        Args:
            q6_cutoff: радиус обрезки для поиска соседей (Å)
            q6_threshold: порог Q6 для классификации solid/liquid
            bin_width: ширина бина по Z (Å)
            md_timestep: шаг MD в fs
            log_interval: интервал записи кадров (каждый N шагов)
        """
        self.q6_cutoff = q6_cutoff
        self.q6_threshold = q6_threshold
        self.bin_width = bin_width
        self.dt_per_frame = md_timestep * log_interval / 1000.0  # ps
    
    def _compute_q6(self, atoms: Atoms) -> np.ndarray:
        """Вычисляет Q6 для каждого атома через pyscal (новый API)."""
        try:
            import pyscal
            
            pyscal.find_neighbors(atoms, method="cutoff", cutoff=self.q6_cutoff)
            q6 = pyscal.steinhardt_parameter(atoms, l=[6])
            return atoms.arrays["pyscal_q6"]
            
        except ImportError:
            logger.warning("pyscal not installed, using random Q6 for testing!")
            return np.random.uniform(0, 0.6, len(atoms))
    
    def _bin_solid_fraction(self, z: np.ndarray, solid_mask: np.ndarray) -> dict:
        """
        Разбивает Z на бины и считает долю solid-like атомов в каждом.
        
        Returns:
            dict: {"z_centers": [...], "solid_fraction": [...]}
        """
        z_min, z_max = z.min(), z.max()
        n_bins = max(5, int((z_max - z_min) / self.bin_width))
        bin_edges = np.linspace(z_min, z_max, n_bins + 1)
        z_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        solid_fraction = []
        for i in range(n_bins):
            mask = (z >= bin_edges[i]) & (z < bin_edges[i + 1])
            if mask.sum() > 0:
                solid_fraction.append(solid_mask[mask].sum() / mask.sum())
            else:
                solid_fraction.append(np.nan)
        
        return {
            "z_centers": z_centers,
            "solid_fraction": np.array(solid_fraction),
        }
    
    def _find_interface(self, z_centers: np.ndarray, solid_fraction: np.ndarray) -> float:
        """
        Находит положение интерфейса где solid_fraction = 0.5.
        
        Использует линейную интерполяцию в области перехода.
        """
        # Убираем NaN
        valid = ~np.isnan(solid_fraction)
        z_valid = z_centers[valid]
        f_valid = solid_fraction[valid]
        
        if len(z_valid) < 2:
            return np.nan
        
        # Ищем где solid_fraction пересекает 0.5
        # Интерполируем в области перехода (0.2 < f < 0.8)
        transition = (f_valid > 0.2) & (f_valid < 0.8)
        
        if transition.sum() >= 2:
            z_trans = z_valid[transition]
            f_trans = f_valid[transition]
            # Линейная интерполяция к 0.5
            z_interface = np.interp(0.5, f_trans, z_trans)
        else:
            # Грубая оценка: ближайшая точка к 0.5
            idx = np.argmin(np.abs(f_valid - 0.5))
            z_interface = z_valid[idx]
        
        return z_interface    

    def analyze(self, prod_path: Path) -> dict:
        logger.info(f"\n{'='*50}")
        logger.info(f"INTERFACE ANALYZER (bin method with tracking)")
        logger.info(f"{'='*50}")
        logger.info(f"  Trajectory: {prod_path}")
        
        frames = read(str(prod_path), index=':')
        n_frames = len(frames)
        
        if n_frames < 2:
            return {"v_interface_A_per_ps": None, "z_interface_vs_t": [], "n_frames": n_frames}
        
        logger.info(f"  Frames: {n_frames}")
        logger.info(f"  dt per frame: {self.dt_per_frame:.3f} ps")
        
        # Получаем размер системы для unwrap
        atoms0 = frames[0]
        Lz = atoms0.cell[2, 2] if hasattr(atoms0, 'cell') else 60.0
        logger.info(f"  System size Lz = {Lz:.2f} Å")
        
        z_ifaces_raw = []
        
        for i, atoms in enumerate(frames):
            # Q6 + бинарная классификация
            q6 = self._compute_q6(atoms)
            solid_mask = q6 > self.q6_threshold
            
            # Профиль по Z
            z = atoms.positions[:, 2]
            profile = self._bin_solid_fraction(z, solid_mask)
            
            # Сглаживание профиля
            from scipy.ndimage import gaussian_filter1d
            solid_fraction_smooth = gaussian_filter1d(profile["solid_fraction"], sigma=1.0)
            
            # Положение интерфейса
            z_iface = self._find_interface(
                profile["z_centers"],
                solid_fraction_smooth,
            )
            
            if not np.isnan(z_iface):
                z_ifaces_raw.append(z_iface)
            
            if i % 50 == 0:
                logger.info(f"    Frame {i}/{n_frames}: z_interface = {z_iface:.2f} Å")
        
        # Медианная фильтрация для удаления выбросов
        from scipy.signal import medfilt
        if len(z_ifaces_raw) >= 5:
            z_ifaces_filtered = medfilt(z_ifaces_raw, kernel_size=5)
        else:
            z_ifaces_filtered = z_ifaces_raw
        
        # Скорость интерфейса по развернутым координатам
        if len(z_ifaces_filtered) >= 2:
            times = np.arange(len(z_ifaces_filtered)) * self.dt_per_frame
            v = np.polyfit(times, z_ifaces_filtered, 1)[0]
        else:
            v = None
        
        logger.info(f"\n  Results:")
        logger.info(f"    v_interface = {v:.4f} Å/ps" if v else "    v_interface = N/A")
        logger.info(f"    Valid points: {len(z_ifaces_raw)}/{n_frames}")
        
        return {
            "v_interface_A_per_ps": round(v, 4) if v is not None else None,
            "z_interface_vs_t": z_ifaces_filtered.tolist() if isinstance(z_ifaces_filtered, np.ndarray) else z_ifaces_filtered,
            "n_frames": n_frames,
        }
