"""
LayerAnalyzer: Universal layer analyzer and mask generator for slab structures.
"""

import logging
import numpy as np
from typing import Optional, Literal, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from ase import Atoms

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """Information about a single atomic layer."""
    id: int
    indices: np.ndarray
    z_mean: float
    spread: float
    n_atoms: int
    chem: Dict[str, int]


@dataclass
class MaskResult:
    """Result of mask generation with metadata."""
    mask: np.ndarray
    mode: str
    part: str
    n_layers_total: int
    n_layers_selected: int
    layer_ids: List[int]
    z_cut: Optional[float] = None
    description: str = ""


class LayerAnalyzer:
    """
    Universal layer analyzer for slab structures.
    
    Works in fractional coordinates to handle tetragonal/hexagonal cell tilts.
    
    Usage:
        analyzer = LayerAnalyzer(tol=0.01)
        result = analyzer.get_mask(atoms, mode="cut", fraction=0.5, part="top")
        pos[result.mask] += shift_vector
    """
    
    def __init__(
        self,
        tol: float = 0.01,
        require_even_split: bool = False,
    ):
        """
        Initialize analyzer.
        
        Args:
            tol: Tolerance for grouping atoms into layers (in fractional Z).
            require_even_split: If True, warn when top/bottom atom counts differ.
        """
        self.tol = tol
        self.require_even_split = require_even_split
        
        # Cached analysis results
        self._atoms: Optional[Atoms] = None
        self._layers: List[LayerInfo] = []
        self._n_layers: int = 0
    
    def analyze(self, atoms: Atoms) -> List[LayerInfo]:
        """
        Analyze slab and build layer structure.
        
        Args:
            atoms: ASE Atoms object (should be centered along Z)
        
        Returns:
            List of LayerInfo objects, one per atomic layer
        """
        self._atoms = atoms
        
        # Work in fractional coordinates
        frac_coords = atoms.get_scaled_positions()
        z_frac = frac_coords[:, 2]
        
        # Sort by fractional Z
        order = np.argsort(z_frac)
        
        # Build layers
        raw_layers = []
        current = [order[0]]
        z_frac_ref = z_frac[order[0]]
        
        for idx in order[1:]:
            if abs(z_frac[idx] - z_frac_ref) < self.tol:
                current.append(idx)
            else:
                raw_layers.append(current)
                current = [idx]
                z_frac_ref = z_frac[idx]
        raw_layers.append(current)
        
        self._n_layers = len(raw_layers)
        
        if self._n_layers < 2:
            raise RuntimeError(
                f"Too few layers detected: {self._n_layers}. "
                f"Try adjusting tol (current: {self.tol})."
            )
        
        # Build layer info
        self._layers = []
        
        for lid, inds in enumerate(raw_layers):
            inds_arr = np.array(inds)
            
            # Cartesian Z for logging
            zz_cart = atoms.positions[inds_arr, 2]
            z_mean = np.mean(zz_cart)
            spread = np.ptp(zz_cart)
            
            # Chemical composition
            species = np.array(atoms.symbols)[inds_arr]
            uniq, counts = np.unique(species, return_counts=True)
            chem = {str(u): int(c) for u, c in zip(uniq, counts)}
            
            self._layers.append(LayerInfo(
                id=lid,
                indices=inds_arr,
                z_mean=z_mean,
                spread=spread,
                n_atoms=len(inds_arr),
                chem=chem,
            ))
        
        self._log_analysis()
        
        return self._layers
    
    def _log_analysis(self):
        """Log layer analysis results."""
        logger.info("\n" + "=" * 80)
        logger.info("LAYER ANALYSIS (fractional Z -> cartesian logs)")
        logger.info("=" * 80)
        
        for layer in self._layers:
            logger.info(
                f"Layer {layer.id:3d} | "
                f"atoms={layer.n_atoms:3d} | "
                f"z={layer.z_mean:10.6f} Å | "
                f"spread={layer.spread:8.6f} Å | "
                f"chem={layer.chem}"
            )
        
        logger.info("=" * 80)
    
    def _select_layers(
        self,
        fraction: Optional[float] = None,
        n_layers: Optional[int] = None,
        part: Literal["top", "bottom"] = "top",
        slice_pos: Optional[Literal["top", "bottom", "center"]] = None,
    ) -> List[int]:
        """
        Select layer indices based on criteria.
        
        Returns:
            List of selected layer indices (always sorted by Z)
        """
        if self._n_layers == 0:
            raise RuntimeError("No layers analyzed. Call .analyze() first.")
        
        total = self._n_layers
        
        # Step 1: Select by fraction
        if fraction is not None:
            n_select = max(1, int(round(total * fraction)))
        elif n_layers is not None:
            n_select = min(n_layers, total)
        else:
            n_select = total
        
        # Step 2: Choose top or bottom
        if part == "top":
            pool = list(range(total - n_select, total))
        else:  # bottom
            pool = list(range(n_select))
        
        # Step 3: Apply slice within pool
        if slice_pos is None or n_layers is None:
            selected = pool
        else:
            n_slice = min(n_layers, len(pool))
            if slice_pos == "top":
                selected = pool[-n_slice:]
            elif slice_pos == "bottom":
                selected = pool[:n_slice]
            elif slice_pos == "center":
                start = (len(pool) - n_slice) // 2
                selected = pool[start:start + n_slice]
            else:
                selected = pool
        
        return sorted(selected)
    
    def get_mask(
        self,
        atoms: Atoms,
        mode: Literal["cut", "select"] = "cut",
        fraction: Optional[float] = None,
        part: Literal["top", "bottom"] = "top",
        n_layers: Optional[int] = None,
        slice_pos: Optional[Literal["top", "bottom", "center"]] = None,
    ) -> MaskResult:
        """
        Generate boolean mask for selected atoms.
        
        Args:
            atoms: ASE Atoms object
            mode: "cut" (between layers) or "select" (whole layers)
            fraction: Fraction of total layers to select (0.0 to 1.0)
            part: "top" or "bottom"
            n_layers: Exact number of layers (overrides fraction for count)
            slice_pos: For mode="select", which slice of selected layers
                       ("top", "bottom", "center", or None for all)
        
        Returns:
            MaskResult with boolean mask and metadata
        """
        # Analyze if not cached or different atoms
        if self._atoms is None or len(self._atoms) != len(atoms):
            self.analyze(atoms)
        
        total = self._n_layers
        
        # Default fraction
        if fraction is None and n_layers is None:
            fraction = 0.5
        
        # Get selected layer indices
        selected_ids = self._select_layers(
            fraction=fraction,
            n_layers=n_layers,
            part=part,
            slice_pos=slice_pos,
        )
        
        # Build mask
        if mode == "cut":
            # Cut BETWEEN layers: include layers from selected and above
            if part == "top":
                cut_after = selected_ids[0] - 1
                # Top: everything ABOVE cut_after
                layer_ids = list(range(cut_after + 1, total))
            else:
                cut_after = selected_ids[-1]
                # Bottom: everything UP TO cut_after
                layer_ids = list(range(cut_after + 1))
            
            # Calculate cut Z position
            if part == "top":
                lower = self._layers[cut_after]
                upper = self._layers[cut_after + 1]
            else:
                lower = self._layers[cut_after]
                upper = self._layers[cut_after + 1] if cut_after + 1 < total else self._layers[cut_after]
            
            z_cut = 0.5 * (lower.z_mean + upper.z_mean)
            
        elif mode == "select":
            # Select exact layers
            layer_ids = selected_ids
            z_cut = None
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Build boolean mask
        mask = np.zeros(len(atoms), dtype=bool)
        for lid in layer_ids:
            mask[self._layers[lid].indices] = True
        
        n_top = int(np.sum(mask))
        n_bottom = len(atoms) - n_top
        
        # Log
        logger.info("\n" + "=" * 80)
        logger.info(f"MASK: mode={mode}, part={part}, layers={len(layer_ids)}/{total}")
        if z_cut is not None:
            logger.info(f"Cut plane: z = {z_cut:.6f} Å")
        logger.info(f"Selected atoms: {n_top} / {len(atoms)}")
        logger.info(f"Layer IDs: {layer_ids}")
        
        if self.require_even_split and n_top != n_bottom:
            logger.warning(
                f"Uneven split! Top: {n_top}, Bottom: {n_bottom}"
            )
        
        # Sanity check
        for layer in self._layers:
            mask_values = mask[layer.indices]
            if not (np.all(mask_values) or np.all(~mask_values)):
                raise RuntimeError(
                    f"Layer {layer.id} was split across the fault plane!"
                )
        
        logger.info("=" * 80)
        
        # Build description
        desc_parts = [f"{mode}"]
        if fraction is not None:
            desc_parts.append(f"fraction={fraction:.2f}")
        if n_layers is not None:
            desc_parts.append(f"n={n_layers}")
        desc_parts.append(part)
        if slice_pos:
            desc_parts.append(f"slice={slice_pos}")
        
        return MaskResult(
            mask=mask,
            mode=mode,
            part=part,
            n_layers_total=total,
            n_layers_selected=len(layer_ids),
            layer_ids=layer_ids,
            z_cut=z_cut,
            description=" | ".join(desc_parts),
        )
    
    @property
    def n_layers(self) -> int:
        """Total number of layers detected."""
        return self._n_layers
    
    @property
    def layers(self) -> List[LayerInfo]:
        """List of layer information."""
        return self._layers


# ======================== ТЕСТ ========================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Quick test with dummy slab
    from ase.build import fcc111
    
    atoms = fcc111('Al', size=(2, 2, 6), vacuum=10.0)
    atoms.center(axis=2)
    
    print(f"\nTest slab: {len(atoms)} atoms")
    
    analyzer = LayerAnalyzer(tol=0.01)
    
    # Test modes
    for mode in ["cut", "select"]:
        for part in ["top", "bottom"]:
            result = analyzer.get_mask(
                atoms,
                mode=mode,
                fraction=0.5,
                part=part,
            )
            print(f"\n{mode} {part}: {result.n_layers_selected}/{result.n_layers_total} layers")
            print(f"  Z cut: {result.z_cut}")
            print(f"  Mask sum: {result.mask.sum()}/{len(atoms)}")
    
    # Test select with slice
    print("\n--- Select with slice ---")
    result = analyzer.get_mask(
        atoms,
        mode="select",
        fraction=0.5,
        part="top",
        n_layers=2,
        slice_pos="center",
    )
    print(f"Top half, 2 center layers: {result.layer_ids}")
    print(f"  Mask sum: {result.mask.sum()}/{len(atoms)}")