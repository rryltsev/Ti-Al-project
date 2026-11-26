"""PointResultLammps: Container for LAMMPS temperature point result."""

from pathlib import Path
from typing import Optional


class PointResultLammps:
    """Result container for a single temperature point."""

    def __init__(self, T: float):
        self.T = T
        self.equil_path: Optional[Path] = None
        self.overheat_path: Optional[Path] = None
        self.cool_path: Optional[Path] = None
        self.melt_path: Optional[Path] = None

    @property
    def is_complete(self) -> bool:
        return self.melt_path is not None and self.melt_path.exists()

    def to_dict(self) -> dict:
        return {
            "T_K": self.T,
            "equil_path": str(self.equil_path) if self.equil_path else None,
            "overheat_path": str(self.overheat_path) if self.overheat_path else None,
            "cool_path": str(self.cool_path) if self.cool_path else None,
            "melt_path": str(self.melt_path) if self.melt_path else None,
        }