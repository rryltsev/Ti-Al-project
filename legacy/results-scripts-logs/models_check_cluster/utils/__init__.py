'''
from utils.calculator import CalculatorFactory
from utils.structure import StructureFactory
from utils.slab import SlabFactory
from utils.layer import LayerAnalyzer

from utils.bulk_modulus import get_bulk_modulus
from utils.checkpoint_manager import CheckpointManager
from utils.trajectory_analyzer import TrajectoryAnalyzer
from utils.md_runner import MDRunner
from utils.liquid_generator import LiquidStructureGenerator
from utils.liquid_manager import LiquidStructureManager

from utils.melting.interface_analyzer import InterfaceAnalyzer

from utils.melting.temperature_scan_manager_lammps import LammpsTemperatureScanManager
from utils.melting.temperature_point_runner_lammps import LammpsTemperaturePointRunner

from utils.lammps_callback import ExternalASEForce
'''

__all__ = [
    # Core
    "CalculatorFactory",
    "StructureFactory",
    # Slab / layer
    "SlabFactory",
    "LayerAnalyzer",
    # Properties
    "get_bulk_modulus",
    "CheckpointManager",
    "TrajectoryAnalyzer",
    "MDRunner",
    # Liquid
    "LiquidStructureGenerator",
    "LiquidStructureManager",
    # Melting
    "InterfaceAnalyzer",
    "LammpsTemperatureScanManager",
    "LammpsTemperaturePointRunner",
    # LAMMPS
    "ExternalASEForce",
]


def __getattr__(name):
    if name == "CalculatorFactory":
        from .calculator import CalculatorFactory
        return CalculatorFactory
    elif name == "StructureFactory":
        from .structure import StructureFactory
        return StructureFactory
    elif name == "SlabFactory":
        from .slab import SlabFactory
        return SlabFactory
    elif name == "LayerAnalyzer":
        from .layer import LayerAnalyzer
        return LayerAnalyzer
    elif name == "get_bulk_modulus":
        from .bulk_modulus import get_bulk_modulus
        return get_bulk_modulus
    elif name == "CheckpointManager":
        from .checkpoint_manager import CheckpointManager
        return CheckpointManager
    elif name == "TrajectoryAnalyzer":
        from .trajectory_analyzer import TrajectoryAnalyzer
        return TrajectoryAnalyzer
    elif name == "MDRunner":
        from .md_runner import MDRunner
        return MDRunner
    elif name == "LiquidStructureGenerator":
        from .liquid_generator import LiquidStructureGenerator
        return LiquidStructureGenerator
    elif name == "LiquidStructureManager":
        from .liquid_manager import LiquidStructureManager
        return LiquidStructureManager
    elif name == "InterfaceAnalyzer":
        from .melting.interface_analyzer import InterfaceAnalyzer
        return InterfaceAnalyzer
    elif name == "LammpsTemperatureScanManager":
        from .melting.temperature_scan_manager_lammps import LammpsTemperatureScanManager
        return LammpsTemperatureScanManager
    elif name == "LammpsTemperaturePointRunner":
        from .melting.temperature_point_runner_lammps import LammpsTemperaturePointRunner
        return LammpsTemperaturePointRunner
    elif name == "ExternalASEForce":
        from .lammps_callback import ExternalASEForce
        return ExternalASEForce
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__