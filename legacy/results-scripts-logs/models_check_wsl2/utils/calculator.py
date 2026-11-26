from pathlib import Path
from typing import Optional, Union, Literal, overload, Any
import logging

logger = logging.getLogger(__name__)

class CalculatorFactory:
    """
    Фабрика для создания калькуляторов с условным импортом.
    """
    
    def __init__(self):
        """Инициализация фабрики. Путь и имя модели передаются позже."""
        self._model_path: Optional[Path] = None
        self._model_name: Optional[str] = None
        self._device: str = "cpu"
        self._dtype: str = "float32"
        self._calculator: Optional[Any] = None
    
    def configure(
        self, 
        model_path: Union[str, Path], 
        model_name: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "float32"
    ) -> 'CalculatorFactory':
        """
        Настройка параметров модели.
        
        Args:
            model_path: Путь к директории с моделью
            model_name: Имя модели
            device: Устройство для вычислений ("cpu", "cuda", etc.)
            dtype: Тип данных ("float32", "float64", etc.)
        
        Returns:
            self для цепочки вызовов
        """
        
        self._model_path = Path(model_path)
        self._model_name = model_name
        self._device = device
        self._dtype = dtype
        self._calculator = None  # Сбрасываем кэш при изменении параметров
        return self
    
    def calculator(
        self,
        calc_type: Literal["mace", "grace", "deepmd"]
    ) -> Any:
        """
        Создание калькулятора указанного типа.
        
        Args:
            calc_type: Тип калькулятора ("mace", "grace" или "deepmd")
        
        Returns:
            Экземпляр калькулятора
        
        Raises:
            ValueError: Если модель не настроена или тип неизвестен
            ImportError: Если не удалось импортировать требуемый модуль
        """
        if self._model_path is None and self._model_name is None:
            raise ValueError("Model not configured. Call .configure() first")
        
        if self._model_name:
            model_full_path = self._model_path / self._model_name
        else:
            model_full_path = self._model_path
        
        if calc_type == "mace":
            return self._create_mace_calculator(model_full_path)
        elif calc_type == "grace":
            return self._create_grace_calculator(model_full_path)
        elif calc_type == "deepmd":
            return self._create_deepmd_calculator(model_full_path)
        else:
            raise ValueError(f"Unknown calculator type: {calc_type}")
    
    def _create_mace_calculator(self, model_path: Path) -> Any:
        """Создание MACE калькулятора с условным импортом."""
        try:
            # Условный импорт, как в примере
            from mace.calculators import mace_mp
            return mace_mp(
                model=str(model_path), 
                device=self._device, 
                default_dtype=self._dtype
            )
        except ImportError as e:
            raise ImportError(
                f"Failed to import mace. Make sure it's installed. Error: {e}"
            )
    
    def _create_grace_calculator(self, model_path: Path) -> Any:
        """Создание GRACE калькулятора с условным импортом."""
        try:
            from tensorpotential.calculator import TPCalculator
            return TPCalculator(str(model_path))
        except ImportError as e:
            raise ImportError(
                f"Failed to import tensorpotential. Make sure it's installed. Error: {e}"
            )
    
    def _create_deepmd_calculator(self, model_path: Path) -> Any:
        """Создание DeepMD калькулятора."""
        try:
            from deepmd.calculator import DP

            logger.info("=== DEEPMD DP CREATION START ===")
            logger.info(f"model_path: {model_path}")
            logger.info(f"device: {self._device}")
            logger.info(f"default_dtype: {self._dtype}")
            logger.info("DeepMD DP imported successfully")
            logger.info("Calling DP(...)")

            try:
                calc = DP(
                    model=str(model_path),
                    device=self._device,
                    default_dtype=self._dtype
                )
            except Exception:
                logger.exception("!!! DP(...) FAILED WITH PYTHON EXCEPTION !!!")
                raise

            logger.info("DP(...) returned successfully")
            logger.info(f"calculator type: {type(calc)}")

            return calc

        except ImportError as e:
            logger.exception("!!! DEEPMD IMPORT FAILED !!!")
            raise ImportError(
                f"Failed to import deepmd. Make sure it's installed. Error: {e}"
            ) from e
