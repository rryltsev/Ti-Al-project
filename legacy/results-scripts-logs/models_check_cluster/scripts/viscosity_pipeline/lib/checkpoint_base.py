"""Base checkpoint: JSON I/O + atomic save. Никакой логики о "completed"."""

import json
import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class BaseCheckpoint:
    """
    Минимальный чекпойнт: JSON I/O + atomic save.

    Наследники определяют:
      - структуру self.data
      - методы is_completed / mark_completed / mark_failed
      - что значит "валидная запись"
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict = {}
        self._load()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info(f"Checkpoint loaded: {self.path} ({len(self.data)} keys)")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Checkpoint corrupted ({e}), starting fresh")
            self.data = {}

    def save(self) -> None:
        """Atomic save: пишем в .tmp, потом os.replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)