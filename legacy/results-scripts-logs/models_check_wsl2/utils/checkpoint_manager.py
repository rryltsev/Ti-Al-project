#models_check/utils/checkpoint_manager.py
"""CheckpointManager: Save and restore calculation progress."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage calculation checkpoints for resuming interrupted runs.
    
    Usage:
        cpm = CheckpointManager("checkpoint.json")
        if cpm.is_completed("TiAl", 2000):
            result = cpm.load_result("TiAl", 2000)
        else:
            # run calculation...
            cpm.mark_completed("TiAl", 2000, result)
    """
    
    def __init__(self, checkpoint_path: Path):
        self.path = Path(checkpoint_path)
        self.data: Dict = {}
        self._load()
    
    def _load(self) -> None:
        if self.path.exists():
            with open(self.path, 'r') as f:
                self.data = json.load(f)
            logger.info(f"Loaded checkpoint: {len(self.data)} entries")
    
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def _make_key(self, composition: str, temperature: float) -> str:
        return f"{composition}_T{int(temperature)}K"
    
    def is_completed(self, composition: str, temperature: float) -> bool:
        """Check if calculation is already done."""
        key = self._make_key(composition, temperature)
        return key in self.data and self.data[key].get("status") == "completed"
    
    def is_relax_done(self, composition: str) -> bool:
        key = f"{composition}_relax"
        return key in self.data and self.data[key].get("status") == "completed"

    def mark_relax_done(self, composition: str, cache_path: str) -> None:
        key = f"{composition}_relax"
        self.data[key] = {
            "status": "completed",
            "cache_path": cache_path,
            "completed_at": datetime.now().isoformat(),
        }
        self._save()

    def load_relax_cache(self, composition: str) -> Optional[str]:
        key = f"{composition}_relax"
        entry = self.data.get(key)
        return entry.get("cache_path") if entry else None
    
    def mark_completed(
        self, composition: str, temperature: float, result: Dict
    ) -> None:
        """Mark full calculation as completed."""
        key = self._make_key(composition, temperature)
        self.data[key] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            **result,
        }
        self._save()
        logger.info(f"Checkpoint saved: {key}")
    
    def load_result(self, composition: str, temperature: float) -> Optional[Dict]:
        """Load saved result."""
        key = self._make_key(composition, temperature)
        return self.data.get(key)