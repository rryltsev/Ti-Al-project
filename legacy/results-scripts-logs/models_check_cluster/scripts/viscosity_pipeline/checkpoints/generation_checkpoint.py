"""Checkpoint для шага 01_generate_liquid."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from lib.checkpoint_base import BaseCheckpoint

logger = logging.getLogger(__name__)


class GenerationCheckpoint(BaseCheckpoint):
    """
    Чекпойнт для генерации структур.

    JSON:
    {
      "_meta": {
        "version": 1,
        "params_hash": "abc123456789",
        "created_at": "...",
        "updated_at": "..."
      },
      "cases": {
        "c_Ti_0.00": {
          "status": "completed",
          "c_Ti": 0.0,
          "n_atoms": 4096,
          "data_path": ".../c_Ti_0.00/data.lammps",
          "completed_at": "..."
        },
        "c_Ti_0.50": {
          "status": "failed",
          "c_Ti": 0.5,
          "error": "...",
          "failed_at": "..."
        }
      }
    }
    """

    VERSION = 1
    META_KEY = "_meta"
    CASES_KEY = "cases"

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------
    def init_meta(self, params_hash: str) -> None:
        """Создать _meta, если её ещё нет."""
        if self.META_KEY in self.data:
            return
        self.data[self.META_KEY] = {
            "version": self.VERSION,
            "params_hash": params_hash,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.data.setdefault(self.CASES_KEY, {})
        self.save()

    def get_params_hash(self) -> Optional[str]:
        return self.data.get(self.META_KEY, {}).get("params_hash")

    def check_params_hash(self, current_hash: str) -> bool:
        """True если хэш совпадает (можно переиспользовать записи)."""
        saved = self.get_params_hash()
        if saved is None:
            return True
        return saved == current_hash

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------
    def _cases(self) -> dict:
        return self.data.setdefault(self.CASES_KEY, {})

    def is_completed(self, case_name: str, verify_paths: bool = True) -> bool:
        entry = self._cases().get(case_name)
        if not entry or entry.get("status") != "completed":
            return False
        if verify_paths:
            p = entry.get("data_path")
            if not p or not Path(p).exists():
                logger.warning(
                    f"Checkpoint says completed, but {p} missing — will recompute"
                )
                return False
        return True

    def is_failed(self, case_name: str) -> bool:
        entry = self._cases().get(case_name)
        return bool(entry and entry.get("status") == "failed")

    def mark_completed(
        self,
        case_name: str,
        c_Ti: float,
        data_path: Path,
        n_atoms: int,
    ) -> None:
        self._cases()[case_name] = {
            "status": "completed",
            "c_Ti": c_Ti,
            "n_atoms": n_atoms,
            "data_path": str(data_path),
            "completed_at": datetime.now().isoformat(),
        }
        self._touch()
        self.save()
        logger.info(f"Checkpoint: {case_name} -> completed")

    def mark_failed(self, case_name: str, c_Ti: float, error: str) -> None:
        self._cases()[case_name] = {
            "status": "failed",
            "c_Ti": c_Ti,
            "error": error,
            "failed_at": datetime.now().isoformat(),
        }
        self._touch()
        self.save()
        logger.warning(f"Checkpoint: {case_name} -> failed ({error})")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        cases = self._cases()
        completed = [k for k, v in cases.items() if v.get("status") == "completed"]
        failed = [k for k, v in cases.items() if v.get("status") == "failed"]
        return {
            "total": len(cases),
            "completed": len(completed),
            "failed": len(failed),
            "completed_list": sorted(completed),
            "failed_list": sorted(failed),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _touch(self) -> None:
        if self.META_KEY in self.data:
            self.data[self.META_KEY]["updated_at"] = datetime.now().isoformat()


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def compute_params_hash(params: dict) -> str:
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]