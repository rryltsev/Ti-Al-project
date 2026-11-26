"""Checkpoint для шага 02_nvt_snapshots."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from lib.checkpoint_base import BaseCheckpoint

logger = logging.getLogger(__name__)


class SnapshotsCheckpoint(BaseCheckpoint):
    """
    Чекпойнт для NVT-прогона со снимками.

    JSON:
    {
      "_meta": {
        "version": 1,
        "params_hash": "abc123456789",
        "params_snapshot": {...},
        "created_at": "...",
        "updated_at": "..."
      },
      "cases": {
        "c_Ti_0.00": {
          "status": "completed",
          "c_Ti": 0.0,
          "input_data": ".../01_generate_liquid/c_Ti_0.00/data.lammps",
          "output_dir": ".../02_nvt_snapshots/c_Ti_0.00",
          "n_snapshots": 51,
          "first_snapshot": ".../c_Ti_0.00/data.0",
          "last_snapshot":  ".../c_Ti_0.00/data.50",
          "T": 1943.0,
          "P": 1.0,
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
    def init_meta(self, params_hash: str, params_snapshot: dict = None) -> None:
        if self.META_KEY in self.data:
            return
        self.data[self.META_KEY] = {
            "version": self.VERSION,
            "params_hash": params_hash,
            "params_snapshot": params_snapshot or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.data.setdefault(self.CASES_KEY, {})
        self.save()

    def get_params_hash(self) -> Optional[str]:
        return self.data.get(self.META_KEY, {}).get("params_hash")

    def get_params_snapshot(self) -> dict:
        return self.data.get(self.META_KEY, {}).get("params_snapshot", {})

    def check_params_hash(self, current_hash: str) -> bool:
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
            first = entry.get("first_snapshot")
            last = entry.get("last_snapshot")
            if not first or not Path(first).exists():
                logger.warning(
                    f"Checkpoint says completed, but {first} missing — will recompute"
                )
                return False
            if not last or not Path(last).exists():
                logger.warning(
                    f"Checkpoint says completed, but {last} missing — will recompute"
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
        input_data: Path,
        output_dir: Path,
        n_snapshots: int,
        first_snapshot: Path,
        last_snapshot: Path,
        T: float,
        P: float,
    ) -> None:
        self._cases()[case_name] = {
            "status": "completed",
            "c_Ti": c_Ti,
            "input_data": str(input_data),
            "output_dir": str(output_dir),
            "n_snapshots": n_snapshots,
            "first_snapshot": str(first_snapshot),
            "last_snapshot": str(last_snapshot),
            "T": T,
            "P": P,
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