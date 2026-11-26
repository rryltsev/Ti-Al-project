"""Checkpoint для шага 03_stress_autocorr."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from lib.checkpoint_base import BaseCheckpoint

logger = logging.getLogger(__name__)


class StressAutocorrCheckpoint(BaseCheckpoint):
    """
    Чекпойнт для SACF-прогонов.

    JSON:
    {
      "_meta": {
        "version": 1,
        "params_hash": "abc123",
        "params_snapshot": {...},
        "created_at": "...",
        "updated_at": "..."
      },
      "cases": {
        "c_Ti_0.00": {
          "status": "completed",
          "c_Ti": 0.0,
          "input_dir": ".../02_nvt_snapshots/c_Ti_0.00",
          "output_dir": ".../03_stress_autocorr/c_Ti_0.00",
          "n_snapshots": 50,
          "completed_viscs": [1, 2, ..., 50],
          "T": 1943.0,
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

    def _recover_completed_viscs(self, entry: dict) -> list:
        """
        Восстановить список готовых visc{i} — либо из completed_viscs,
        либо из n_snapshots, либо сканированием output_dir/visc*/sacf_*.data.
        """
        # 1. Явное поле
        completed = entry.get("completed_viscs")
        if completed is not None:
            return sorted(completed)

        # 2. Из n_snapshots
        n_saved = entry.get("n_snapshots", 0)
        if n_saved > 0:
            return list(range(1, n_saved + 1))

        # 3. Скан output_dir
        output_dir = entry.get("output_dir")
        if not output_dir:
            return []
        output_dir = Path(output_dir)
        if not output_dir.exists():
            return []

        found = []
        for d in sorted(output_dir.glob("visc*")):
            if not d.is_dir():
                continue
            try:
                i = int(d.name.replace("visc", ""))
            except ValueError:
                continue
            sacf = d / f"sacf_{i}.data"
            if sacf.exists():
                found.append(i)
        return sorted(found)

    def is_completed(
        self,
        case_name: str,
        n_snapshots_required: int,
        verify_paths: bool = True,
    ) -> bool:
        entry = self._cases().get(case_name)
        if not entry or entry.get("status") != "completed":
            return False

        completed_viscs = self._recover_completed_viscs(entry)

        if len(completed_viscs) < n_snapshots_required:
            logger.info(
                f"{case_name}: only {len(completed_viscs)}/{n_snapshots_required} viscs done"
            )
            return False

        if verify_paths:
            output_dir = entry.get("output_dir")
            if not output_dir:
                logger.warning(f"{case_name}: no output_dir in checkpoint")
                return False
            output_dir = Path(output_dir)
            for i in range(1, n_snapshots_required + 1):
                sacf = output_dir / f"visc{i}" / f"sacf_{i}.data"
                if not sacf.exists():
                    logger.warning(
                        f"Checkpoint says completed, but {sacf} missing — will recompute"
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
        input_dir: Path,
        output_dir: Path,
        n_snapshots: int,
        completed_viscs: List[int],
        T: float,
    ) -> None:
        self._cases()[case_name] = {
            "status": "completed",
            "c_Ti": c_Ti,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "n_snapshots": n_snapshots,
            "completed_viscs": sorted(completed_viscs),
            "T": T,
            "completed_at": datetime.now().isoformat(),
        }
        self._touch()
        self.save()
        logger.info(f"Checkpoint: {case_name} -> completed ({len(completed_viscs)}/{n_snapshots})")

    def mark_failed(self, case_name: str, c_Ti: float, error: str) -> None:
        entry = self._cases().get(case_name, {})
        entry["status"] = "failed"
        entry["c_Ti"] = c_Ti
        entry["error"] = error
        entry["failed_at"] = datetime.now().isoformat()
        self._cases()[case_name] = entry
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