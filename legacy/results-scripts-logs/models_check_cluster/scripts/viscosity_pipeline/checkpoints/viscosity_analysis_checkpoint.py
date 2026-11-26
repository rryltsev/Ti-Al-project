"""Checkpoint для шага 04_viscosity_analysis."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from lib.checkpoint_base import BaseCheckpoint

logger = logging.getLogger(__name__)


class ViscosityAnalysisCheckpoint(BaseCheckpoint):
    """
    Чекпойнт для анализа вязкости.

    JSON:
    {
      "_meta": {
        "version": 1,
        "input_hash": "abc123",
        "params_hash": "def456",
        "params_snapshot": {...},
        "created_at": "...",
        "updated_at": "..."
      },
      "cases": {
        "c_Ti_0.00": {
          "status": "completed",
          "c_Ti": 0.0,
          "input_dir": ".../03_stress_autocorr/c_Ti_0.00",
          "output_dir": ".../04_viscosity_analysis/c_Ti_0.00",
          "n_snapshots": 50,
          "viscosity_mPa_s": 4.21,
          "viscosity_err_mPa_s": 0.15,
          "kinematic_visc_m2_s": 2.05e-6,
          "fit_params": {"A": ..., "alpha": ..., "tau1": ..., "tau2": ...},
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
    def init_meta(
        self,
        params_hash: str,
        input_hash: str,
        params_snapshot: dict = None,
    ) -> None:
        if self.META_KEY in self.data:
            return
        self.data[self.META_KEY] = {
            "version": self.VERSION,
            "input_hash": input_hash,
            "params_hash": params_hash,
            "params_snapshot": params_snapshot or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.data.setdefault(self.CASES_KEY, {})
        self.save()

    def get_params_hash(self) -> Optional[str]:
        return self.data.get(self.META_KEY, {}).get("params_hash")

    def get_input_hash(self) -> Optional[str]:
        return self.data.get(self.META_KEY, {}).get("input_hash")

    def check_hashes(self, params_hash: str, input_hash: str) -> tuple:
        """Возвращает (params_ok, input_ok)."""
        return (
            self.get_params_hash() == params_hash,
            self.get_input_hash() == input_hash,
        )

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
            output_dir = entry.get("output_dir")
            if not output_dir:
                return False
            output_dir = Path(output_dir)
            for fname in ("result.txt", "summary.json", "visc_vs_time.png"):
                if not (output_dir / fname).exists():
                    logger.warning(
                        f"Checkpoint says completed, but {output_dir / fname} missing"
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
        n_snapshots_requested: int,
        n_snapshots_used: int,
        viscosity_mPa_s: float,
        kinematic_visc_m2_s: float,
        fit_params: dict,
        T: float,
    ) -> None:
        self._cases()[case_name] = {
            "status": "completed",
            "c_Ti": c_Ti,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "n_snapshots_requested": n_snapshots_requested,
            "n_snapshots_used": n_snapshots_used,
            "viscosity_mPa_s": viscosity_mPa_s,
            "kinematic_visc_m2_s": kinematic_visc_m2_s,
            "fit_params": fit_params,
            "T": T,
            "completed_at": datetime.now().isoformat(),
        }        
        self._touch()
        self.save()
        logger.info(
            f"Checkpoint: {case_name} -> completed "
            f"(eta = {viscosity_mPa_s:.3f} mPa*s)"
        )

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
# Helpers
# ------------------------------------------------------------------
def compute_params_hash(params: dict) -> str:
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def compute_input_hash(prev_checkpoint_path: Path) -> str:
    """MD5 от значимых полей чекпойнта предыдущего шага."""
    with open(prev_checkpoint_path, "r", encoding="utf-8") as f:
        prev_data = json.load(f)

    cases = prev_data.get("cases", {})
    significant = {}
    for case_name, entry in cases.items():
        significant[case_name] = {
            "status": entry.get("status"),
            "completed_viscs": sorted(entry.get("completed_viscs", [])),
            "n_snapshots": entry.get("n_snapshots"),
        }
    s = json.dumps(significant, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]