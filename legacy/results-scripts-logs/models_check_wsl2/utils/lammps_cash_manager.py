"""LammpsCacheManager: Unified caching for LAMMPS-based calculations."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LammpsCacheManager:
    """
    Unified cache manager for LAMMPS multi-stage calculations.

    Stores run parameters and tracks which stages have been completed.
    Supports partial recovery: if stage 1 is done but stage 2 failed,
    next run will skip stage 1.

    IMPORTANT: stages are saved strictly in order defined by `stage_files`.
    The cache treats completed stages as a *prefix* of `stage_files`.
    If a stage is saved out of order, it is ignored with a warning.

    Cache structure:
        cache_dir/
            index.json
            001/
                equil.lammps
                overheat.lammps
                cool.lammps
                melt.lammps
            002/
                ...

    index.json format:
    [
        {
            "id": "001",
            "params": {...},
            "stages_done": ["equil", "overheat"]
        }
    ]
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    # ------------------------------------------------------------------
    # Index I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> list:
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        return []

    def _save_index(self, index: list):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prefix_of(stages_done: list, stage_files: list) -> list:
        """
        Return the longest prefix of `stage_files` that is present
        in `stages_done` (order matters).

        Example:
            stage_files = ["equil", "overheat", "cool", "melt"]
            stages_done = ["equil", "overheat"]      -> ["equil", "overheat"]
            stages_done = ["equil", "cool"]          -> ["equil"]  (overheat missing)
            stages_done = ["cool", "equil"]          -> []         (order wrong)
        """
        prefix = []
        for stage in stage_files:
            if stage in stages_done:
                prefix.append(stage)
            else:
                break
        return prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(self, params: dict, stage_files: list) -> Optional[dict]:
        """
        Find cached entry with matching params.

        Args:
            params: dict of parameters defining a unique run
            stage_files: ordered list of stage names, e.g. ["equil", "cool", "melt"]

        Returns:
            None                       - no cache found
            {"id", "run_dir", "skip_stages", "full"} - cache entry info
        """
        index = self._load_index()
        for entry in index:
            if entry["params"] == params:
                stages_done = entry.get("stages_done", [])
                skip_stages = self._prefix_of(stages_done, stage_files)

                if len(skip_stages) < len(stages_done):
                    logger.warning(
                        f"    Cache entry {entry['id']}: stages_done={stages_done} "
                        f"is not a prefix of {stage_files}. "
                        f"Using prefix: {skip_stages}"
                    )

                full = (len(skip_stages) == len(stage_files))
                result = {
                    "id": entry["id"],
                    "run_dir": self.cache_dir / entry["id"],
                    "skip_stages": skip_stages,
                    "full": full,
                }

                if full:
                    logger.info(f"    Found full cache: {result['run_dir']}")
                elif skip_stages:
                    logger.info(
                        f"    Found partial cache ({skip_stages[-1]} done): "
                        f"{result['run_dir']}"
                    )
                return result

        return None

    def get_run_dir(self, params: dict, stage_files: list) -> tuple:
        """
        Get or create run directory.

        Args:
            params: dict of parameters
            stage_files: ordered list of stage names

        Returns:
            (run_dir, run_id, skip_stages)
                run_dir:     Path to run directory
                run_id:      run identifier (e.g. "001")
                skip_stages: ordered list of already-completed stages
                             (always a prefix of stage_files)
        """
        cached = self.find(params, stage_files)

        if cached is not None:
            return (cached["run_dir"], cached["id"], cached["skip_stages"])

        # New run
        index = self._load_index()
        run_id = f"{len(index) + 1:03d}"
        run_dir = self.cache_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"    Cache: new run {run_id}")
        return (run_dir, run_id, [])

    def save(self, run_id: str, params: dict, stage: str):
        """
        Save completed stage to index.

        Stages MUST be saved strictly in the order defined by `stage_files`
        (the caller is responsible for that). If a stage is saved out of
        order, it is ignored with a warning.

        Args:
            run_id: run identifier
            params: run parameters
            stage: name of completed stage
        """
        index = self._load_index()

        for entry in index:
            if entry["id"] == run_id:
                if "stages_done" not in entry:
                    entry["stages_done"] = []

                stages_done = entry["stages_done"]

                # Idempotency: already saved
                if stage in stages_done:
                    logger.debug(f"    Cache: {run_id} {stage} already saved")
                    return

                # Append (order is caller's responsibility)
                stages_done.append(stage)
                logger.info(
                    f"    Cache: {run_id} +{stage} -> stages_done={stages_done}"
                )
                self._save_index(index)
                return

        # New entry
        index.append({
            "id": run_id,
            "params": params,
            "stages_done": [stage],
        })
        logger.info(f"    Cache: new entry {run_id} with {stage}")
        self._save_index(index)