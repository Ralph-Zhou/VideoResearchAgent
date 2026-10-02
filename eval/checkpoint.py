"""Checkpoint management for resumable benchmark evaluation."""

import json
import logging
from pathlib import Path
from typing import Set, List, Dict

logger = logging.getLogger(__name__)


class EvalCheckpoint:
    """Track completed row_ids to enable resume-from-checkpoint evaluation."""

    def __init__(self, checkpoint_dir: str, run_name: str):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = Path(checkpoint_dir) / f"{run_name}_checkpoint.json"
        self.completed_ids: Set[str] = self._load()

    def _load(self) -> Set[str]:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                logger.error("Failed to load checkpoint file: %s", self.checkpoint_file)
                return set()
            count = len(data.get("completed", []))
            logger.info("Checkpoint loaded: %d already completed", count)
            return set(data["completed"])
        return set()

    def mark_done(self, row_id: str):
        self.completed_ids.add(str(row_id))
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({"completed": sorted(self.completed_ids)}, f)

    def is_done(self, row_id: str) -> bool:
        return str(row_id) in self.completed_ids

    def filter_pending(self, data: List[Dict]) -> List[Dict]:
        pending = [row for row in data if not self.is_done(str(row.get("row_id", "")))]
        skipped = len(data) - len(pending)
        if skipped > 0:
            logger.info("Checkpoint: skipping %d already completed items", skipped)
        return pending
