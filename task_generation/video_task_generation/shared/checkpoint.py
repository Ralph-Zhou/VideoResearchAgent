"""Stage-level checkpoint for resumable processing.

Each stage writes a `<key>\t<status>` line per processed item; on re-run,
already-processed keys are skipped.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

_PathLike = Union[str, Path]


class StageCheckpoint:
    """Append-only tab-separated checkpoint file."""

    def __init__(self, path: Optional[_PathLike]):
        self.path: Optional[Path] = Path(path) if path else None
        self._done: Dict[str, str] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                if "\t" in line:
                    key, status = line.split("\t", 1)
                else:
                    key, status = line, "done"
                self._done[key] = status
        logger.info("checkpoint loaded: %d items from %s", len(self._done), self.path)

    @property
    def processed_count(self) -> int:
        return len(self._done)

    def is_processed(self, key: str) -> bool:
        return key in self._done

    def get_status(self, key: str) -> Optional[str]:
        return self._done.get(key)

    def mark(self, key: str, status: str = "done") -> None:
        """Append a new record (idempotent by key)."""
        if self.path is None:
            return
        with self._lock:
            if key in self._done:
                return
            self._done[key] = status
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"{key}\t{status}\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
