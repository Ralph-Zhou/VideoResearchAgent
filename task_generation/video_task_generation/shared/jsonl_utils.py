"""Safe JSONL read/append helpers — tolerant to partial lines from crashes."""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

_PathLike = Union[str, Path]

# Process-level lock registry, keeping concurrent writes to one file safe.
_FILE_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_LOCK = threading.Lock()


def _get_lock(path: _PathLike) -> threading.Lock:
    key = str(Path(path).resolve())
    with _LOCKS_LOCK:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[key] = lock
        return lock


def load_jsonl_safe(path: _PathLike) -> List[Dict[str, Any]]:
    """Load JSONL, silently skipping invalid/truncated lines."""
    path = Path(path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("skip malformed JSONL line in %s: %s", path, exc)
    return out


def load_jsonl(path: _PathLike) -> List[Dict[str, Any]]:
    """Strict load — raises if file missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return load_jsonl_safe(path)


def append_jsonl_to_path(rec: Dict[str, Any], path: _PathLike) -> None:
    """Append one dict as a line to JSONL, creating the file/dir if needed.

    Thread-safe within the current process; cross-process use is not needed
    here (one pipeline process per output dir).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_lock(path)
    line = json.dumps(rec, ensure_ascii=False)
    with lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
