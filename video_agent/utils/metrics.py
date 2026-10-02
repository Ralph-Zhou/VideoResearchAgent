"""Token and timing metrics utilities."""

import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def timer():
    """Context manager that yields a dict with elapsed_ms after exiting."""
    result: Dict[str, float] = {}
    start = time.time()
    yield result
    result["elapsed_ms"] = (time.time() - start) * 1000
