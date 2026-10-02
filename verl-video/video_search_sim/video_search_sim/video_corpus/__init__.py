"""Offline video-corpus preparation pipeline.

Public surface:
    - ``build_corpus(config, output_dir, max_videos=None)``: orchestrates the
      full pipeline end-to-end.
    - ``VideoRecord`` (re-exported from ``common.schemas``): the row shape of
      the produced ``videos.parquet``.
"""

from ..common.schemas import VideoRecord
from .pipeline import build_corpus

__all__ = ["VideoRecord", "build_corpus"]
