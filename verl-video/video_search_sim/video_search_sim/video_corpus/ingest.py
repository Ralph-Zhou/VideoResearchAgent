"""HuggingFace -> VideoRecord ingestion.

This module is intentionally thin: its only job is to walk one or more
``HFSourceConfig`` entries, materialise each video on local disk (downloading
only when needed), and emit a ``VideoRecord`` per successful item.

The heavy lifting (keyframe extraction, CLIP encoding, BM25 indexing) lives in
sibling modules; this one cares only about *inventory*.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..common.config import HFSourceConfig
from ..common.fake_url import fake_url_from_path, fake_video_id
from ..common.schemas import VideoRecord

logger = logging.getLogger(__name__)


def _row_get(row: dict[str, Any], key: str | None, default: Any = None) -> Any:
    """Tolerant dict accessor: ``None`` key returns default."""
    if key is None:
        return default
    return row.get(key, default)


def _as_string(value: Any) -> str:
    """Coerce arbitrary dataset column values to a printable string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    return str(value)


def _as_scene_splits(value: Any) -> list[tuple[float, float]]:
    """Normalise ``scene_splits``-style columns into ``[(start, end), ...]``.

    Accepts:
      - list of dicts with ``{"start": .., "end": ..}`` keys
      - list of ``[start, end]`` pairs
      - ``None`` / empty -> ``[]``
    """
    if not value:
        return []
    out: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict) and "start" in item and "end" in item:
            out.append((float(item["start"]), float(item["end"])))
        elif isinstance(item, list | tuple) and len(item) == 2:
            out.append((float(item[0]), float(item[1])))
    return out


def _as_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def _resolve_video_path(raw: Any, cache_dir: Path) -> Path | None:
    """Given whatever the dataset stored in ``video_field``, return a local path.

    Handled cases:
      - str / os.PathLike pointing to an existing local file -> returned verbatim
      - HF ``Video`` feature that exposes ``.path`` or ``"path"`` key
      - None / missing -> returns None (caller skips)

    We do *not* download remote files here; datasets built with
    ``datasets.Video`` already cache to ``~/.cache/huggingface`` and expose a
    local path. If you need to pull from a remote mirror, do so in the
    source-specific loader and pass a local path through.
    """
    del cache_dir  # kept in the signature for forward-compat with remote download
    if raw is None:
        return None

    path: str | None = None
    if isinstance(raw, str | os.PathLike):
        path = str(raw)
    elif isinstance(raw, dict):
        # Common HF shapes: {"path": "/abs/..."} or {"bytes": ..., "path": ...}
        if "path" in raw and raw["path"]:
            path = str(raw["path"])
    else:
        # ``datasets.Video`` / ``VideoReader`` objects usually expose ``.path``
        path = getattr(raw, "path", None)

    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = p.resolve()
    return p


def _parse_timestamp(ts_str: str | None) -> float:
    """Convert a '00:00:28.779' timestamp to seconds (float)."""
    if not ts_str:
        return 0.0
    try:
        parts = ts_str.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return float(ts_str)
    except:
        return 0.0


def iter_records_from_source(
    source: any,  # source: HFSourceConfig
    cache_dir: Path,
    max_videos: int | None = 6000,
    *,
    cache_videos: bool = False,
    skip_video_ids: set[str] | None = None,
) -> Iterator[VideoRecord]:
    """
    Iteratively extract video data from locally downloaded Parquet shards
    and construct index records.

    By default (``cache_videos=False``), mp4 bytes are passed through in-memory
    via ``VideoRecord.video_bytes`` and never written to disk; downstream
    ``extract_keyframes`` decodes them directly from RAM. When
    ``cache_videos=True``, each mp4 is materialised under ``cache_dir`` and
    ``record.local_path`` points to it (legacy behaviour, useful for offline
    debugging).

    ``skip_video_ids`` filters out rows whose ``video_id`` (derived from
    ``original_video_filename``) already appears in a checkpointed batch,
    so resumed builds don't re-decode mp4 bytes for already-done videos.
    The filter is applied **before** mp4 bytes are materialised, which is
    where the bulk of the per-video cost lives.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "`datasets` library not installed. Please run `pip install datasets`"
        ) from e

    # 1. Determine local path. Resolution order:
    #    (a) source.local_parquet_dir from corpus.yaml
    #    (b) VSS_LOCAL_PARQUET_DIR environment variable
    #    (c) default './data/corpus/local_parquet/data' (relative to CWD)
    raw_dir = (
        getattr(source, "local_parquet_dir", None)
        or os.environ.get("VSS_LOCAL_PARQUET_DIR")
        or "./data/corpus/local_parquet/data"
    )
    local_data_path = Path(raw_dir).expanduser()
    local_files = sorted([str(f) for f in local_data_path.glob("*.parquet")])

    if not local_files:
        logger.error(
            f"No local Parquet files found in {local_data_path} "
            f"(resolved from {'config' if getattr(source, 'local_parquet_dir', None) else ('env VSS_LOCAL_PARQUET_DIR' if os.environ.get('VSS_LOCAL_PARQUET_DIR') else 'default')}). "
            f"Set 'local_parquet_dir' on the source in corpus.yaml, "
            f"export VSS_LOCAL_PARQUET_DIR, or run the download script first."
        )
        return

    logger.info(
        f"Loading {len(local_files)} shard files from {local_data_path}..."
    )

    # 2. Load local dataset
    # Use the 'parquet' loader to read local files directly
    try:
        ds = load_dataset(
            "parquet", data_files=local_files, split="train", streaming=True
        )
    except Exception as e:
        logger.error(f"Failed to load local dataset: {e}")
        return

    count = 0
    skipped_for_resume = 0
    for row in ds:
        # Extract binary video data and metadata
        video_bytes = row.get("mp4")
        meta = row.get("json")  # FineVideo metadata is nested within the 'json' field

        # Parse if the metadata is provided as a JSON string
        if isinstance(meta, str):
            meta = json.loads(meta)

        if not video_bytes or not meta:
            continue

        # 3. Determine video ID
        # Prioritize using the original filename, stripping the extension
        video_unique_id = meta.get("original_video_filename", f"v_{count}").replace(
            ".mp4", ""
        )

        # 3a. Resume short-circuit: skip videos already committed in a
        # previous run before we copy bytes / write to disk / decode frames.
        if skip_video_ids is not None and video_unique_id in skip_video_ids:
            skipped_for_resume += 1
            if skipped_for_resume % 1000 == 0:
                logger.info(
                    "Skipped %d already-committed videos so far (resume)",
                    skipped_for_resume,
                )
            continue

        # 4. Resolve the path used for fake_url derivation. Two modes:
        #    - cache_videos=True : write mp4 to cache_dir, record.local_path
        #      points at the real file, video_bytes is dropped (legacy mode).
        #    - cache_videos=False (default) : skip the write, use a stable
        #      virtual path 'fineVideo://<id>.mp4' as both the fake_url seed
        #      and the local_path string. The mp4 bytes ride along on
        #      record.video_bytes for keyframe extraction and are discarded
        #      by the pipeline immediately afterwards.
        if cache_videos:
            video_save_path = cache_dir / f"{video_unique_id}.mp4"
            if not video_save_path.is_file():
                try:
                    with open(video_save_path, "wb") as f:
                        f.write(video_bytes)
                except Exception as e:
                    logger.error(f"Failed to write video file {video_unique_id}: {e}")
                    continue
            local_path_str = str(video_save_path)
            inline_bytes: bytes | None = None
        else:
            local_path_str = f"fineVideo://{video_unique_id}.mp4"
            inline_bytes = bytes(video_bytes)

        # 5. Parse metadata
        content_meta = meta.get("content_metadata", {})

        # Extract YouTube description
        description = _as_string(meta.get("youtube_description", ""))

        # Extract scene segments: [(start, end), (start, end), ...]
        raw_scenes = content_meta.get("scenes", [])
        scene_splits = []
        for s in raw_scenes:
            ts = s.get("timestamps", {})
            start = _parse_timestamp(ts.get("start_timestamp"))
            end = _parse_timestamp(ts.get("end_timestamp"))
            scene_splits.append((start, end))

        # 6. Wrap into a VideoRecord object used by the system
        record = VideoRecord(
            fake_url=fake_url_from_path(local_path_str),
            video_id=video_unique_id,
            local_path=local_path_str,
            title=_as_string(meta.get("youtube_title", "Untitled")),
            description=description,
            subtitle=_as_string(meta.get("text_to_speech", "")),
            duration=float(meta.get("duration_seconds", 0.0)),
            tags=meta.get("youtube_tags", []),
            scene_splits=scene_splits,
            source_dataset="FineVideo-Local",
            source_id=video_unique_id,
            video_bytes=inline_bytes,
        )

        yield record

        count += 1
        # Log indexing progress
        if count % 50 == 0:
            logger.info(f"Processed and indexed: {count}/{max_videos} videos")

        if max_videos is not None and count >= max_videos:
            logger.info(f"Reached the maximum limit of {max_videos}, stopping process.")
            break


def iter_records(
    sources: Iterable[HFSourceConfig],
    cache_dir: Path,
    max_videos: int | None = None,
    *,
    cache_videos: bool = False,
    skip_video_ids: set[str] | None = None,
) -> Iterator[VideoRecord]:
    """Streams ``VideoRecord`` from all sources in order, honouring ``max_videos``.

    ``max_videos`` caps the *total* across sources; callers that need per-source
    caps should slice before passing.

    ``cache_videos`` is forwarded to :func:`iter_records_from_source` to control
    whether mp4 payloads are written to ``cache_dir`` or carried in-memory.

    ``skip_video_ids`` is forwarded so resumed builds skip already-committed
    videos at ingest time, before any heavy per-video work.
    """
    remaining = max_videos
    for source in sources:
        per_source_cap = remaining if remaining is not None else None
        for rec in iter_records_from_source(
            source,
            cache_dir=cache_dir,
            max_videos=per_source_cap,
            cache_videos=cache_videos,
            skip_video_ids=skip_video_ids,
        ):
            yield rec
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    return
