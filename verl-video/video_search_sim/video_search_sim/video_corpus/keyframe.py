"""Keyframe extraction.

Three strategies are supported; the active one is chosen via
``KeyframeConfig.strategy``:

- ``uniform_fps``: sample one frame every ``1/fps`` seconds, capped at
  ``max_frames_per_video``.
- ``scene``: use scene boundaries if the ``VideoRecord`` carries them; fall
  back to ``uniform_fps`` if the list is empty.
- ``first_only``: sample the first frame only (useful for schema smoke tests
  on tiny corpora where CLIP encoding budget is a concern).

The extractor returns numpy RGB frames — shape ``(H, W, 3)`` uint8 — and the
corresponding timestamps in seconds. All file I/O goes through ``decord`` for
consistent, thread-safe decoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..common.config import KeyframeConfig
from ..common.schemas import VideoRecord

logger = logging.getLogger(__name__)


@dataclass
class KeyframeBundle:
    """Keyframes extracted from a single video.

    Attributes:
        video_id: The 11-char fake YouTube ID.
        frames: ``(N, H, W, 3)`` uint8 numpy array. Empty if extraction failed.
        timestamps: Wall-clock timestamps (seconds) for each frame, length N.
    """

    video_id: str
    frames: np.ndarray
    timestamps: np.ndarray  # (N,) float32


def _select_indices_uniform(total_frames: int, fps_video: float, target_fps: float, cap: int) -> np.ndarray:
    """Pick frame indices evenly spaced at ``target_fps``, then cap at ``cap`` frames."""
    if total_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if target_fps <= 0 or fps_video <= 0:
        # Degenerate metadata: fall back to a single center frame.
        return np.array([total_frames // 2], dtype=np.int64)
    step = max(1, int(round(fps_video / target_fps)))
    idx = np.arange(0, total_frames, step, dtype=np.int64)
    if idx.size > cap:
        # Downsample uniformly to the cap.
        idx = idx[np.linspace(0, idx.size - 1, cap).round().astype(np.int64)]
    return idx


def _select_indices_scene(record: VideoRecord, fps_video: float, cap: int) -> np.ndarray | None:
    """Scene-aware selection: one frame at the midpoint of each scene.

    Returns ``None`` when the record has no scene splits so the caller can
    gracefully fall back to ``uniform_fps``.
    """
    if not record.scene_splits or fps_video <= 0:
        return None
    midpoints_sec = np.array(
        [0.5 * (s + e) for s, e in record.scene_splits if e > s],
        dtype=np.float64,
    )
    if midpoints_sec.size == 0:
        return None
    if midpoints_sec.size > cap:
        # Uniformly thin out, keeping endpoints.
        keep = np.linspace(0, midpoints_sec.size - 1, cap).round().astype(np.int64)
        midpoints_sec = midpoints_sec[keep]
    return (midpoints_sec * fps_video).astype(np.int64)


def extract_keyframes(record: VideoRecord, cfg: KeyframeConfig) -> KeyframeBundle:
    """Extract keyframes from one video.

    Returns a ``KeyframeBundle`` with ``frames.shape[0] == 0`` on failure;
    callers must handle empty bundles (typically: skip the video).

    Source resolution order:
      1. ``record.video_bytes`` (in-memory mp4 buffer) — preferred when set,
         lets us skip writing the file to disk entirely.
      2. ``record.local_path`` — fall back to the on-disk file.
    """
    try:
        import decord
    except ImportError as e:  # pragma: no cover - declared in pyproject
        raise RuntimeError("`decord` not installed; `pip install decord`") from e

    # decord releases the GIL around reads but is not fully thread-safe on its
    # own VideoReader handle; we keep one per call to stay on the safe side.
    video_bytes = getattr(record, "video_bytes", None)
    try:
        if video_bytes is not None:
            import io
            vr = decord.VideoReader(io.BytesIO(video_bytes))
        else:
            vr = decord.VideoReader(record.local_path)
    except Exception as e:
        logger.warning(
            "decord failed to open %s: %s",
            record.video_id if video_bytes is not None else record.local_path,
            e,
        )
        return KeyframeBundle(record.video_id, np.zeros((0, 1, 1, 3), dtype=np.uint8), np.zeros(0, dtype=np.float32))

    total = len(vr)
    fps = float(vr.get_avg_fps() or 0.0)

    if cfg.strategy == "first_only":
        idx = np.array([0], dtype=np.int64)
    elif cfg.strategy == "scene":
        idx = _select_indices_scene(record, fps_video=fps, cap=cfg.max_frames_per_video)
        if idx is None:
            idx = _select_indices_uniform(total, fps_video=fps, target_fps=cfg.fps, cap=cfg.max_frames_per_video)
    else:
        idx = _select_indices_uniform(total, fps_video=fps, target_fps=cfg.fps, cap=cfg.max_frames_per_video)

    if idx.size == 0:
        return KeyframeBundle(record.video_id, np.zeros((0, 1, 1, 3), dtype=np.uint8), np.zeros(0, dtype=np.float32))

    # decord requires python ints in the list form.
    idx_list = idx.tolist()
    try:
        batch = vr.get_batch(idx_list)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("decord failed to decode batch for %s: %s", record.local_path, e)
        return KeyframeBundle(record.video_id, np.zeros((0, 1, 1, 3), dtype=np.uint8), np.zeros(0, dtype=np.float32))

    frames = batch.asnumpy() if hasattr(batch, "asnumpy") else np.asarray(batch)  # (N, H, W, 3)
    timestamps = (idx.astype(np.float64) / max(fps, 1e-6)).astype(np.float32)
    return KeyframeBundle(video_id=record.video_id, frames=frames, timestamps=timestamps)
