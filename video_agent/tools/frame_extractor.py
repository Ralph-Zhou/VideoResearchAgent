"""Frame extraction from local video files using decord (preferred) or OpenCV."""

import base64
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

import cv2

logger = logging.getLogger(__name__)


# Short-side cap for encoded frames. Frames whose short side already fits within
# this value are untouched; larger frames (e.g. 1080p source videos) are
# downscaled so that the short side equals this cap, preserving aspect ratio.
# 720 is chosen to match typical YouTube-downloaded videos (480p-720p) and to
# keep Qwen2.5-VL visual-token usage manageable (~950 tokens / frame at 720×1280
# vs ~2100 at 1080×1920). Set FRAME_MAX_SHORT_SIDE=0 to disable.
_DEFAULT_FRAME_MAX_SHORT_SIDE = 720


def _maybe_downscale(frame_bgr: np.ndarray, max_short_side: int) -> np.ndarray:
    """Downscale if the short side exceeds `max_short_side`, keeping aspect.

    The resized dimensions are rounded down to multiples of 28 so they align
    with Qwen2.5-VL's 14-px patch + 2×2 merger (28-px visual-token cell) and
    avoid awkward padding inside vLLM.
    """
    if max_short_side <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    short = min(h, w)
    if short <= max_short_side:
        return frame_bgr
    scale = max_short_side / short
    new_w = max(28, int(round(w * scale / 28) * 28))
    new_h = max(28, int(round(h * scale / 28) * 28))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


@dataclass
class FrameData:
    timestamp: float
    image_b64: str


class FrameExtractor:
    """Extract frames from local video files (sparse uniform + dense window)."""

    def __init__(self, jpeg_quality: int = 85,
                 max_short_side: Optional[int] = None):
        self.jpeg_quality = jpeg_quality
        # Precedence: explicit arg > env var > default (720).
        if max_short_side is None:
            env = os.getenv("FRAME_MAX_SHORT_SIDE")
            if env is not None:
                try:
                    max_short_side = int(env)
                except ValueError:
                    max_short_side = _DEFAULT_FRAME_MAX_SHORT_SIDE
            else:
                max_short_side = _DEFAULT_FRAME_MAX_SHORT_SIDE
        self.max_short_side = max_short_side
        if DECORD_AVAILABLE:
            logger.info("Using decord backend for frame extraction")
        else:
            logger.info("decord not available, falling back to OpenCV")
        if self.max_short_side > 0:
            logger.info(
                "Frame downscale enabled: short side capped at %d px", self.max_short_side
            )

    def extract_sparse(self, video_path: str, n_frames: int = 16) -> List[FrameData]:
        """Uniformly sample n_frames from the entire video."""
        if DECORD_AVAILABLE:
            return self._extract_sparse_decord(video_path, n_frames)
        return self._extract_sparse_cv2(video_path, n_frames)

    def extract_dense(self, video_path: str, start: float, end: float,
                      fps: float = 1.0, max_frames: int = 32) -> List[FrameData]:
        """Extract frames from [start, end] window at the given fps."""
        if DECORD_AVAILABLE:
            return self._extract_dense_decord(video_path, start, end, fps, max_frames)
        return self._extract_dense_cv2(video_path, start, end, fps, max_frames)

    # ── decord implementations ──

    def _extract_sparse_decord(self, video_path: str, n_frames: int) -> List[FrameData]:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            logger.error("decord failed to open %s: %s, falling back to cv2", video_path, e)
            return self._extract_sparse_cv2(video_path, n_frames)

        total = len(vr)
        if total == 0:
            logger.warning("decord reports 0 frames for %s, falling back to cv2", video_path)
            return self._extract_sparse_cv2(video_path, n_frames)
        video_fps = vr.get_avg_fps()
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int).tolist()
        try:
            frames_np = vr.get_batch(indices).asnumpy()
        except Exception as e:
            logger.warning("decord get_batch failed for %s: %s, falling back to cv2", video_path, e)
            return self._extract_sparse_cv2(video_path, n_frames)

        if frames_np.size == 0 or len(frames_np) == 0:
            logger.warning("decord returned empty frames for %s, falling back to cv2", video_path)
            return self._extract_sparse_cv2(video_path, n_frames)

        results = []
        for i, frame_rgb in enumerate(frames_np):
            ts = indices[i] / video_fps if video_fps > 0 else 0
            frame_bgr = frame_rgb[:, :, ::-1]
            results.append(FrameData(
                timestamp=round(ts, 2),
                image_b64=self._encode_frame(frame_bgr),
            ))
        return results

    def _extract_dense_decord(self, video_path: str, start: float, end: float,
                              fps: float, max_frames: int) -> List[FrameData]:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            logger.error("decord failed: %s, falling back to cv2", e)
            return self._extract_dense_cv2(video_path, start, end, fps, max_frames)

        video_fps = vr.get_avg_fps()
        total = len(vr)
        if total == 0:
            return self._extract_dense_cv2(video_path, start, end, fps, max_frames)

        timestamps = []
        t = start
        while t <= end and len(timestamps) < max_frames:
            timestamps.append(t)
            t += 1.0 / fps

        indices = [min(int(ts * video_fps), total - 1) for ts in timestamps]
        if not indices:
            return []

        try:
            frames_np = vr.get_batch(indices).asnumpy()
        except Exception as e:
            logger.warning("decord dense get_batch failed: %s, falling back to cv2", e)
            return self._extract_dense_cv2(video_path, start, end, fps, max_frames)

        if frames_np.size == 0:
            return self._extract_dense_cv2(video_path, start, end, fps, max_frames)

        results = []
        for i, frame_rgb in enumerate(frames_np):
            frame_bgr = frame_rgb[:, :, ::-1]
            results.append(FrameData(
                timestamp=round(timestamps[i], 2),
                image_b64=self._encode_frame(frame_bgr),
            ))
        return results

    # ── OpenCV fallback implementations ──

    def _extract_sparse_cv2(self, video_path: str, n_frames: int) -> List[FrameData]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0 or video_fps <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, min(n_frames, total_frames), dtype=int)
        results = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            results.append(FrameData(
                timestamp=round(idx / video_fps, 2),
                image_b64=self._encode_frame(frame),
            ))
        cap.release()
        return results

    def _extract_dense_cv2(self, video_path: str, start: float, end: float,
                           fps: float, max_frames: int) -> List[FrameData]:
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            cap.release()
            return []

        timestamps = []
        t = start
        while t <= end and len(timestamps) < max_frames:
            timestamps.append(t)
            t += 1.0 / fps

        results = []
        for ts in timestamps:
            frame_idx = int(ts * video_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            results.append(FrameData(
                timestamp=round(ts, 2),
                image_b64=self._encode_frame(frame),
            ))
        cap.release()
        return results

    def _encode_frame(self, frame_bgr: np.ndarray) -> str:
        frame_bgr = _maybe_downscale(frame_bgr, self.max_short_side)
        _, buf = cv2.imencode(".jpg", frame_bgr,
                              [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return base64.b64encode(buf).decode("utf-8")
