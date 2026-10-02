"""Authoritative data contracts shared between the corpus, the service and the verl tools.

These models are the *only* place where the public surface of the video-search
simulation is defined. Every other module imports from here.

The shapes are intentionally kept close to what the real SFT-time
``video_search`` tool emitted (YouTube-style URLs, short snippets, float
durations in seconds) so that tokenisation of tool responses stays stable
between SFT and RL rollouts.

Scope of this module
--------------------
Only the **video_search** and optional remote **watch_video** surfaces live
here — those are the wire contracts shared between the FastAPI service and
the verl tools.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class VideoRecord(BaseModel):
    """One row of ``videos.parquet`` — the authoritative description of a corpus entry."""

    model_config = {"arbitrary_types_allowed": True}

    fake_url: str = Field(..., description="YouTube-style URL, e.g. https://www.youtube.com/watch?v=<id>")
    video_id: str = Field(..., description="11-char YouTube-style ID, stable across rebuilds")
    local_path: str = Field(
        ...,
        description=(
            "Path or virtual handle for the mp4. When the corpus is built without "
            "caching mp4s on disk (default), this is a 'fineVideo://<id>.mp4' style "
            "virtual path that does not exist on disk; consumers like watch_video "
            "must then fall back to remote download."
        ),
    )
    title: str = ""
    description: str = ""
    subtitle: str = Field("", description="ASR / caption concatenated into a single string")
    duration: float = Field(0.0, description="Video duration in seconds; 0 if unknown")
    tags: list[str] = Field(default_factory=list)
    scene_splits: list[tuple[float, float]] = Field(
        default_factory=list,
        description="List of (start_sec, end_sec) tuples for scene boundaries; may be empty",
    )
    source_dataset: str = Field("", description="Origin dataset, e.g. 'FineVideo', 'LVBench'")
    source_id: str = Field("", description="Original id inside source_dataset")

    # Transient field: holds the raw mp4 bytes between ingest and keyframe
    # extraction so we can avoid writing the file to disk. Excluded from
    # ``model_dump`` / parquet serialization. Cleared (set to None) by the
    # pipeline as soon as keyframes have been extracted.
    video_bytes: bytes | None = Field(default=None, exclude=True, repr=False)


class SearchHit(BaseModel):
    """One entry in a ``/video_search`` response."""

    fake_url: str
    title: str
    snippet: str = Field(..., description="Short human-readable summary; typically <= 200 chars")
    duration: float = 0.0
    thumbnail: str = Field(..., description="Thumbnail URL (fabricated to look like youtube hqdefault)")
    score: float = Field(..., description="Final fused retrieval score (higher = better)")

    @field_validator("snippet")
    @classmethod
    def _cap_snippet_length(cls, v: str) -> str:
        return v if len(v) <= 512 else v[:509] + "..."


class VideoSearchRequest(BaseModel):
    """Request body for ``POST /video_search``."""

    query: str = Field(..., min_length=1, max_length=2048)
    topk: int = Field(5, ge=1, le=50)
    corpus_shard: str | None = None


class VideoSearchResponse(BaseModel):
    """Response body for ``POST /video_search``."""

    query: str
    results: list[SearchHit]
    latency_ms: float = 0.0
    debug: dict[str, Any] = Field(default_factory=dict)


class WatchVideoRequest(BaseModel):
    """Request body for ``POST /watch_video``.

    This mirrors the in-process WatchVideoTool parameters so the rollout side
    can offload CPU-heavy frame extraction to the retrieval service host.
    """

    url: str = Field(..., min_length=1)
    mode: str = Field("sparse", pattern="^(sparse|dense)$")
    n_frames: int = Field(16, ge=1, le=64)
    start_time: float | None = None
    end_time: float | None = None
    fps: float = Field(1.0, gt=0.0, le=30.0)
    backend: str = Field("local", pattern="^(local|remote)$")


class WatchVideoResponse(BaseModel):
    """Response body for ``POST /watch_video``.

    ``frames_b64`` intentionally matches the existing WatchVideoTool internal
    body so the client can keep using the same PIL decoding path.
    """

    url: str
    mode: str
    duration_sec: float = 0.0
    local_hit: bool = False
    timestamps: list[float] = Field(default_factory=list)
    frames_b64: list[str] = Field(default_factory=list)
    transcript: str = ""
    transcript_truncated: bool = False
    latency_ms: float = 0.0
    error: str | None = None
