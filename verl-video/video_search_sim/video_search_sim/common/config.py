"""Config models + loader.

Two top-level configs live here:

- ``CorpusConfig`` — consumed by the offline ``video_corpus`` pipeline.
- ``ServiceConfig`` — consumed by the ``retrieval_service`` FastAPI app.

Both are plain pydantic models so they can be loaded from YAML with
``load_yaml_config(Path, Model)`` and validated up-front.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class HFSourceConfig(BaseModel):
    """One Hugging Face dataset to pull into the corpus."""

    name: str = Field(..., description="HF repo id, e.g. 'HuggingFaceFV/FineVideo'")
    split: str = "train"
    subset: str | None = Field(None, description="Some datasets have sub-configs (e.g. 'lvbench-full'); None = default")
    text_fields: list[str] = Field(
        default_factory=lambda: ["title", "description"],
        description="Which dataset columns to concatenate into searchable text",
    )
    video_field: str = Field(
        "video_path",
        description="Dataset column holding the video path; may be an HF-hosted URL or local path",
    )
    duration_field: str | None = "duration"
    tags_field: str | None = "tags"
    scene_field: str | None = "scene_splits"
    subtitle_field: str | None = Field(
        None,
        description="Dataset column holding ASR / caption text; None = no subtitle for this source",
    )
    source_id_field: str | None = None
    local_parquet_dir: str | None = Field(
        None,
        description=(
            "Optional path to a directory containing pre-downloaded Parquet shards "
            "(*.parquet) for this source. When set, ingestion loads from disk "
            "instead of streaming from Hugging Face. Supports '~' expansion. "
            "If None, falls back to the VSS_LOCAL_PARQUET_DIR env var, then to "
            "'./data/corpus/local_parquet/data' relative to the working directory."
        ),
    )


class KeyframeConfig(BaseModel):
    """How keyframes are sampled for CLIP indexing."""

    strategy: Literal["uniform_fps", "scene", "first_only"] = "uniform_fps"
    fps: float = Field(1.0, gt=0, description="When strategy == uniform_fps")
    max_frames_per_video: int = Field(32, ge=1, le=512)


class CLIPConfig(BaseModel):
    """CLIP encoder configuration."""

    model_name: str = "openai/clip-vit-large-patch14-336"
    device: str = Field("cuda:0", description="'cuda:N' | 'cpu'")
    batch_size: int = 64
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    embedding_dim: int = Field(768, description="Set to match the model head; ViT-L/14 = 768")
    normalize: bool = True


class BM25Config(BaseModel):
    """BM25 indexer configuration."""

    backend: Literal["bm25s", "rank_bm25"] = "bm25s"
    k1: float = 1.5
    b: float = 0.75
    lowercase: bool = True


class CorpusConfig(BaseModel):
    """Top-level config for building an offline corpus."""

    output_dir: Path
    sources: list[HFSourceConfig] = Field(..., min_length=1)
    keyframe: KeyframeConfig = KeyframeConfig()
    clip: CLIPConfig = CLIPConfig()
    bm25: BM25Config = BM25Config()
    max_videos: int | None = Field(None, description="Cap total videos across sources; None = unlimited")
    num_download_workers: int = 8
    seed: int = 42
    cache_videos: bool = Field(
        False,
        description=(
            "If True, materialise each ingested mp4 so that watch_video can "
            "re-extract frames later. The actual storage layout is controlled "
            "by ``cache_layout``. If False (default), mp4 bytes are streamed "
            "in-memory through keyframe extraction and never touch disk."
        ),
    )
    cache_layout: Literal["sharded_tar", "loose_files"] = Field(
        "sharded_tar",
        description=(
            "How cached mp4s are laid out on disk. 'sharded_tar' (default and "
            "strongly recommended for >10K videos) packs every "
            "``videos_per_shard`` mp4s into one uncompressed tar; "
            "'loose_files' writes one mp4 per file under "
            "<output_dir>/downloads_cache/ (legacy behaviour, fine for tiny "
            "smoke tests but stresses lustre/cephfs metadata at scale)."
        ),
    )
    cache_dir: Path | None = Field(
        None,
        description=(
            "Directory holding the mp4 cache. If None, falls back to "
            "<output_dir>/downloads_cache/. Recommended setting: a "
            "large-capacity shared scratch path, distinct from output_dir, "
            "so the index artefacts and the bulky mp4 storage can be sized "
            "and backed up independently."
        ),
    )
    videos_per_shard: int = Field(
        1024,
        ge=1,
        description=(
            "Number of mp4s packed into one tar shard when "
            "``cache_layout == 'sharded_tar'``. With ~30 MB videos this "
            "gives ~30 GB shards, friendly to lustre and to the page cache."
        ),
    )
    max_shard_bytes: int | None = Field(
        None,
        description=(
            "Optional soft cap (bytes) on the size of a single tar shard. "
            "When set, a shard is rolled over before adding an entry that "
            "would push it past this size."
        ),
    )
    checkpoint_every: int = Field(
        200,
        ge=1,
        description=(
            "Commit a resumable batch checkpoint to <output_dir>/_partial/ "
            "every N successfully processed videos. On crash, the next run "
            "skips everything in committed batches and continues from the "
            "first uncommitted video. Set very large (e.g. 10**9) to "
            "effectively disable mid-run checkpointing."
        ),
    )
    keep_partial: bool = Field(
        False,
        description=(
            "If True, keep <output_dir>/_partial/ around after a successful "
            "finalize (useful for debugging the resume path). Default False "
            "deletes the per-batch artefacts once the canonical index has "
            "been written."
        ),
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def _expand_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


class RetrievalConfig(BaseModel):
    """Runtime retrieval knobs for ``/video_search``."""

    max_topk: int = 50
    default_topk: int = 5
    # Reciprocal Rank Fusion k
    rrf_k: int = 60
    # Per-source candidate pool before fusion
    bm25_candidate_pool: int = 100
    dense_candidate_pool: int = 100
    # Aggregate keyframe scores to video-level: 'max' (best for event queries) or 'mean'
    dense_aggregation: Literal["max", "mean"] = "max"
    # Set False to bypass BM25 or dense entirely (useful for ablation tests)
    enable_bm25: bool = True
    enable_dense: bool = True


class ServerConfig(BaseModel):
    """Uvicorn server knobs."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    # On a GPU host keep workers=1 (one CLIP/FAISS singleton per process — more
    # would duplicate the index in GPU memory and OOM). On a CPU-only host with
    # spare RAM you MAY raise it to bypass the single-process GIL ceiling; each
    # worker reloads the full index (several GB), so size by RAM. Note: enabling
    # workers>1 requires the import-string launch path in server.py.
    workers: int = 1
    # Size of the anyio threadpool that runs the blocking bm25/dense searches
    # (see app.py). The default anyio pool is only 40 threads, which caps
    # request concurrency far below a many-core CPU host's capacity. Raise this
    # toward the physical core count to let concurrent searches fan out across
    # cores. Pairs with per-request single-threading (torch/faiss pinned to 1
    # thread at startup) so N concurrent requests map cleanly to N cores.
    threadpool_size: int = 64


class WatchConfig(BaseModel):
    """Runtime knobs for the optional ``/watch_video`` endpoint."""

    enable: bool = True
    max_n_frames_sparse: int = 16
    max_n_frames_dense: int = 32
    max_transcript_chars: int = 4000
    jpeg_quality: int = 85
    shard_materialise_dir: Path = Path("/tmp/vss_watch_service_shard_cache")
    shard_root: Path | None = None
    downloader_cache_dir: Path = Path("data/cache/videos")
    transcript_cache_dir: Path = Path("data/cache/transcripts")
    remote_download_timeout_sec: int = 120
    concurrency: int = 64

    @field_validator("shard_materialise_dir", "shard_root", "downloader_cache_dir", "transcript_cache_dir", mode="before")
    @classmethod
    def _expand_optional_path(cls, v: str | Path | None) -> Path | None:
        if v in (None, ""):
            return None
        return Path(v).expanduser().resolve()


class ServiceConfig(BaseModel):
    """Top-level config for the FastAPI service."""

    corpus_dir: Path
    clip: CLIPConfig = CLIPConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    server: ServerConfig = ServerConfig()
    watch: WatchConfig = WatchConfig()

    @field_validator("corpus_dir", mode="before")
    @classmethod
    def _expand_corpus_dir(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()


def load_yaml_config(path: str | Path, model: type[BaseModel]) -> BaseModel:
    """Load ``path`` as YAML and validate against ``model``.

    Raises ``pydantic.ValidationError`` on schema mismatch (with file location
    baked into the message for easier debugging).
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return model.model_validate(raw)
    except Exception as e:
        raise ValueError(f"Failed to validate {p} against {model.__name__}: {e}") from e
