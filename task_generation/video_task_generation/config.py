"""TOML-driven pipeline config (pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

try:
    import tomllib  # Python 3.11+
    _TOML_BINARY = True
except ImportError:  # pragma: no cover
    import toml as tomllib  # type: ignore
    _TOML_BINARY = False


# ──────────────────────────────────────────────────────────────────────
# Section schemas
# ──────────────────────────────────────────────────────────────────────


class LLMConfig(BaseModel):
    model_text: str = "your-synthesis-model"
    model_vision: str = "your-synthesis-model"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120
    max_retries: int = 3


class Stage1Config(BaseModel):
    num_seeds: int = 80
    batch_size: int = 25
    workers: int = 8


class Stage2Config(BaseModel):
    # ``graph_depth`` doubles as an **upper bound** on the random depth (i.e. the
    # number of real hops retained in ``graph.entities`` after the initial seed
    # entity is dropped).  The actual depth per seed is drawn from
    # ``depth_distribution``:
    #   depth_distribution[i] == P(final_graph_node_count == i+1)
    # Default [0.2, 0.3, 0.3, 0.2] → final node count 1/2/3/4 with those probs
    # (the initial LLM-fabricated seed entity is used only as a search inducement
    # and does NOT count as a graph node).
    # Set ``depth_distribution = []`` to disable randomness and always use ``graph_depth``.
    graph_depth: int = 4
    depth_distribution: List[float] = Field(default_factory=lambda: [0.2, 0.3, 0.3, 0.2])
    frames_per_video: int = 24
    video_duration_min: int = 30
    video_duration_max: int = 1500
    ytsearch_topk: int = 8
    max_video_resolution: str = "360"
    frame_caption_workers: int = 4
    graph_workers: int = 2
    caption_max_chars: int = 400
    # Per-hop text-search entity enrichment (properties + relations) appended
    # at the tail of Stage 2 so downstream task generation can produce more
    # uniquely-converging questions.
    enrich_enabled: bool = True
    enrich_search_topk: int = 3
    enrich_visit_topk: int = 2
    enrich_workers: int = 4


class Stage3Config(BaseModel):
    workers: int = 4
    max_agent_turns: int = 10
    oversampling_factor: int = 2
    max_qa_retries: int = 5
    self_check: bool = True
    require_frame_evidence: bool = True


class Stage4Config(BaseModel):
    """Search-agent-driven difficulty enhancement on Stage 3 accepted tasks."""
    enabled: bool = True
    workers: int = 4
    max_rounds: int = 2
    max_agent_turns: int = 8


class SearchConfig(BaseModel):
    provider: str = "serper"
    max_results: int = 5
    enable_visit: bool = True
    visit_timeout: int = 30


class YTDLPConfig(BaseModel):
    cache_dir: str = "data/cache/videos_taskgen"


class WorkflowConfig(BaseModel):
    output_dir: str = "data/video_task_output"
    target_tasks: int = 50
    workers: int = 4
    log_level: str = "INFO"


class RuntimeConfig(BaseModel):
    """Top-level pipeline mode switch.

    ``mode = "online"``  reproduces the original behaviour: yt-dlp + YouTube
    + Serper for both video sourcing and text-search verification.

    ``mode = "local"``   builds task graphs over a pre-built local video
    corpus served by ``video_search_sim`` (BM25+CLIP retrieval, fake YouTube
    URLs, local mp4 paths via ``url_to_path.json``). Synthesised tasks reuse
    the exact same JSONL schema; the only difference is that
    ``graph.entities[*].video.url`` is a fake URL produced by the corpus
    builder, which lines up 1-to-1 with the verl reward function's
    ``gold_urls`` set.
    """

    mode: str = "online"  # "online" | "local"


class LocalCorpusConfig(BaseModel):
    """Settings consumed when ``runtime.mode == 'local'``.

    The corpus is the artifact directory produced by ``vss-build-corpus``;
    it must contain ``videos.parquet``, ``url_to_path.json``, ``manifest.json``,
    ``bm25/``, and ``clip/``. The retrieval service is queried over HTTP for
    cross-machine compatibility.

    ``text_search_mode`` controls how Stage 3 / Stage 4 verify task
    "video-dependence" when no Serper-equivalent web search is available:

    - ``"disabled"`` : skip the text search agent entirely, only run the
      LLM-internal-knowledge self-check + frame_evidence enforcement. This
      is logically self-consistent because the local-simulation RL training
      environment also has no real web search tool, so any task the
      auditor "would have solved with Serper" is still genuinely
      video-dependent at training time.
    - ``"keep_serper"`` : keep using Serper for verification even though
      the agent at training time will not use it. Requires SERPER_API_KEY.
    """

    corpus_dir: str = ""
    service_url: str = "http://127.0.0.1:8000/video_search"
    service_timeout: int = 30
    shard_root: str = ""               # optional override; pulled from manifest.json otherwise
    shard_materialise_dir: str = "/tmp/vss_shard_cache"
    top_k: int = 10
    text_search_mode: str = "disabled"   # "disabled" | "keep_serper"


# ──────────────────────────────────────────────────────────────────────
# Top-level container
# ──────────────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Top-level container loaded from config.toml."""

    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stage1_seed_generation: Stage1Config = Field(default_factory=Stage1Config)
    stage2_graph_construction: Stage2Config = Field(default_factory=Stage2Config)
    stage3_task_generation: Stage3Config = Field(default_factory=Stage3Config)
    stage4_difficulty_enhancement: Stage4Config = Field(default_factory=Stage4Config)
    search: SearchConfig = Field(default_factory=SearchConfig)
    yt_dlp: YTDLPConfig = Field(default_factory=YTDLPConfig)
    local_corpus: LocalCorpusConfig = Field(default_factory=LocalCorpusConfig)

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "PipelineConfig":
        path = Path(path)
        if _TOML_BINARY:
            with open(path, "rb") as fh:
                data: Dict[str, Any] = tomllib.load(fh)
        else:
            data = tomllib.load(str(path))  # type: ignore[attr-defined]
        return cls(**data)
