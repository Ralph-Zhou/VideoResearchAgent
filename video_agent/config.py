"""Configuration system with Pydantic models and YAML loading."""

import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LLMNodeOverride(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enable_thinking: Optional[bool] = None
    repetition_penalty: Optional[float] = None


class LLMConfig(BaseModel):
    model: str = "your-policy-model"
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    node_overrides: Dict[str, LLMNodeOverride] = {}
    num_instances: int = 1
    start_port: int = 8000
    # Qwen-compatible chat templates accept this through
    # extra_body.chat_template_kwargs.  None preserves the endpoint default.
    enable_thinking: Optional[bool] = None
    repetition_penalty: Optional[float] = None

    def get_node_config(self, node_name: str) -> "LLMConfig":
        override = self.node_overrides.get(node_name, LLMNodeOverride())
        return LLMConfig(
            model=override.model or self.model,
            temperature=(
                override.temperature
                if override.temperature is not None
                else self.temperature
            ),
            max_tokens=override.max_tokens or self.max_tokens,
            api_key=override.api_key or self.api_key,
            base_url=override.base_url or self.base_url,
            enable_thinking=(
                override.enable_thinking
                if override.enable_thinking is not None
                else self.enable_thinking
            ),
            repetition_penalty=(
                override.repetition_penalty
                if override.repetition_penalty is not None
                else self.repetition_penalty
            ),
        )


class AgentConfig(BaseModel):
    max_iterations: int = 50
    enabled_tools: Optional[List[str]] = None
    scaffold_mode: str = "standard"  # "hybrid" (Kimi) or "standard" (OpenAI-compatible / vLLM)
    # When True, export every unpruned image-bearing trajectory under
    # `full_trajectory_dir`: a per-case OpenAI-style archive for inspection and
    # an ms-swift compatible combined all.jsonl for downstream training. Base64
    # images are decoded to disk and referenced by relative path.
    full_trajectory: bool = False
    full_trajectory_dir: Optional[str] = None
    # System prompt selection. Accepts either:
    #   - a bare name like "default_system_prompt" -> resolved to
    #     video_agent/prompts/<name>.md
    #   - an absolute/relative path to a custom .md file
    system_prompt: str = "default_system_prompt"




class SearchConfig(BaseModel):
    video_provider: str = "youtube"
    text_provider: str = "serper"
    max_results: int = 10
    youtube_cookies: Optional[str] = Field(default_factory=lambda: os.getenv("YOUTUBE_COOKIES_FILE") or None)




class WatcherConfig(BaseModel):
    num_sparse_frames: int = 16
    transcript_max_chars: int = 25000
    dense_fps: float = 1.0
    max_dense_frames_per_window: int = 32
    video_downloader: str = "ytdlp"
    max_video_resolution: str = "480"
    download_timeout: int = 120
    transcript_provider: str = "ytdlp"
    image_detail: str = "low"


class GroundingConfig(BaseModel):
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.


class CacheConfig(BaseModel):
    enabled: bool = True
    videos_dir: str = "data/cache/videos"
    transcripts_dir: str = "data/cache/transcripts"
    frames_dir: str = "data/cache/frames"


class EvalConfig(BaseModel):
    benchmark_file: str = "data/benchmark/video_browsecomp.jsonl"
    output_dir: str = "data/results/eval_reports"
    trajectory_dir: str = "data/results/trajectories"
    max_workers: int = 4
    resume: bool = True


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge used to evaluate answer correctness.

    The judge is a fully independent OpenAI-compatible client so it can use a
    different provider/endpoint than the agent's own LLM (e.g. run the agent
    on a local vLLM deployment while the judge calls a hosted API).
    """

    model: str = "your-judge-model"
    temperature: float = 1.0
    max_tokens: int = 8192
    enable_quick_judge: bool = False
    max_attempts: int = Field(default=5, ge=1)
    retry_backoff_sec: float = Field(default=1.0, ge=0)
    api_key: str = Field(
        default_factory=lambda: os.getenv("JUDGE_API_KEY")
        or os.getenv("OPENAI_API_KEY", "")
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("JUDGE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )


class LoggingConfig(BaseModel):
    level: str = "INFO"
    save_trajectory: bool = True
    trajectory_dir: str = "data/results/trajectories"
    print_steps: bool = True


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    agent: AgentConfig = AgentConfig()
    search: SearchConfig = SearchConfig()
    watcher: WatcherConfig = WatcherConfig()
    grounding: GroundingConfig = GroundingConfig()
    cache: CacheConfig = CacheConfig()
    eval: EvalConfig = EvalConfig()
    judger: JudgeConfig = JudgeConfig()
    logging: LoggingConfig = LoggingConfig()


_config: Optional[AppConfig] = None


def load_config(config_path: str = "config/default.yaml") -> AppConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return AppConfig(**data)


def get_config(config_path: str = "config/default.yaml") -> AppConfig:
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config
