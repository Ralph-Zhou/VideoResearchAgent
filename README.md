<div align="center">

# VideoResearchAgent

[Paper](https://arxiv.org/abs/2610.04911) | [HuggingFace · Dataset](https://huggingface.co/datasets/vra-review/VideoResearchAgent-Data) | [HuggingFace · SFT checkpoint](https://huggingface.co/vra-review/VideoResearchAgent-4B-SFT) | [HuggingFace · RL checkpoint](https://huggingface.co/vra-review/VideoResearchAgent-4B-RL)

</div>

## Overview

**VideoResearchAgent** studies open-web video research: given a question, an agent
must discover relevant videos, navigate their timelines, and verify visual evidence
before answering. We develop a training framework that connects video-grounded
task synthesis, efficient interaction with a local video corpus, and policy transfer
to the live web.

The framework has three components:

- **Video-grounded task synthesis.** Construct multi-hop questions from timestamped
  visual evidence across videos, with verification and refinement to reduce text-only
  shortcuts.
- **Video research simulator.** Replace live video retrieval with a local corpus
  backend while preserving the agent's search and viewing interfaces, enabling
  efficient reinforcement learning.
- **Retrieval-domain-randomized GRPO (RDR-GRPO).** Vary candidate rankings,
  distractors, metadata, and result structure during training to improve transfer
  beyond a fixed local retriever.

The agent combines four tools—`search_youtube`, `web_search`, `watch_video`, and
`visual_grounding`—for source discovery, temporal navigation, and visual verification.

## Results

Accuracy on **Video-BrowseComp** (210 questions), as reported in the paper:

| Qwen3.5-4B policy | Accuracy |
| --- | ---: |
| Base | 30.00% |
| + SFT | 36.67% |
| + SFT + RDR-GRPO | **40.48%** |

Post-training also reduces cumulative API-token consumption by **74.9%** relative
to the base policy under the paper's evaluation protocol.

## Models and Data

All released artifacts are hosted on the Hugging Face Hub.

| Resource | Size | Description |
| --- | ---: | --- |
| [VideoResearchAgent-4B-SFT](https://huggingface.co/vra-review/VideoResearchAgent-4B-SFT) | 4B | Policy after supervised fine-tuning |
| [VideoResearchAgent-4B-RL](https://huggingface.co/vra-review/VideoResearchAgent-4B-RL) | 4B | Policy after RDR-GRPO |
| [SFT tasks](https://huggingface.co/datasets/vra-review/VideoResearchAgent-Data/tree/main/sft_tasks) | 3,534 | Generated research questions and reference answers |
| [SFT trajectories](https://huggingface.co/datasets/vra-review/VideoResearchAgent-Data/tree/main/sft_trajectories) | 2,301 | Tool-use trajectories with embedded video frames |
| [RL tasks](https://huggingface.co/datasets/vra-review/VideoResearchAgent-Data/tree/main/rl_tasks) | 3,864 | Training questions for simulator interaction |

The dataset repository also includes a 63-question Video-BrowseComp validation
subset. See its [README](https://huggingface.co/datasets/vra-review/VideoResearchAgent-Data/blob/main/README.md)
for loading the Parquet files and exporting trajectories with images.

## Quick Start

### Installation

Use Python 3.11–3.13 and install FFmpeg and a JavaScript runtime supported by yt-dlp.
Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**YouTube cookies** — Video tools (`search_youtube`, `watch_video`) require a
Netscape-format cookies file to handle age-gated, region-restricted, and
bot-checked videos. Export one from your browser with
[yt-dlp's `--cookies-from-browser` flag](https://github.com/yt-dlp/yt-dlp?tab=readme-ov-file#filesystem-options)
or a browser extension like "Get cookies.txt LOCALLY". Set the path in `.env`:

```bash
YOUTUBE_COOKIES_FILE=youtube_cookies.txt
```

or in `config/default.yaml` (`search.youtube_cookies`). RL training passes this
through `YOUTUBE_COOKIES_FILE` in `scripts/submit_training.py`.

> **Note:** Using your personal YouTube account with yt-dlp runs the risk of
> temporary or permanent bans. Consider a throwaway account for bulk downloads.

### Run the Agent

Serve either released checkpoint through an OpenAI-compatible multimodal endpoint.
Configure the following in `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | Policy endpoint |
| `SERPER_API_KEY` | Web search |
| `JUDGE_BASE_URL`, `JUDGE_API_KEY` | Answer evaluation |
| `YOUTUBE_COOKIES_FILE` | (Optional) Netscape cookies for video tools |

Set `llm.model` in `config/default.yaml` to your served model name, and
`judger.model` to your evaluation judge. Then run:

```bash
python scripts/run_example.py --config config/default.yaml \
  --query 'Your video research question'
```

### Evaluation

The benchmark files are not included in this code repository. To reproduce the
paper evaluation, download the released benchmark/validation data from the
project dataset repository and place the JSONL file at the path configured by
`eval.benchmark_file` (the default is
`data/benchmark/video_browsecomp.jsonl`). The `data/` directory is intentionally
git-ignored. `--benchmark-name` is the logical benchmark name, not a JSONL path;
use `video_browsecomp` for the default benchmark. Each record contains `row_id`,
`question`, and `answer`; `level` and `category` are optional.

```bash
python scripts/run_benchmark.py \
  --benchmark-name video_browsecomp \
  --config config/default.yaml \
  --run-name evaluation
```

The default configuration uses temperature 0 and a maximum of 50 agent turns.

## Repository layout

```text
video_agent/           Agent loop, the four tools, prompts, and trajectory export
eval/                  Benchmark runner, judge, and metrics
config/                Agent and evaluation configuration
scripts/               Inference, evaluation, and trajectory entry points
task_generation/       Video-grounded task synthesis and RL data preparation
verl-video/
├── video_search_sim/  Local video simulator, retrieval corpus, and RDR
├── verl/              Vendored GRPO training runtime (upstream license retained)
└── examples/video/    RL training recipe
```

The repository holds three separable parts. Each installs on its own, so you
only need the one you are running.

### 1. Inference and evaluation

`video_agent/` plus `eval/`. This is the deployment-facing agent: the loop in
`video_agent/agent/video_research_agent.py`, the four tools under
`video_agent/tools/`, and the system prompt in `video_agent/prompts/`. It talks
to a served model and the live web only, and never to the simulator.

```bash
pip install -e .
python scripts/run_example.py --config config/default.yaml --query 'Your question'
python scripts/run_benchmark.py --config config/default.yaml --run-name evaluation
```

### 2. Task synthesis

`task_generation/`. Turns timestamped visual evidence into multi-hop research
questions, with verification and refinement to filter text-only shortcuts. Runs
either against live YouTube discovery or against a running simulator.

```bash
pip install -e '.[synthesis]'
python task_generation/run.py --config task_generation/config.toml
```

See [task_generation/README.md](task_generation/README.md) for the stage list and
the local-corpus path.

### 3. Local simulator and RL training

`verl-video/`. Two packages with different dependency stacks:

- **`video_search_sim/`** is the simulator: FineVideo corpus ingestion, keyframe
  extraction, CLIP/FAISS and BM25 retrieval with reciprocal rank fusion, a
  FastAPI retrieval service, the four `verl` tool adapters, and
  retrieval-domain randomization (RDR).
- **`verl/`** is a vendored snapshot of the GRPO training runtime, with
  `examples/video/run_grpo.sh` as the recipe. Upstream Apache-2.0 license and
  notice are preserved; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

RDR is applied per training sample during data preparation, not by starting the
service. Evaluation rows always use the live backend.

```bash
pip install -e ./verl-video
pip install -e ./verl-video/video_search_sim
python scripts/prepare_tool_config.py --corpus-dir data/corpus \
  --service-url http://127.0.0.1:8000 --output data/runtime/tool_config.yaml
bash verl-video/examples/video/run_grpo.sh
```

The corpus service must be running before rollouts. On a multi-node cluster every
worker needs a reachable service URL and identical corpus and project paths.
See the [simulator guide](verl-video/video_search_sim/README.md) for corpus
construction and [docs/training.md](docs/training.md) for checkpoint download,
GRPO settings, and trajectory export.

## Development

Run from the repository root:

```bash
pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```

To include the simulator's offline tests:

```bash
pip install -e './verl-video/video_search_sim[dev]'
PYTHONPATH=.:verl-video/video_search_sim PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -p pytest_asyncio.plugin tests verl-video/video_search_sim/tests -q
```

## License

Original code in this repository is Apache-2.0; see [LICENSE](LICENSE).
[Third-party notices](THIRD_PARTY_NOTICES.md) cover the vendored runtime.
