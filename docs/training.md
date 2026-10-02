# RL training and trajectory export

Run commands from the repository root. Use a separate environment for distributed
RL when its Torch/CUDA requirements differ from inference or corpus services.

## Model and data

The [SFT checkpoint](https://huggingface.co/vra-review/VideoResearchAgent-4B-SFT)
provides the 4B initialization. Download the
[dataset](https://huggingface.co/datasets/vra-review/VideoResearchAgent-Data)
to `data/released` with the Hugging Face Hub Python client:

```bash
pip install huggingface_hub
python -c 'from huggingface_hub import snapshot_download; snapshot_download("vra-review/VideoResearchAgent-4B-SFT", local_dir="data/models/VideoResearchAgent-4B-SFT")'
python -c 'from huggingface_hub import snapshot_download; snapshot_download("vra-review/VideoResearchAgent-Data", repo_type="dataset", local_dir="data/released")'
```

The RL inputs are `data/released/rl_tasks/train.parquet` and
`data/released/rl_tasks/val.parquet`. Corpus media and the matching identifier
mapping are separate runtime inputs; follow the [simulator guide](../verl-video/video_search_sim/README.md).

To prepare additional tasks from local synthesis, use:

```bash
python task_generation/prepare_rl_data.py \
  --train_jsonl data/video_task_output_local/stage4_tasks.jsonl \
  --val_browsecomp data/benchmark/video_browsecomp.jsonl \
  --out_dir data/rl
```

This writes `train.parquet` and `val.parquet`. RDR is enabled only for training
rows. Add `--disable_domain_randomization` and use another output directory for
the fixed-retrieval ablation. Validation uses live search.

## Simulator and runtime

```bash
pip install -e ./verl-video
pip install -e ./verl-video/video_search_sim
python scripts/prepare_tool_config.py --corpus-dir data/corpus \
  --service-url http://127.0.0.1:8000 --output data/runtime/tool_config.yaml
```

Start the corpus service before rollouts. On a multi-node cluster, use a
worker-accessible service URL and identical project/corpus paths on each worker.
`--execution-backend remote_server` optionally runs corpus watching on the service;
visual grounding still needs local media access.

The GRPO recipe requires a Qwen3.5-compatible vLLM, Megatron, mbridge, and attention
kernel stack. Its default context parallelism is CP=4; the original custom
Megatron build is not bundled. Dependency ranges are not a validated GPU lockfile.

## GRPO

Start the Ray cluster and export `JUDGER_MODEL`, `JUDGER_API_KEY`,
`JUDGER_BASE_URL`, and `SERPER_API_KEY` in your shell. Then run:

```bash
export MODEL_PATH="$(pwd)/data/models/VideoResearchAgent-4B-SFT"
export TRAIN_FILE="$(pwd)/data/released/rl_tasks/train.parquet"
export TEST_FILE="$(pwd)/data/released/rl_tasks/val.parquet"
export TOOL_CONFIG="$(pwd)/data/runtime/tool_config.yaml"
bash verl-video/examples/video/run_grpo.sh
```

`RAY_ADDRESS` defaults to `http://127.0.0.1:8265`. The submitter passes credentials
through a temporary runtime file. The recipe uses group size 8, training batch 64,
LR 2e-6, KL coefficient 0.01, and TP/PP/CP = 2/1/4. Terminal reward is binary
semantic answer correctness plus 0.1 answer-format credit.

## Trajectory export

Configure a teacher endpoint, then collect complete image-bearing trajectories:

```bash
python scripts/run_benchmark.py --config config/default.yaml \
  --run-name teacher --enable-full-trajectory
```

The exporter writes full message history, relative image files, and `all.jsonl`.
Compact evaluation logs do not retain the image bytes. To rebuild trajectories
from released Parquet records, see the extraction script documented in the
[dataset repository](https://huggingface.co/datasets/vra-review/VideoResearchAgent-Data).
SFT framework settings are described in the paper.
