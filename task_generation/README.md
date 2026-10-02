# Video-grounded task synthesis

Run from the top-level repository root after installing `.[synthesis]`.
Set the text and vision model names in the chosen TOML config and supply
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `SERPER_API_KEY` through `.env`.

```bash
python task_generation/run.py --config task_generation/config.toml
```

The stages sample taxonomy-conditioned seeds, build timestamped cross-video
entity graphs, generate and verify grounded questions, and refine questions
against text-only shortcuts. Outputs/checkpoints live under the configured
`workflow.output_dir`; use `--help` for resume and stage-selection options.

For the local synthesis path, build and serve the simulator first, then set
`local_corpus.corpus_dir` and `local_corpus.service_url`:

```bash
pip install -e ./verl-video/video_search_sim
python task_generation/run.py --config task_generation/config_local.toml
```

`text_search_mode = "disabled"` retains frame provenance and closed-book checks;
`"keep_serper"` additionally uses real web search. The online path uses real
YouTube retrieval, and general webpage visits use direct HTTP requests.

For local training tasks, convert accepted synthesis records with
`prepare_rl_data.py`; see [training](../docs/training.md). Generated corpora and
records are runtime artifacts, not part of this code distribution.
