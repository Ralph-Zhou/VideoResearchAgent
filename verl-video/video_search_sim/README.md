# Local video simulator

The simulator provides FineVideo corpus ingestion, keyframe extraction, CLIP/FAISS
and BM25 retrieval with reciprocal rank fusion, a FastAPI service, four verl tool
adapters, and retrieval-domain randomization.

Run all commands below from the **top-level project root**:

```bash
pip install -e ./verl-video/video_search_sim
vss-build-corpus --config verl-video/video_search_sim/configs/corpus.yaml
vss-serve --config verl-video/video_search_sim/configs/service.yaml
```

Configure `corpus.yaml` for your media source and GPU before building. For downloaded
FineVideo Parquet files, set `sources[0].local_parquet_dir`. Set the CLIP device to
`cpu` and dtype to `float32` for CPU execution. Corpus artifacts are created under
`data/corpus`; media are external runtime inputs and are not distributed here.

The service defaults to `127.0.0.1:8000`. On a distributed cluster, configure a host
reachable by all workers. Run `scripts/prepare_tool_config.py` to generate absolute
corpus/cache paths for the training tools. Every worker must be able to resolve the
same corpus and project paths, including visual grounding's local video access.

RDR is enabled per training sample by `task_generation/prepare_rl_data.py`; it is
not enabled by merely starting the service. Evaluation rows use the live backend.
