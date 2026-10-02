"""``search_youtube`` tool adapter that plugs into verl's ``tool_agent_loop``.

Runtime flow during RL rollout:

1. The agent emits a ``<tool_call><function=search_youtube>...`` call; verl's
   ``Qwen3XMLToolParser`` decodes it into ``{"query": "...", "max_results": N}``.
2. verl's ``ToolAgentLoop`` resolves the name ``search_youtube`` to an instance
   of this class (registered via ``tool_config.yaml``) and calls ``execute()``.
3. ``execute()`` posts to the local FastAPI retrieval service (HTTP), which
   returns the RRF-fused ranking.
4. The hits are rendered into a JSON string and handed back as
   ``ToolResponse(text=...)``. No images/videos are attached at this stage —
   the agent is expected to call ``watch_video`` on a selected URL next.

Backend routing — ``backend ∈ {"local", "remote"}``
---------------------------------------------------

The same tool instance is shared between train and eval rollouts inside one
verl process, but each rollout sample carries its own
``tools_kwargs.search_youtube.create_kwargs.backend`` value (set by the data
preprocessing step). ``create()`` snapshots that flag onto the per-instance
state so ``execute()`` knows which path to take:

* ``local``  — POST to the ``video_search_sim`` FastAPI service (current
  default, fake YouTube URLs).
* ``remote`` — call ``video_agent.tools.video_search.YouTubeSearchTool``
  against the real YouTube via yt-dlp. Used for real-world benchmarks
  (video_browsecomp etc.) where the corpus is irrelevant.

Both paths produce the *same* JSON envelope (``{"query","results":[{"url",
"title","snippet","duration_sec","thumbnail"},...]}``) so the downstream
``watch_video`` call site, the reward regex, and the SFT/RL token
distribution all stay aligned.

Tool naming: this tool is registered as ``search_youtube`` to align with the
SFT training data (which was collected with the video_agent framework that
uses ``search_youtube`` as the tool name). The underlying retrieval uses the
local corpus simulation instead of real YouTube.

Concurrency: we share the Ray execution pool pattern in ``_common.py`` with
the other HTTP/GPU-heavy tools; each tool holds its own global token-bucket
limiter (named after the tool) so a surge of ``watch_video`` decodes cannot
stall ``search_youtube`` HTTP calls.

.. note::

    This module imports ``verl`` at load time. That means you can only import
    ``video_search_sim.verl_tools.video_search_tool`` inside a Python environment
    that has verl installed — the RL rollout environment. Offline index building
    (``video_search_sim.video_corpus``) and the FastAPI service
    (``video_search_sim.retrieval_service``) do **not** touch verl and remain
    importable elsewhere.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any
from uuid import uuid4

import httpx
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

from ._common import (
    dumps_tool_text,
    ensure_video_agent_on_path,
    init_execution_pool,
    resolve_youtube_cookies,
)
# Pure-function post-processing layer for retrieval-domain randomization. It
# takes effect only on the training (local backend) path; see
# ``_apply_domain_randomization`` and the notes in domain_randomization.py.
from .domain_randomization import apply_domain_randomization

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VIDEO_SEARCH_SIM_LOG_LEVEL", "WARNING"))


def _format_tool_text(query: str, results: list[dict[str, Any]]) -> str:
    """Render search results into a compact JSON string.

    We emit a JSON object rather than free-form prose so:

    - the agent's subsequent ``watch_video`` tool call can lift the URL
      unambiguously from the ``results[*].url`` field,
    - reward / scoring code can regex the trajectory deterministically,
    - SFT-time and RL-time tool response tokenisation stays identical (the
      SFT collector emits the same schema).

    Keys are intentionally short and stable — renaming them would drift the
    token distribution between SFT and RL.
    """
    payload = {
        "query": query,
        "results": [
            {
                "url": r.get("fake_url", "") or r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("snippet", "") or r.get("description", ""),
                "duration_sec": float(r.get("duration", 0.0) or 0.0),
                "thumbnail": r.get("thumbnail", "") or r.get("thumbnail_url", ""),
            }
            for r in results
        ],
    }
    return dumps_tool_text(payload)


class VideoSearchTool(BaseTool):
    """HTTP-backed (or remote-YouTube) ``search_youtube`` tool.

    Expected ``config`` keys (all optional unless marked):

    ``retrieval_service_url`` (str, **required**)
        Full URL of the FastAPI ``/video_search`` endpoint (used when the
        per-sample ``backend == "local"``).
    ``topk`` (int, default 5)
        Default number of results when the agent does not specify ``topk``.
    ``timeout`` (int seconds, default 30)
    ``num_workers`` (int, default 64)
        Max concurrency of the Ray execution pool.
    ``rate_limit`` (int, default 64)
        Global ceiling on simultaneous in-flight HTTP requests.
    ``enable_global_rate_limit`` (bool, default True)
    ``type`` (str, default ``"native"``)
        Required by verl's ``tool_registry``; keep at ``"native"``.

    Per-sample ``create_kwargs`` (set by the data preprocessing step):

    ``backend`` (str, default ``"local"``)
        ``"local"`` → FastAPI retrieval service; ``"remote"`` → real YouTube
        via ``video_agent.tools.video_search.YouTubeSearchTool``.
    ``corpus_shard`` (str, default ``"default"``)
        Reserved for multi-corpus fan-out; currently echoed back in metrics.
    ``domain_randomization`` (dict, default ``{}``)
        Switch and strength settings for retrieval-domain randomization. Written
        and applied **only for training samples** (backend=="local"); when empty,
        when ``enable!=True``, or when backend=="remote", randomization is fully
        off. A light config such as ``{"enable": True, "candidate_pool": 50}``
        suffices; the tier table lives in ``domain_randomization.py``.
    ``gold_urls`` (list[str], default ``[]``)
        URL of the gold answer video for this question, used for targeted gold
        sinking (passed through for training samples only).
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        self.retrieval_service_url = config.get("retrieval_service_url")
        if not self.retrieval_service_url:
            raise ValueError("VideoSearchTool requires `retrieval_service_url` in config")

        self.topk = int(config.get("topk", 5))
        self.timeout = int(config.get("timeout", 30))
        # Retry knobs for the backend="local" FastAPI call. The retrieval
        # server can be overwhelmed by the rollout fan-out (dozens of
        # ExecutionWorkers hitting it at once), so a single request often
        # times out. We retry transient failures with exponential backoff +
        # full jitter (see _do_http_local). max_retries=10 -> up to 11 attempts.
        # Each retry also widens the per-request timeout by ``retry_timeout_step``
        # seconds (attempt 0 → timeout, attempt 1 → timeout+1, ...) so a slow
        # server gets progressively more headroom instead of failing on the same
        # ReadTimeout every time.
        self.max_retries = int(config.get("max_retries", 10))
        self.retry_backoff = float(config.get("retry_backoff", 0.5))
        # Cap the exponential backoff so the jittered sleep window never exceeds
        # this many seconds (without it, retry_backoff * 2**attempt blows up to
        # hundreds of seconds at high attempt counts).
        self.retry_backoff_max = float(config.get("retry_backoff_max", 30.0))
        self.retry_timeout_step = float(config.get("retry_timeout_step", 1.0))
        self.num_workers = int(config.get("num_workers", 64))
        self.rate_limit = int(config.get("rate_limit", 64))
        self.enable_global_rate_limit = bool(config.get("enable_global_rate_limit", True))

        # YouTube cookies for the ``backend="remote"`` path (real yt-dlp search).
        # Resolution order: tool_config key → env var → None (anonymous).
        # resolve_youtube_cookies validates Netscape format and degrades to
        # anonymous (with a warning) on a missing/malformed file, so a bad
        # cookies.txt cannot break every remote search with
        # "does not look like a Netscape format cookies file".
        self.cookies_file = resolve_youtube_cookies(config.get("youtube_cookies_file"))

        self.execution_pool = init_execution_pool(
            num_workers=self.num_workers,
            enable_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            limiter_name="vss-video-search-rate-limiter",
        )

        # Lazy YouTube client — only constructed the first time a ``backend=
        # "remote"`` sample arrives. We don't want to import yt-dlp on every
        # rank when the typical run is 100% local.
        self._remote_client: Any = None

        logger.info(
            "VideoSearchTool ready (url=%s topk=%d rate_limit=%d)",
            self.retrieval_service_url,
            self.topk,
            self.rate_limit,
        )

    # ------------------------------------------------------------- verl API

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        """Snapshot per-sample ``backend`` from ``create_kwargs``.

        verl's ``tool_agent_loop._call_tool`` invokes:

            tool.create(create_kwargs=tools_kwargs[tool_name].get("create_kwargs", {}))

        so the dict passed in here is exactly the ``create_kwargs`` field of
        a single parquet row's ``extra_info.tools_kwargs.search_youtube``.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        ck = (kwargs.get("create_kwargs") or {})
        backend = str(ck.get("backend", "local")).lower().strip() or "local"
        if backend not in ("local", "remote"):
            logger.warning("VideoSearchTool: unknown backend=%r, falling back to 'local'", backend)
            backend = "local"
        # Per-sample domain-randomization config and gold identifiers. Data
        # preprocessing writes these two fields for training samples only (see
        # examples/data_preprocess/video_research_judge.py); validation and
        # real-evaluation samples omit them, so the defaults here are an empty
        # dict and an empty list and randomization stays off. ``execute()`` adds
        # a second hard gate on backend=="local" so that backend="remote" (the
        # live YouTube API, used at test time) is never perturbed.
        dr_config = ck.get("domain_randomization") or {}
        gold_urls = list(ck.get("gold_urls") or [])
        self._instance_dict[instance_id] = {
            "queries": [],
            "num_calls": 0,
            "retrieved_urls": [],
            "backend": backend,
            "corpus_shard": ck.get("corpus_shard", "default"),
            "domain_randomization": dr_config,
            "gold_urls": gold_urls,
        }
        return instance_id, ToolResponse()

    # ------------------------------------------------------------- backends

    def _do_http_local(self, query: str, topk: int) -> dict[str, Any]:
        """Blocking HTTP call to FastAPI retrieval service; runs in Ray worker.

        Robust against a transiently-overloaded server: transient failures
        (timeouts, connection refused/reset, protocol errors, and 5xx) are
        retried up to ``self.max_retries`` times (default 10) with exponential
        backoff and FULL JITTER, and the per-request timeout grows by
        ``self.retry_timeout_step`` seconds each attempt.
        The jitter is important — without it, a fleet of workers
        that all time out at the same instant would retry in lockstep and
        re-overload the server (a thundering herd). 4xx responses are NOT
        retried (the request itself is bad). When the retry budget is
        exhausted we still return the standard empty-result envelope so the
        agent trajectory stays alive instead of crashing the rollout.
        """
        payload = {"query": query, "topk": topk}
        last_err = "unknown"
        # One client for all attempts (connection reuse). The per-request
        # timeout widens by ``retry_timeout_step`` seconds on each successive
        # attempt (attempt 0 → self.timeout, attempt 1 → self.timeout + step,
        # ...) so a transiently slow retrieval server gets more headroom on
        # retry rather than hitting the same ReadTimeout repeatedly.
        with httpx.Client() as client:
            for attempt in range(self.max_retries + 1):
                attempt_timeout = self.timeout + attempt * self.retry_timeout_step
                try:
                    resp = client.post(
                        self.retrieval_service_url, json=payload, timeout=attempt_timeout
                    )
                    resp.raise_for_status()
                    return resp.json()
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response is not None else 0
                    last_err = f"http_{status}"
                    # 4xx = malformed/unsupported request; retrying is futile.
                    if status < 500:
                        logger.warning("video_search non-retryable HTTP %s: %s", status, e)
                        return {"query": query, "results": [], "error": f"http_error: {e}"}
                    # 5xx (server overloaded/unavailable) → fall through to retry.
                except httpx.TransportError as e:
                    # Base class covering TimeoutException (connect/read/write/
                    # pool), NetworkError (ConnectError/ReadError/...), and
                    # ProtocolError (RemoteProtocolError). All transient under
                    # load → retry.
                    last_err = repr(e)
                except Exception as e:  # noqa: BLE001 - keep trajectory alive on weird errors
                    logger.warning("video_search unexpected error: %s", e)
                    return {"query": query, "results": [], "error": repr(e)}

                # Reached only when the attempt failed with a retryable error.
                if attempt < self.max_retries:
                    # Exponential backoff with full jitter: sleep ~U(0, base),
                    # base doubling each attempt (retry_backoff, *2, *4, ...),
                    # capped at retry_backoff_max so the sleep never runs away.
                    base = min(self.retry_backoff * (2 ** attempt), self.retry_backoff_max)
                    time.sleep(random.uniform(0.0, base))

        logger.warning(
            "video_search exhausted %d retries for query=%r (last_err=%s)",
            self.max_retries, query[:80], last_err,
        )
        return {"query": query, "results": [], "error": f"retries_exhausted: {last_err}"}

    def _do_youtube_remote(self, query: str, topk: int) -> dict[str, Any]:
        """Blocking yt-dlp ytsearch; runs in Ray worker.

        Returns the same envelope shape as ``_do_http_local`` so callers
        downstream cannot tell the two paths apart.
        """
        if self._remote_client is None:
            ensure_video_agent_on_path()
            try:
                from video_agent.tools.video_search import YouTubeSearchTool  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover - environmental
                return {
                    "query": query,
                    "results": [],
                    "error": f"video_agent.tools.video_search unavailable: {e}",
                }
            # max_results is overridden per-call below; ctor value is just a default.
            self._remote_client = YouTubeSearchTool(
                max_results=topk, cookies_file=self.cookies_file
            )

        try:
            results = self._remote_client.search(query, max_results=topk) or []
        except Exception as e:  # noqa: BLE001
            logger.warning("video_search remote yt-dlp error: %s", e)
            return {"query": query, "results": [], "error": f"yt_dlp_error: {e}"}

        # Adapt VideoSearchResult dataclass fields → tool envelope dicts.
        out = []
        for r in results:
            out.append({
                "fake_url": r.url,                  # alias used by _format_tool_text
                "title": r.title,
                "snippet": r.description,
                "duration": r.duration,
                "thumbnail": r.thumbnail_url,
            })
        return {"query": query, "results": out}

    # ------------------------------------------------------------- execute

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        query = parameters.get("query", "")
        if not isinstance(query, str) or not query.strip():
            msg = "Error: `query` must be a non-empty string."
            return ToolResponse(text=msg), 0.0, {"error": msg}

        # `display_topk` is the number of results actually rendered to the model
        # (the max_results it requested). Accept both `max_results` (SFT-aligned)
        # and `topk` (legacy) parameters
        display_topk = int(parameters.get("max_results") or parameters.get("topk") or self.topk)
        display_topk = max(1, min(display_topk, 50))  # server also caps; defend-in-depth here

        rec = self._instance_dict.get(instance_id) or {}
        backend = rec.get("backend", "local")
        worker_fn = self._do_youtube_remote if backend == "remote" else self._do_http_local

        # ---- Domain-randomization gate ----
        # Belt and braces: randomization applies only when the dr config says
        # enable AND the backend is local (training). backend=='remote' (the live
        # YouTube API, used for testing and evaluation) is never perturbed, which
        # enforces "real API and no randomization at test time".
        dr_config = rec.get("domain_randomization") or {}
        dr_active = bool(dr_config.get("enable")) and backend == "local"

        # Decouple candidate pool size from display count: when randomization is
        # on, fetch extra candidates from the server (default 50, the server's
        # max_topk) so the surplus real corpus videos form a pool of hard
        # negatives that watch_video can still open. Only display_topk of them
        # reach the model. With randomization off the two are equal, so behaviour
        # is identical to before this layer existed.
        fetch_topk = display_topk
        if dr_active:
            fetch_topk = max(1, min(max(display_topk, int(dr_config.get("candidate_pool", 50))), 50))

        try:
            body_ref = self.execution_pool.execute.remote(worker_fn, query, fetch_topk)
            body = await body_ref  # ray's ObjectRef is awaitable in 2.x
        except Exception as e:  # noqa: BLE001
            logger.warning("VideoSearchTool pool error: %s", e)
            body = {"query": query, "results": [], "error": repr(e)}

        results = body.get("results", []) or []

        # ---- Apply retrieval-domain randomization (training/local backend only,
        #      and only when the server reported no error) ----
        dr_metrics: dict[str, Any] = {}
        if dr_active and not body.get("error"):
            gold_urls = set(rec.get("gold_urls") or [])
            # Deterministic per-call random source: the seed combines the base
            # seed, the instance_id, and how many searches this instance has
            # already issued. Different rollouts have different instance_ids, so
            # the same question lands in different difficulty tiers across the
            # GRPO group (the counterpart of mechanism 2); multiple searches
            # within one rollout also get different seeds, so successive rounds
            # are not perturbed identically.
            rng = random.Random(f"{int(dr_config.get('seed', 0))}:{instance_id}:{int(rec.get('num_calls', 0))}")
            results, dr_metrics = apply_domain_randomization(
                results,
                gold_urls=gold_urls,
                config=dr_config,
                rng=rng,
                display_topk=display_topk,
            )

        # hit_urls come from the results the model is actually shown (not the
        # server's raw candidate pool), so what the trajectory records and what
        # watch_video can open are exactly the urls the model saw.
        hit_urls = [(r.get("fake_url") or r.get("url") or "") for r in results]
        hit_urls = [u for u in hit_urls if u]

        if instance_id in self._instance_dict:
            rec = self._instance_dict[instance_id]
            rec["queries"].append(query)
            rec["retrieved_urls"].extend(hit_urls)
            rec["num_calls"] += 1

        tool_text = _format_tool_text(query, results)
        metrics = {
            "num_results": len(results),
            "latency_ms": float(body.get("latency_ms", 0.0) or 0.0),
            "retrieved_urls": hit_urls,
            "backend": backend,
            "error": body.get("error"),
        }
        if dr_metrics:
            # Surface the randomization tier and related monitoring (the metrics
            # entry point for mechanism 4)
            metrics["domain_randomization"] = dr_metrics
        return ToolResponse(text=tool_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Per-tool reward shaping is not used here; overall reward is
        # computed at trajectory end by ``custom_reward_function``.
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
