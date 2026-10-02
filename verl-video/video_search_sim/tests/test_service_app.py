"""End-to-end app test with stubbed retrievers.

We construct a fake ``Corpus`` in memory and monkey-patch the service's
``load_corpus`` + ``DenseRetriever`` + ``BM25Retriever`` so the app can spin
up without CLIP / FAISS / any files on disk. The test asserts the *wiring*
between the FastAPI app, retrievers, fusion and schema.

We use ``fastapi.testclient.TestClient`` (starlette-based) rather than
``httpx.ASGITransport`` because the latter does not, by default, execute the
``lifespan`` handler that loads the corpus into ``app.state``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from video_search_sim.common.config import CLIPConfig, RetrievalConfig, ServerConfig, ServiceConfig
from video_search_sim.common.schemas import VideoRecord, VideoSearchResponse
from video_search_sim.retrieval_service import app as app_module


def _make_records(n: int = 3) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    for i in range(n):
        records.append(
            VideoRecord(
                fake_url=f"https://www.youtube.com/watch?v=vid{i:07d}",
                video_id=f"vid{i:07d}".ljust(11, "0")[:11],
                local_path=f"/fake/{i}.mp4",
                title=f"Video {i}",
                description=f"Description for video {i} about cats and dogs",
                duration=100.0 + i,
            )
        )
    return records


def _make_stub_corpus(records: list[VideoRecord]):
    """Assemble a mock ``Corpus`` without touching disk / FAISS / BM25 state."""
    import pandas as pd

    # videos_df is only used for display; loader uses the records list.
    df = pd.DataFrame([r.model_dump() for r in records])

    corpus = SimpleNamespace(
        root=Path("/tmp/fake_corpus"),
        videos=records,
        videos_df=df,
        url_to_path={r.fake_url: r.local_path for r in records},
        bm25_backend="bm25s",
        bm25_state=object(),  # non-None -> BM25Retriever path is exercised
        faiss_index=object(),
        keyframe_meta=pd.DataFrame(columns=["video_idx", "timestamp", "vector_row"]),
        embedding_dim=768,
        manifest={"num_videos": len(records), "version": 1},
    )
    corpus.num_videos = len(records)
    corpus.num_keyframes = 0
    return corpus


class _StubBM25:
    def __init__(self, *_a, **_kw): ...
    def search(self, query: str, topk: int):
        # Return documents in a fixed permutation to exercise fusion.
        return [(0, 2.0), (2, 1.0), (1, 0.5)][:topk]


class _StubDense:
    def __init__(self, *_a, **_kw): ...
    def search(self, query: str, topk: int, candidate_pool: int | None = None):
        return [(1, 0.9), (0, 0.5), (2, 0.1)][:topk]


@pytest.fixture
def stub_client(monkeypatch):
    """FastAPI TestClient wired against stubbed retrievers.

    Using TestClient as a context manager guarantees that the app's lifespan
    handler (which populates ``app.state.service``) runs before tests touch
    any endpoint.
    """
    records = _make_records(3)
    corpus = _make_stub_corpus(records)

    # Patch the loader + retrievers at the import site used by `app.create_app`.
    monkeypatch.setattr(app_module, "load_corpus", lambda _root: corpus)
    monkeypatch.setattr(app_module, "BM25Retriever", _StubBM25)
    monkeypatch.setattr(app_module, "DenseRetriever", _StubDense)

    class _DummyEmbedder:
        def __init__(self, *_a, **_kw): ...
        def encode_texts(self, _):
            return None

    monkeypatch.setattr(app_module, "CLIPEmbedder", _DummyEmbedder)

    cfg = ServiceConfig(
        corpus_dir=Path("/tmp/fake_corpus"),
        clip=CLIPConfig(device="cpu"),
        retrieval=RetrievalConfig(enable_bm25=True, enable_dense=True, default_topk=3, max_topk=10),
        server=ServerConfig(),
    )

    app = app_module.create_app(cfg)
    with TestClient(app) as client:
        yield client


def test_healthz(stub_client) -> None:
    resp = stub_client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["num_videos"] == 3


def test_video_search_returns_fused_results(stub_client) -> None:
    resp = stub_client.post("/video_search", json={"query": "dogs", "topk": 3})
    assert resp.status_code == 200

    body = VideoSearchResponse.model_validate(resp.json())
    assert body.query == "dogs"
    assert len(body.results) == 3

    urls = {r.fake_url for r in body.results}
    assert urls == {
        "https://www.youtube.com/watch?v=vid0000000",
        "https://www.youtube.com/watch?v=vid0000001",
        "https://www.youtube.com/watch?v=vid0000002",
    }

    # Video 0 is at rank 1 in BM25 and rank 2 in dense -> canonical RRF winner.
    top = body.results[0]
    assert top.fake_url == "https://www.youtube.com/watch?v=vid0000000"
    assert top.thumbnail.endswith("/hqdefault.jpg")
    assert top.score > 0


def test_video_search_rejects_empty_query(stub_client) -> None:
    resp = stub_client.post("/video_search", json={"query": "", "topk": 3})
    assert resp.status_code == 422


def test_video_search_caps_topk_at_corpus_size(stub_client) -> None:
    # Requesting topk=10 is valid by the request schema (le=50); the fused
    # result set is bounded by the corpus and stub retrievers (3 items).
    resp = stub_client.post("/video_search", json={"query": "dogs", "topk": 10})
    assert resp.status_code == 200
    body = VideoSearchResponse.model_validate(resp.json())
    assert len(body.results) == 3


def test_video_search_request_validation_rejects_oversized_topk(stub_client) -> None:
    # VideoSearchRequest caps ``topk`` at 50.
    resp = stub_client.post("/video_search", json={"query": "dogs", "topk": 999})
    assert resp.status_code == 422
