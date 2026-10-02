"""pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Register custom markers and skip integration tests by default."""
    skip_integration = pytest.mark.skip(reason="integration test; run with `pytest -m integration`")
    if not config.getoption("-m", default=""):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require network / external models (CLIP, HF, FAISS-GPU)",
    )
