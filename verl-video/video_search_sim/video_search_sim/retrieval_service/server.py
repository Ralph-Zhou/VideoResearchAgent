"""``vss-serve`` CLI entry point.

Launches the FastAPI app with uvicorn. Kept distinct from ``app.py`` so tests
can exercise the app in-process without binding a port.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

from ..common.config import ServiceConfig, load_yaml_config
from .app import create_app

# When ``workers > 1`` uvicorn must import the app from a string (it re-imports
# in each forked worker), so it cannot receive a pre-built app object. We hand
# the workers the config-file path (+ corpus_dir override) via env vars and let
# each process re-load the YAML and rebuild the app in ``create_app_from_env``.
_CONFIG_PATH_ENV = "VSS_SERVICE_CONFIG_PATH"
_CORPUS_DIR_ENV = "VSS_SERVICE_CORPUS_DIR"


def create_app_from_env():
    """uvicorn factory used for the multi-worker launch path (factory=True)."""
    cfg_path = os.environ.get(_CONFIG_PATH_ENV)
    if not cfg_path:
        raise RuntimeError(f"{_CONFIG_PATH_ENV} not set; cannot build app in worker process.")
    cfg = load_yaml_config(cfg_path, ServiceConfig)
    assert isinstance(cfg, ServiceConfig)
    corpus_override = os.environ.get(_CORPUS_DIR_ENV)
    if corpus_override:
        cfg.corpus_dir = Path(corpus_override).expanduser().resolve()
    logging.basicConfig(
        level=cfg.server.log_level.upper(),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    return create_app(cfg)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="vss-serve",
        description="Run the local video_search FastAPI service.",
    )
    ap.add_argument("--config", type=Path, required=True, help="Path to a ServiceConfig YAML file")
    ap.add_argument("--corpus-dir", type=Path, default=None, help="Override config.corpus_dir")
    ap.add_argument("--host", type=str, default=None, help="Override config.server.host")
    ap.add_argument("--port", type=int, default=None, help="Override config.server.port")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cfg_obj = load_yaml_config(args.config, ServiceConfig)
    assert isinstance(cfg_obj, ServiceConfig)

    if args.corpus_dir is not None:
        cfg_obj.corpus_dir = args.corpus_dir.expanduser().resolve()
    if args.host is not None:
        cfg_obj.server.host = args.host
    if args.port is not None:
        cfg_obj.server.port = args.port

    logging.basicConfig(
        level=cfg_obj.server.log_level.upper(),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    workers = cfg_obj.server.workers or 1
    if workers > 1:
        # Multi-process path: each worker re-imports and rebuilds the app (its
        # own GIL + its own in-memory index, so size workers by RAM). Required
        # because uvicorn ignores ``workers`` when handed an app instance.
        os.environ[_CONFIG_PATH_ENV] = str(args.config)
        os.environ[_CORPUS_DIR_ENV] = str(cfg_obj.corpus_dir)
        uvicorn.run(
            "video_search_sim.retrieval_service.server:create_app_from_env",
            factory=True,
            host=cfg_obj.server.host,
            port=cfg_obj.server.port,
            log_level=cfg_obj.server.log_level,
            workers=workers,
        )
    else:
        app = create_app(cfg_obj)
        uvicorn.run(
            app,
            host=cfg_obj.server.host,
            port=cfg_obj.server.port,
            log_level=cfg_obj.server.log_level,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
