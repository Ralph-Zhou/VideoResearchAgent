"""Shared helpers for safe, retryable yt-dlp calls."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, TextIO

# With session cookies, YouTube's default logged-in client can return an
# unusable ``tv_downgraded`` response: video formats fail ("The page needs to be
# reloaded" / "No video formats found") and captions are not listed. The
# ``android_vr`` client does not support cookies at all. These clients keep
# cookie-backed downloads and subtitle lookups working.
COOKIE_PLAYER_CLIENTS = ["default", "web_embedded"]


def use_cookie_player_clients(opts: dict) -> None:
    """Pin the YouTube player clients that work with a cookie file in ``opts``."""
    opts["extractor_args"] = {"youtube": {"player_client": list(COOKIE_PLAYER_CLIENTS)}}


def cookiefile_snapshot(
    cookies_file: Optional[str],
    *,
    logger: logging.Logger,
) -> Optional[TextIO]:
    """Return an in-memory copy of a Netscape cookie file.

    yt-dlp writes its cookie jar back when ``YoutubeDL.close()`` runs. Passing
    the same on-disk cookie file to several evaluation workers can therefore
    truncate/corrupt the file while another worker is reading it. A fresh
    StringIO for every YoutubeDL instance keeps the initial cookies but makes
    yt-dlp's write-back private to that call.
    """
    if not cookies_file:
        return None

    try:
        text = Path(cookies_file).read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Unable to read yt-dlp cookies from %s: %s", cookies_file, exc)
        return None

    header_lines = text.lstrip("\ufeff\r\n ").splitlines()[:5]
    if not any("Netscape HTTP Cookie File" in line for line in header_lines):
        logger.warning(
            "Ignoring invalid yt-dlp cookie file %s (expected Netscape format)",
            cookies_file,
        )
        return None

    return io.StringIO(text)
