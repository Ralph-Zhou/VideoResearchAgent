"""Fabricate YouTube-style URLs and IDs for local videos.

We need the strings emitted by the local video_search service to be
indistinguishable in *shape* from real YouTube URLs, because the SFT corpus
was collected with the real tool and downstream tokenisation is sensitive to
URL structure.

Design:

- ID is 11 characters, drawn from YouTube's alphabet ``[A-Za-z0-9_-]``.
- ID is derived deterministically from the local path via SHA-1, so rebuilds
  on the same path yield the same ID (important for incremental indexing and
  for reproducibility of RL rollouts).
"""

from __future__ import annotations

import base64
import hashlib
import re

_YT_ID_LEN = 11
_YT_URL_PREFIX = "https://www.youtube.com/watch?v="
_YT_THUMB_FMT = "https://www.youtube.com/vi/{vid}/hqdefault.jpg"

# Base64url alphabet == YouTube ID alphabet, so we can slice the digest
# directly without any further character substitution.
_FAKE_URL_RE = re.compile(r"^https://www\.youtube\.com/watch\?v=([A-Za-z0-9_-]{11})$")


def fake_video_id(local_path: str) -> str:
    """Deterministically derive an 11-char YouTube-style ID from ``local_path``.

    The digest is URL-safe base64-encoded so every byte lands in
    ``[A-Za-z0-9_-]``; we then keep the first 11 characters.

    Args:
        local_path: Absolute (or at least stable) path of the video file.

    Returns:
        11-character YouTube-style ID. Stable across rebuilds.
    """
    digest = hashlib.sha1(local_path.encode("utf-8")).digest()
    b64 = base64.urlsafe_b64encode(digest).decode("ascii")
    return b64[:_YT_ID_LEN]


def fake_url_from_path(local_path: str) -> str:
    """Convenience wrapper returning the full fake watch URL."""
    return _YT_URL_PREFIX + fake_video_id(local_path)


def thumbnail_url(video_id: str) -> str:
    """Return a fake hqdefault thumbnail URL for the given video ID."""
    return _YT_THUMB_FMT.format(vid=video_id)


def parse_fake_url(url: str) -> str | None:
    """Extract the 11-char ID from a fake URL. Returns None if it doesn't match."""
    m = _FAKE_URL_RE.match(url.strip())
    return m.group(1) if m else None
