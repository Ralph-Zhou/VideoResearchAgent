"""Sharded mp4 storage for the corpus.

Why this exists
---------------
Materialising 50K+ mp4 files (each ~30 MB) on a shared filesystem like
lustre / cephfs creates two real problems:

1. **Metadata pressure.** A single directory with 50K+ entries hammers the
   MDT; even with two-level hashing it still stresses the metadata server
   far more than a handful of large files do.
2. **Random-access cost.** ``watch_video`` opens each file once per
   rollout; a few thousand parallel rollouts can spike open() rates.

We solve both by packing every N videos into one **uncompressed tar shard**
(``shard_00000.tar``) on a user-chosen large-capacity directory. A
companion ``shard_index.parquet`` records, for each video, the byte range
inside its shard. Random read is then a plain ``open + seek + read length``
on a file the OS keeps page-cached cheaply — no tarfile parsing required.

URI scheme
----------
A video stored in shards is referenced via a virtual ``local_path`` of the
form::

    shard://<shard_relpath>?o=<offset>&l=<length>

where ``<shard_relpath>`` is the shard file's path **relative to the shard
root** (not absolute, so the corpus is portable across machines). Examples::

    shard://shard_00000.tar?o=512&l=31457280

These URIs:
- start with ``shard://`` so any consumer can detect them with a string
  check before falling through to filesystem access;
- never resolve as ``Path(...).is_file() == True``, which keeps the
  existing watch_video fallback cascade well-defined;
- carry enough information to resolve the bytes without consulting the
  parquet index, so cold-start tools don't have to load the index at all.

The ``shard_index.parquet`` is still produced for debugging / manifest /
re-deriving fake URLs across rebuilds.
"""

from __future__ import annotations

import logging
import re
import tarfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote

import pandas as pd

logger = logging.getLogger(__name__)


_SHARD_URI_PREFIX = "shard://"


# --------------------------------------------------------------------------- URI


def make_shard_uri(shard_relpath: str, offset: int, length: int) -> str:
    """Build a stable virtual ``shard://`` URI for a given byte range."""
    return f"{_SHARD_URI_PREFIX}{quote(shard_relpath, safe='/')}?o={int(offset)}&l={int(length)}"


def is_shard_uri(value: str) -> bool:
    """Return True iff ``value`` looks like a ``shard://`` URI."""
    return isinstance(value, str) and value.startswith(_SHARD_URI_PREFIX)


@dataclass(frozen=True)
class ShardLocator:
    """Parsed components of a ``shard://`` URI."""

    shard_relpath: str
    offset: int
    length: int


def parse_shard_uri(uri: str) -> ShardLocator:
    """Parse a ``shard://`` URI back into ``(shard_relpath, offset, length)``."""
    if not is_shard_uri(uri):
        raise ValueError(f"Not a shard URI: {uri!r}")
    # urlparse treats ``shard://...`` as ``scheme=shard, netloc=<first segment>``
    # which is awkward for relative paths, so we strip the prefix manually.
    body = uri[len(_SHARD_URI_PREFIX) :]
    if "?" not in body:
        raise ValueError(f"Shard URI missing query string: {uri!r}")
    relpath, _, query = body.partition("?")
    relpath = unquote(relpath)
    qs = parse_qs(query, strict_parsing=True)
    try:
        offset = int(qs["o"][0])
        length = int(qs["l"][0])
    except (KeyError, IndexError, ValueError) as e:
        raise ValueError(f"Malformed shard URI {uri!r}: {e}") from e
    return ShardLocator(shard_relpath=relpath, offset=offset, length=length)


# --------------------------------------------------------------------------- write


@dataclass
class _ShardEntry:
    video_id: str
    shard_relpath: str
    offset: int
    length: int
    size_bytes: int  # == length, kept explicit for readability in the parquet
    source_id: str = ""


class ShardedTarWriter:
    """Append-only writer that packs mp4 bytes into rolling tar shards.

    Usage::

        with ShardedTarWriter(root, prefix="shard", videos_per_shard=1024) as w:
            for video_id, mp4_bytes, source_id in stream:
                shard_uri = w.write(video_id, mp4_bytes, source_id=source_id)
                # ... attach shard_uri to the VideoRecord
        index_path = w.write_index()   # produces shard_index.parquet

    The tar entry name is ``<video_id>.mp4``. We seek the underlying file
    pointer to capture the **payload** offset (after the 512-byte tar header),
    so readers can use plain ``open + seek + read`` without parsing tar at all.

    Resume semantics
    ----------------
    When ``resume=True`` and ``root`` already contains shards, the writer
    refuses to append to the trailing shard (tar append mode is fragile);
    instead it skips past the highest existing shard index and starts a
    *new* shard at ``max_existing + 1``. The trailing shard is left
    untouched (potentially under-full) — wasted space is bounded by one
    shard, which is acceptable at our scale.
    """

    def __init__(
        self,
        root: Path,
        *,
        prefix: str = "shard",
        videos_per_shard: int = 1024,
        max_shard_bytes: int | None = None,
        resume: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.videos_per_shard = max(1, int(videos_per_shard))
        # Cap an individual shard's size in bytes (best-effort soft cap; we
        # check before adding the next entry). ``None`` disables the cap.
        self.max_shard_bytes = max_shard_bytes

        self._shard_idx: int = -1
        self._shard_path: Path | None = None
        self._tar: tarfile.TarFile | None = None
        self._fp = None  # the underlying binary file object, for tell()
        self._entries_in_shard: int = 0
        self._index: list[_ShardEntry] = []
        self._closed: bool = False

        # Resume: pick up after the highest existing shard so we never touch
        # an already-written tar (which may be truncated / mid-entry on a
        # crash). We let ``_open_new_shard`` advance ``_shard_idx`` from
        # the last existing one when it next runs.
        if resume:
            existing = self._discover_existing_shards()
            if existing:
                self._shard_idx = max(existing)
                logger.info(
                    "ShardedTarWriter resume: found shards 0..%d under %s; "
                    "next shard will be %d",
                    self._shard_idx,
                    self.root,
                    self._shard_idx + 1,
                )

    def _discover_existing_shards(self) -> list[int]:
        pat = re.compile(rf"^{re.escape(self.prefix)}_(\d{{5}})\.tar$")
        out: list[int] = []
        for p in self.root.iterdir():
            m = pat.match(p.name)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    # ----- context manager

    def __enter__(self) -> "ShardedTarWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ----- internal

    def _open_new_shard(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
            self._fp = None
        self._shard_idx += 1
        self._shard_path = self.root / f"{self.prefix}_{self._shard_idx:05d}.tar"
        # Use ``tarfile.open(name, mode='w')`` (no compression). We open with
        # the underlying file object so we can call ``tell()`` to record
        # payload offsets reliably across Python versions.
        self._fp = self._shard_path.open("wb")
        self._tar = tarfile.open(fileobj=self._fp, mode="w", format=tarfile.USTAR_FORMAT)
        self._entries_in_shard = 0
        logger.info("Opened shard %s", self._shard_path)

    def _ensure_shard(self, incoming_bytes: int) -> None:
        if self._tar is None:
            self._open_new_shard()
            return
        if self._entries_in_shard >= self.videos_per_shard:
            self._open_new_shard()
            return
        if self.max_shard_bytes is not None and self._fp is not None:
            # tar adds a 512-byte header per entry, plus zero-padding.
            projected = self._fp.tell() + 512 + incoming_bytes
            if projected > self.max_shard_bytes:
                self._open_new_shard()

    # ----- public API

    def write(self, video_id: str, payload: bytes, *, source_id: str = "") -> str:
        """Append one mp4 to the current shard, return its ``shard://`` URI.

        ``video_id`` must be filesystem-safe (it is used as the tar entry
        name). The corpus pipeline already enforces this.
        """
        if self._closed:
            raise RuntimeError("ShardedTarWriter is closed")

        self._ensure_shard(len(payload))
        assert self._tar is not None and self._fp is not None and self._shard_path is not None

        info = tarfile.TarInfo(name=f"{video_id}.mp4")
        info.size = len(payload)
        info.mtime = int(time.time())
        info.mode = 0o644
        info.type = tarfile.REGTYPE

        # Capture the offset AFTER the header is written -- that's where the
        # payload starts. ``addfile`` writes header then payload; we measure
        # ``tell()`` immediately before/after to bracket the payload.
        before_header = self._fp.tell()
        self._tar.addfile(info, BytesIO(payload))
        after_payload = self._fp.tell()
        # Payload offset is header_end == after_payload - padded(payload).
        # tar pads payload to 512-byte boundary, so:
        padded = ((info.size + 511) // 512) * 512
        payload_offset = after_payload - padded
        if payload_offset < before_header + 512:
            # Defensive: should never happen with USTAR, but if it does we
            # bail loudly rather than write a corrupted index.
            raise RuntimeError(
                f"Unexpected tar layout: before={before_header} "
                f"after={after_payload} size={info.size} padded={padded}"
            )

        shard_relpath = self._shard_path.name  # already relative to self.root
        uri = make_shard_uri(shard_relpath, payload_offset, info.size)
        self._index.append(
            _ShardEntry(
                video_id=video_id,
                shard_relpath=shard_relpath,
                offset=payload_offset,
                length=info.size,
                size_bytes=info.size,
                source_id=source_id,
            )
        )
        self._entries_in_shard += 1
        return uri

    def write_index(self, name: str = "shard_index.parquet") -> Path:
        """Persist the in-memory index to ``<root>/<name>`` and return the path.

        When called multiple times across a resumed build, you should pass a
        run-scoped ``name`` (e.g. ``"shard_index_run_3.parquet"``) so the
        previous run's index is not overwritten; a separate finalize step
        can merge all per-run files into a single ``shard_index.parquet``.
        """
        if not self._index:
            logger.warning("ShardedTarWriter index is empty; writing empty parquet anyway")
        df = pd.DataFrame([e.__dict__ for e in self._index])
        out = self.root / name
        df.to_parquet(out, index=False)
        return out

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._tar is not None:
                self._tar.close()
            if self._fp is not None:
                self._fp.close()
        finally:
            self._tar = None
            self._fp = None
            self._closed = True

    @property
    def num_shards(self) -> int:
        """Number of tar shards opened so far (0 if nothing was written)."""
        return max(0, self._shard_idx + 1)

    @property
    def num_entries(self) -> int:
        """Total number of mp4 entries written across all shards."""
        return len(self._index)


def scan_existing_shards(
    root: Path,
    *,
    prefix: str = "shard",
) -> list[_ShardEntry]:
    """Walk every ``<prefix>_NNNNN.tar`` in ``root`` and rebuild a flat entry list.

    Used at finalize time to (a) sanity-check that on-disk shards are still
    readable after a crash and (b) recover the shard_index for shards
    written by an interrupted run that didn't get to call ``write_index``.

    Returns entries in shard-then-payload-offset order. Reads the tar header
    only (no payload), so this is cheap (~ms per shard).
    """
    out: list[_ShardEntry] = []
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{5}})\.tar$")
    shard_paths = sorted(p for p in Path(root).iterdir() if pat.match(p.name))
    for sp in shard_paths:
        try:
            with tarfile.open(sp, "r:") as t:
                for member in t:
                    if not member.isfile():
                        continue
                    name = member.name
                    if not name.endswith(".mp4"):
                        continue
                    out.append(
                        _ShardEntry(
                            video_id=name[: -len(".mp4")],
                            shard_relpath=sp.name,
                            offset=member.offset_data,
                            length=member.size,
                            size_bytes=member.size,
                            source_id="",
                        )
                    )
        except (tarfile.TarError, OSError) as e:
            logger.warning(
                "Failed to read shard %s during scan (will be skipped): %s",
                sp,
                e,
            )
            continue
    return out


# --------------------------------------------------------------------------- read


# Process-wide LRU of opened shard file descriptors. ``watch_video`` typically
# hits the same handful of shards repeatedly within one rollout batch; keeping
# fds warm avoids 'open()' cost on every frame extraction. We don't bother
# with a real LRU library — a small dict + arbitrary cap is enough for our
# scale (tens of shards).
_FD_CACHE: dict[str, int] = {}
_FD_CACHE_CAP = 64


def read_shard_bytes(uri: str, shard_root: str | Path) -> bytes:
    """Resolve a ``shard://`` URI to the raw mp4 bytes.

    ``shard_root`` is the directory that contains the tar shards (the same
    path passed to ``ShardedTarWriter``). The shard file path is
    ``shard_root / locator.shard_relpath``.
    """
    locator = parse_shard_uri(uri)
    root = Path(shard_root).expanduser().resolve()
    shard_path = (root / locator.shard_relpath).resolve()
    # Guard against path traversal via crafted URIs.
    if not str(shard_path).startswith(str(root)):
        raise ValueError(f"Shard path escapes root: {shard_path} not under {root}")
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    fd = _FD_CACHE.get(str(shard_path))
    if fd is None:
        import os
        fd = os.open(str(shard_path), os.O_RDONLY)
        if len(_FD_CACHE) >= _FD_CACHE_CAP:
            # Evict an arbitrary entry; for our access pattern this is fine.
            old_path, old_fd = next(iter(_FD_CACHE.items()))
            _FD_CACHE.pop(old_path, None)
            try:
                os.close(old_fd)
            except OSError:
                pass
        _FD_CACHE[str(shard_path)] = fd

    import os
    data = os.pread(fd, locator.length, locator.offset)
    if len(data) != locator.length:
        raise IOError(
            f"Short read from {shard_path}: got {len(data)} of {locator.length} bytes "
            f"at offset {locator.offset}"
        )
    return data


# --------------------------------------------------------------------------- materialise


# Match valid shard URIs without depending on urllib for the hot path.
_SAFE_VIDEO_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


def materialise_to_tempfile(
    uri: str,
    shard_root: str | Path,
    tmp_dir: str | Path = "/tmp",
) -> Path:
    """Read bytes from a shard and dump them to a uniquely-named temp mp4.

    Returns the path to the temp file. Caller is responsible for unlinking
    it once decoding is done. We do **not** use ``NamedTemporaryFile`` because
    its delete-on-close semantics are awkward across processes; an explicit
    ``unlink`` in the caller's ``finally`` block is clearer.
    """
    locator = parse_shard_uri(uri)
    # Deterministic temp name keyed on (shard, offset, length) lets concurrent
    # callers for the same video share the file rather than racing.
    stem = f"{Path(locator.shard_relpath).stem}_{locator.offset}_{locator.length}"
    if not _SAFE_VIDEO_ID.match(stem):
        # Sanitise just in case: allow only the safe charset above.
        stem = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    tmp_path = Path(tmp_dir).expanduser() / f"vss_shard_{stem}.mp4"
    if tmp_path.is_file() and tmp_path.stat().st_size == locator.length:
        return tmp_path
    payload = read_shard_bytes(uri, shard_root)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish write: write to a sibling .part then rename.
    part = tmp_path.with_suffix(tmp_path.suffix + ".part")
    part.write_bytes(payload)
    part.replace(tmp_path)
    return tmp_path
