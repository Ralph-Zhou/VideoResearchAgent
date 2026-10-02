"""Offline tests for the outer yt-dlp retry and cookie-isolation logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yt_dlp

from video_agent.tools.transcript import TranscriptFetcher
from video_agent.tools.video_download import VideoDownloader
from video_agent.tools.video_search import YouTubeSearchTool


class YtDlpRetryTests(unittest.TestCase):
    def test_video_download_retries_403_with_fresh_ytdlp(self):
        calls = {"downloads": 0, "instances": 0}

        class FakeYoutubeDL:
            def __init__(self, opts):
                calls["instances"] += 1
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def extract_info(self, url, download=False):
                return {
                    "id": "abcdefghijk",
                    "ext": "mp4",
                    "formats": [{"vcodec": "avc1", "acodec": "mp4a"}],
                }

            def download(self, urls):
                calls["downloads"] += 1
                if calls["downloads"] < 3:
                    raise yt_dlp.utils.DownloadError(
                        "unable to download video data: HTTP Error 403: Forbidden"
                    )
                Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"video")

            def prepare_filename(self, info):
                return self.opts["outtmpl"].replace("%(ext)s", info["ext"])

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "video_agent.tools.video_download.yt_dlp.YoutubeDL", FakeYoutubeDL
        ):
            downloader = VideoDownloader(
                cache_dir=tmpdir,
                max_download_attempts=3,
                retry_backoff_sec=0,
            )
            result = downloader.download(
                "https://www.youtube.com/watch?v=abcdefghijk"
            )

            self.assertEqual(result, str(Path(tmpdir) / "abcdefghijk.mp4"))
            self.assertEqual(calls, {"downloads": 3, "instances": 3})

    def test_video_download_does_not_retry_permanent_unavailable(self):
        calls = 0

        class FakeYoutubeDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def extract_info(self, url, download=False):
                return {"formats": [{"vcodec": "avc1", "acodec": "mp4a"}]}

            def download(self, urls):
                nonlocal calls
                calls += 1
                raise yt_dlp.utils.DownloadError(
                    "[youtube] abcdefghijk: Video unavailable"
                )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "video_agent.tools.video_download.yt_dlp.YoutubeDL", FakeYoutubeDL
        ):
            downloader = VideoDownloader(
                cache_dir=tmpdir,
                max_download_attempts=3,
                retry_backoff_sec=0,
            )
            result = downloader.download(
                "https://www.youtube.com/watch?v=abcdefghijk"
            )

        self.assertIsNone(result)
        self.assertEqual(calls, 1)

    def test_cookie_requests_use_cookie_compatible_player_clients(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = Path(tmpdir) / "cookies.txt"
            cookie_path.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            with_cookies = VideoDownloader(cache_dir=tmpdir, cookies_file=str(cookie_path))
            without_cookies = VideoDownloader(cache_dir=tmpdir)

            for attempt in (0, 2):
                opts = with_cookies._build_ydl_opts("abcdefghijk", attempt=attempt)
                self.assertEqual(
                    opts["extractor_args"]["youtube"]["player_client"],
                    ["default", "web_embedded"],
                )
            self.assertNotIn(
                "extractor_args", without_cookies._build_ydl_opts("abcdefghijk", attempt=0)
            )
            self.assertEqual(
                without_cookies._build_ydl_opts("abcdefghijk", attempt=2)[
                    "extractor_args"
                ]["youtube"]["player_client"],
                ["web_safari", "android_vr"],
            )

    def test_cookie_file_is_snapshotted_for_every_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cookie_path = root / "cookies.txt"
            cookie_path.write_text(
                "# Netscape HTTP Cookie File\n"
                ".youtube.com\tTRUE\t/\tFALSE\t0\tSID\tvalue\n",
                encoding="utf-8",
            )
            downloader = VideoDownloader(
                cache_dir=str(root / "videos"),
                cookies_file=str(cookie_path),
            )

            first = downloader._build_ydl_opts("abcdefghijk", attempt=0)
            second = downloader._build_ydl_opts("abcdefghijk", attempt=1)

            first_snapshot = first["cookiefile"]
            second_snapshot = second["cookiefile"]
            self.assertIsNot(first_snapshot, second_snapshot)
            for snapshot in (first_snapshot, second_snapshot):
                self.assertTrue(
                    snapshot.read().startswith("# Netscape HTTP Cookie File")
                )
                self.assertNotIsInstance(snapshot, (str, Path))
                snapshot.seek(0)
            with yt_dlp.YoutubeDL(
                {"cookiefile": first_snapshot, "quiet": True}
            ) as ydl:
                self.assertEqual(len(ydl.cookiejar), 1)
            self.assertTrue(
                cookie_path.read_text(encoding="utf-8").startswith(
                    "# Netscape HTTP Cookie File"
                )
            )

    def test_youtube_search_retries(self):
        calls = 0

        class FakeYoutubeDL:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def extract_info(self, query, download=False):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("temporary network error")
                return {"entries": [{"id": "abcdefghijk", "title": "test"}]}

        with patch("video_agent.tools.video_search.yt_dlp.YoutubeDL", FakeYoutubeDL):
            results = YouTubeSearchTool(
                max_attempts=2,
                retry_backoff_sec=0,
            ).search("query")

        self.assertEqual(calls, 2)
        self.assertEqual(
            [result.video_id for result in results],
            ["abcdefghijk"],
        )

    def test_subtitle_download_retries_and_ignores_missing_media_formats(self):
        calls = 0
        seen_options = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts
                seen_options.append(opts)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def download(self, urls):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden")
                Path(self.opts["outtmpl"] + ".en.json3").write_text(
                    json.dumps(
                        {
                            "events": [
                                {
                                    "tStartMs": 1000,
                                    "dDurationMs": 500,
                                    "segs": [{"utf8": "hello"}],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "yt_dlp.YoutubeDL", FakeYoutubeDL
        ):
            fetcher = TranscriptFetcher(
                cache_dir=tmpdir,
                max_ytdlp_attempts=2,
                retry_backoff_sec=0,
            )
            segments = fetcher._fetch_ytdlp(
                "https://www.youtube.com/watch?v=abcdefghijk"
            )

        self.assertEqual(calls, 2)
        self.assertTrue(seen_options[0]["ignore_no_formats_error"])
        self.assertEqual(segments[0].text, "hello")


if __name__ == "__main__":
    unittest.main()
