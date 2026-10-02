"""Stage 2 — VideoEntityGraph construction.

Per seed entity (which is used only as a **search inducement** and is NOT
counted as a node of the final graph):

    hop_idx = -1 (initial, not saved):
        1. ytsearch topk for ``seed_name`` → randomly pick 1 after duration filtering
        2. download → sparse frame extraction → per-frame VLM caption
        3. video-level synthesis → pick a **next_entity** (the FIRST real graph node)

    hop_idx = 0 .. target_depth-1 (saved into ``graph.entities``):
        4. use the previously chosen ``next_entity`` as the current entity
        5. repeat the ytsearch → download → captions → synthesis cycle
        6. the synthesis again picks a new ``next_entity`` for the *next* real hop

    Finally (optional):
        7. entity enrichment — for each real hop, do a light text-search pass
           to fill in ``properties`` + ``relations``.

Output: one JSON object per seed = one VideoEntityGraph with ``entities`` being
the ordered list of real hops (length == target_depth if every hop succeeded).
"""

from __future__ import annotations

import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from video_task_generation.data_structures import (
    VideoEntity,
    VideoEntityGraph,
    VideoFrameEvidence,
    VideoInfo,
)
from video_task_generation.prompts import (
    FRAME_CAPTION_SYSTEM,
    FRAME_CAPTION_USER,
    VIDEO_SYNTHESIS_PROMPT,
)
from video_task_generation.shared.checkpoint import StageCheckpoint
from video_task_generation.shared.jsonl_utils import append_jsonl_to_path, load_jsonl_safe
from video_task_generation.shared.llm_client import call_llm, call_vlm, extract_tag
from video_task_generation.shared.video_utils import (
    download_video,
    extract_frames,
    rank_videos,
    search_videos,
)
from video_task_generation.stages.entity_enrichment import enrich_graph_in_place

logger = logging.getLogger(__name__)


class GraphConstructor:
    """Expand each seed entity into a VideoEntityGraph chain."""

    def __init__(
        self,
        graph_depth: int = 4,
        depth_distribution: Optional[List[float]] = None,
        frames_per_video: int = 24,
        video_duration_min: int = 30,
        video_duration_max: int = 1500,
        ytsearch_topk: int = 8,
        frame_caption_workers: int = 4,
        graph_workers: int = 2,
        caption_max_chars: int = 400,
        enrich_enabled: bool = True,
        enrich_search_topk: int = 3,
        enrich_visit_topk: int = 2,
        enrich_workers: int = 4,
    ):
        self.graph_depth = max(1, graph_depth)
        # Normalise the depth distribution: drop negatives, renormalise to sum 1.
        # Empty / zero-sum distribution disables randomness.
        dd = [max(0.0, float(x)) for x in (depth_distribution or [])]
        total = sum(dd)
        if total > 0:
            self.depth_distribution: List[float] = [x / total for x in dd]
        else:
            self.depth_distribution = []
        # Cap distribution length to graph_depth; extra entries collapse into the last bucket.
        if len(self.depth_distribution) > self.graph_depth:
            head = self.depth_distribution[: self.graph_depth - 1]
            tail = sum(self.depth_distribution[self.graph_depth - 1 :])
            self.depth_distribution = head + [tail]
        self.frames_per_video = frames_per_video
        self.video_duration_min = video_duration_min
        self.video_duration_max = video_duration_max
        self.ytsearch_topk = ytsearch_topk
        self.frame_caption_workers = frame_caption_workers
        self.graph_workers = graph_workers
        self.caption_max_chars = caption_max_chars
        self.enrich_enabled = enrich_enabled
        self.enrich_search_topk = max(1, int(enrich_search_topk))
        self.enrich_visit_topk = max(0, int(enrich_visit_topk))
        self.enrich_workers = max(1, int(enrich_workers))

    def _sample_target_depth(self) -> int:
        """Draw the target graph depth for one seed."""
        if not self.depth_distribution:
            return self.graph_depth
        # weighted random: depth_distribution[i] corresponds to depth == i + 1
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(self.depth_distribution, start=1):
            cumulative += p
            if r <= cumulative:
                return i
        return len(self.depth_distribution)

    # ── Per-frame caption (parallel inside one video) ────────────────

    def _caption_frame(self, frame, entity: str, video_title: str) -> Optional[VideoFrameEvidence]:
        user_prompt = FRAME_CAPTION_USER.format(
            timestamp=frame.timestamp,
            title=video_title,
            entity=entity,
            max_chars=self.caption_max_chars,
        )
        try:
            resp = call_vlm(
                prompt=user_prompt,
                images_b64=[frame.image_b64],
                system_prompt=FRAME_CAPTION_SYSTEM,
                max_tokens=max(300, self.caption_max_chars + 80),
            )
        except Exception as exc:
            logger.warning("frame caption failed t=%.2fs: %s", frame.timestamp, exc)
            return None
        caption = resp.strip()
        if not caption:
            return None
        if len(caption) > self.caption_max_chars * 2:
            caption = caption[: self.caption_max_chars * 2]
        return VideoFrameEvidence(timestamp=frame.timestamp, caption=caption)

    def _caption_video(
        self, video_path: str, entity: str, video_title: str,
    ) -> List[VideoFrameEvidence]:
        frames = extract_frames(video_path, n_frames=self.frames_per_video)
        if not frames:
            return []
        evidences: List[Optional[VideoFrameEvidence]] = [None] * len(frames)
        with ThreadPoolExecutor(max_workers=self.frame_caption_workers) as pool:
            futures = {
                pool.submit(self._caption_frame, fr, entity, video_title): i
                for i, fr in enumerate(frames)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    ev = fut.result()
                    evidences[i] = ev
                except Exception as exc:
                    logger.warning("caption worker error: %s", exc)
        return [e for e in evidences if e is not None]

    # ── Video-level synthesis + next-entity selection ────────────────

    def _synthesise_video(
        self,
        entity: str,
        video_info: VideoInfo,
        frames: List[VideoFrameEvidence],
    ) -> Optional[dict]:
        if not frames:
            return None
        frame_lines = "\n".join(
            f"- t={e.timestamp:.2f}s: {e.caption}" for e in frames
        )
        prompt = VIDEO_SYNTHESIS_PROMPT.format(
            n_frames=len(frames),
            title=video_info.title,
            channel=video_info.channel or "unknown",
            duration=int(video_info.duration or 0),
            entity=entity,
            frame_captions=frame_lines,
        )
        resp = call_llm([{"role": "user", "content": prompt}])
        summary = extract_tag(resp, "summary")
        next_entity = extract_tag(resp, "next_entity")
        reason = extract_tag(resp, "reason")
        if not summary or not next_entity:
            logger.warning("video synthesis missing tags (entity=%s)", entity)
            return None
        return {
            "summary": summary.strip(),
            "next_entity": next_entity.strip(),
            "reason": (reason or "").strip(),
        }

    # ── Full chain for one seed ──────────────────────────────────────

    # How many candidates to try before giving up on one hop.
    MAX_DOWNLOAD_TRIES_PER_HOP = 4

    def _expand_one_hop(
        self,
        entity_name: str,
        depth: int,
        category: Optional[str],
    ) -> Optional[VideoEntity]:
        """Try candidate videos for ``entity_name`` in order until one yields captions.

        Returns None iff no candidate could be downloaded + captioned.
        """
        results = search_videos(entity_name, topk=self.ytsearch_topk)
        if not results:
            logger.info("[hop %d] no yt results for '%s'", depth, entity_name)
            return None
        candidates = rank_videos(
            results,
            min_duration=float(self.video_duration_min),
            max_duration=float(self.video_duration_max),
            strategy="random",
        )
        if not candidates:
            logger.info("[hop %d] no valid video for '%s'", depth, entity_name)
            return None

        for attempt, choice in enumerate(candidates[: self.MAX_DOWNLOAD_TRIES_PER_HOP], 1):
            logger.info(
                "[hop %d] attempt %d/%d: %s (dur=%s)",
                depth, attempt, min(self.MAX_DOWNLOAD_TRIES_PER_HOP, len(candidates)),
                choice.url, choice.duration,
            )
            video_path = download_video(choice.url)
            if not video_path:
                logger.info("[hop %d] download failed for %s, trying next", depth, choice.url)
                continue

            video_info = VideoInfo(
                video_id=choice.video_id,
                url=choice.url,
                title=choice.title,
                channel=choice.channel,
                duration=choice.duration,
                view_count=choice.view_count,
                upload_date=choice.upload_date,
            )

            frames = self._caption_video(video_path, entity_name, choice.title)
            if not frames:
                logger.info("[hop %d] no captions produced for %s, trying next", depth, choice.url)
                continue

            synth = self._synthesise_video(entity_name, video_info, frames)
            return VideoEntity(
                name=entity_name,
                depth=depth,
                category=category,
                video=video_info,
                frames=frames,
                video_summary=(synth or {}).get("summary"),
                next_entity=(synth or {}).get("next_entity"),
                next_entity_reason=(synth or {}).get("reason"),
            )

        logger.info(
            "[hop %d] all %d candidates exhausted for '%s'",
            depth, min(self.MAX_DOWNLOAD_TRIES_PER_HOP, len(candidates)), entity_name,
        )
        return None

    def _build_graph(
        self,
        seed_name: str,
        category: Optional[str],
        target_depth: Optional[int] = None,
    ) -> Optional[VideoEntityGraph]:
        """Build the chain with the initial ``seed_name`` used only as an inducement.

        ``target_depth`` is interpreted as the **number of real nodes** the
        final graph should contain.  ``seed_name`` itself is NOT added to
        ``graph.entities``; instead we do one extra "seed hop" whose picked
        ``next_entity`` becomes the first real node, then extend further.
        """
        graph = VideoEntityGraph()
        visited: set[str] = set()

        max_depth = target_depth if target_depth is not None else self.graph_depth
        max_depth = max(1, min(self.graph_depth, int(max_depth)))

        # ── Seed hop (inducement only, not saved) ─────────────────────
        seed_key = seed_name.strip().lower()
        visited.add(seed_key)
        seed_hop = self._expand_one_hop(seed_name, depth=-1, category=category)
        if seed_hop is None or not seed_hop.next_entity:
            logger.info(
                "[graph] seed-hop failed for '%s' (no hit or no next_entity)", seed_name,
            )
            return None

        current_name = seed_hop.next_entity

        # ── Real hops (written into graph.entities) ───────────────────
        for depth in range(max_depth):
            key = current_name.strip().lower()
            if key in visited:
                logger.info(
                    "[graph] cycle detected at '%s' (real hop %d), stopping",
                    current_name, depth,
                )
                break
            visited.add(key)

            # category tag only on the first real hop (inherited from seed category)
            hop = self._expand_one_hop(current_name, depth, category if depth == 0 else None)
            if hop is None:
                logger.info(
                    "[graph] real hop %d failed for '%s' (seed=%s), stopping chain",
                    depth, current_name, seed_name,
                )
                break
            graph.add(hop)

            if not hop.next_entity:
                break
            current_name = hop.next_entity

        if graph.depth == 0:
            return None
        return graph

    # ── Public run ───────────────────────────────────────────────────

    def run(
        self,
        seeds_path: Path,
        output_path: Path,
        checkpoint_path: Optional[Path] = None,
    ) -> List[dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        seeds = load_jsonl_safe(seeds_path)
        if not seeds:
            raise RuntimeError(f"no seed entities found in {seeds_path}")

        existing = load_jsonl_safe(output_path)
        existing_keys = set()
        for r in existing:
            nm = (r.get("seed_meta") or {}).get("name") or (
                (r.get("entities") or [{}])[0].get("name", "")
            )
            if nm:
                existing_keys.add(_graph_key(nm))

        ckpt = StageCheckpoint(checkpoint_path) if checkpoint_path else None

        pending = []
        for s in seeds:
            name = s.get("name", "").strip()
            if not name:
                continue
            key = _graph_key(name)
            if key in existing_keys:
                continue
            if ckpt is not None and ckpt.is_processed(key):
                continue
            pending.append(s)

        logger.info(
            "[Stage 2] %d seeds pending (of %d total, %d already done)",
            len(pending), len(seeds), len(existing_keys),
        )

        results: List[dict] = []
        lock = threading.Lock()

        def _proc(seed: dict) -> Tuple[Optional[VideoEntityGraph], int]:
            name = seed["name"]
            category = seed.get("category")
            language = str(seed.get("language") or "").strip().lower() or "zh"
            target_depth = self._sample_target_depth()
            try:
                graph = self._build_graph(name, category, target_depth=target_depth)
            except Exception as exc:
                logger.error("[Stage 2] graph build error for '%s': %s", name, exc, exc_info=True)
                return None, target_depth

            if graph is not None and self.enrich_enabled:
                try:
                    enrich_graph_in_place(
                        graph,
                        language=language,
                        search_topk=self.enrich_search_topk,
                        visit_topk=self.enrich_visit_topk,
                        workers=self.enrich_workers,
                    )
                except Exception as exc:
                    logger.warning(
                        "[Stage 2] enrichment failed for '%s' (non-fatal): %s", name, exc,
                    )
            return graph, target_depth

        with ThreadPoolExecutor(max_workers=self.graph_workers) as pool:
            futures = {pool.submit(_proc, s): s for s in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                seed = futures[fut]
                name = seed["name"]
                key = _graph_key(name)
                try:
                    graph, target_depth = fut.result()
                except Exception as exc:
                    logger.error("[Stage 2] [%d/%d] '%s': %s", i, len(pending), name, exc)
                    if ckpt is not None:
                        ckpt.mark(key, "error")
                    continue

                if graph is None:
                    if ckpt is not None:
                        ckpt.mark(key, "empty")
                    logger.info(
                        "[Stage 2] [%d/%d] '%s' produced empty graph (target_depth=%d)",
                        i, len(pending), name, target_depth,
                    )
                    continue

                record = graph.to_dict()
                record["seed_meta"] = seed
                record["target_depth"] = target_depth
                with lock:
                    append_jsonl_to_path(record, output_path)
                    results.append(record)
                if ckpt is not None:
                    ckpt.mark(key, f"done:{graph.depth}")
                logger.info(
                    "[Stage 2] [%d/%d] '%s' ok — target_depth=%d actual_depth=%d path=%s",
                    i, len(pending), name, target_depth, graph.depth,
                    " → ".join(graph.transition_path),
                )

        logger.info(
            "[Stage 2] done — %d new graphs (total=%d) → %s",
            len(results), len(existing) + len(results), output_path,
        )
        return results


def _graph_key(name: str) -> str:
    return name.strip().lower()
