"""End-to-end pipeline orchestrator for Video DeepResearch task generation."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from video_task_generation.config import PipelineConfig
from video_task_generation.shared.checkpoint import StageCheckpoint
from video_task_generation.shared.jsonl_utils import load_jsonl_safe
from video_task_generation.shared.llm_client import set_defaults as set_llm_defaults
from video_task_generation.shared.text_search import configure_search
from video_task_generation.shared.video_utils import configure_video
from video_task_generation.stages.difficulty_enhancement import DifficultyEnhancementStage
from video_task_generation.stages.graph_construction import GraphConstructor
from video_task_generation.stages.seed_generation import SeedGenerator
from video_task_generation.stages.task_generation import TaskGenerationStage

logger = logging.getLogger(__name__)


STAGE1_FILENAME = "stage1_seeds.jsonl"
STAGE2_FILENAME = "stage2_graphs.jsonl"
STAGE3_FILENAME = "stage3_tasks.jsonl"
STAGE3_REJECT_FILENAME = "stage3_rejected.jsonl"
STAGE4_FILENAME = "stage4_tasks.jsonl"
STAGE4_REJECT_FILENAME = "stage4_rejected.jsonl"
SUMMARY_FILENAME = "run_summary.json"
CHECKPOINT_DIR = ".checkpoints"


class VideoTaskGenerationWorkflow:

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _bootstrap(self) -> None:
        """Push config to module-level singletons.

        Two runtime modes are supported:

        * ``online`` (default) — original behaviour: Serper + yt-dlp.
        * ``local`` — text-search is disabled and Stage 2 sources videos
          from the local corpus via HTTP ``/video_search`` + CorpusBridge.
          Stages 3 and 4 short-circuit the text-agent leg via
          :func:`is_text_search_enabled`.
        """
        llm = self.config.llm
        set_llm_defaults(
            model_text=llm.model_text,
            model_vision=llm.model_vision,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            timeout=llm.timeout,
            max_retries=llm.max_retries,
        )
        mode = (self.config.runtime.mode or "online").lower().strip()

        if mode == "local":
            lc = self.config.local_corpus
            # Hard-fail early if mis-configured rather than mid-pipeline.
            if not lc.corpus_dir:
                raise ValueError(
                    "runtime.mode='local' requires local_corpus.corpus_dir; "
                    "set it to the directory produced by vss-build-corpus."
                )
            # Text-search mode comes from local_corpus.text_search_mode.
            configure_search(
                provider=self.config.search.provider,
                max_results=self.config.search.max_results,
                enable_visit=self.config.search.enable_visit,
                visit_timeout=self.config.search.visit_timeout,
                mode=lc.text_search_mode,
            )
            configure_video(
                cache_dir=self.config.yt_dlp.cache_dir,
                max_resolution=self.config.stage2_graph_construction.max_video_resolution,
                ytsearch_topk=self.config.stage2_graph_construction.ytsearch_topk,
                backend="local",
                local_corpus_dir=lc.corpus_dir,
                local_service_url=lc.service_url,
                local_service_timeout=lc.service_timeout,
                local_shard_root=lc.shard_root,
                local_shard_materialise_dir=lc.shard_materialise_dir,
                local_top_k=lc.top_k,
            )
            logger.info(
                "[runner] runtime=local corpus_dir=%s service=%s text_search=%s",
                lc.corpus_dir, lc.service_url, lc.text_search_mode,
            )
            return

        # online (default)
        s = self.config.search
        configure_search(
            provider=s.provider,
            max_results=s.max_results,
            enable_visit=s.enable_visit,
            visit_timeout=s.visit_timeout,
            mode="keep_serper",
        )
        configure_video(
            cache_dir=self.config.yt_dlp.cache_dir,
            max_resolution=self.config.stage2_graph_construction.max_video_resolution,
            ytsearch_topk=self.config.stage2_graph_construction.ytsearch_topk,
            backend="online",
        )
        logger.info("[runner] runtime=online (Serper + yt-dlp)")

    def run(
        self,
        skip_seeds: bool = False,
        skip_graphs: bool = False,
        skip_tasks: bool = False,
        skip_stage4: bool = False,
        only_stage4: bool = False,
    ) -> List[dict]:
        self._bootstrap()

        start = datetime.now()
        root = Path(self.config.workflow.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        ckpt_dir = root / CHECKPOINT_DIR
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        s1_path = root / STAGE1_FILENAME
        s2_path = root / STAGE2_FILENAME
        s3_path = root / STAGE3_FILENAME
        s3_reject_path = root / STAGE3_REJECT_FILENAME
        s4_path = root / STAGE4_FILENAME
        s4_reject_path = root / STAGE4_REJECT_FILENAME

        s1_ckpt = ckpt_dir / "stage1.ckpt"
        s2_ckpt = ckpt_dir / "stage2.ckpt"
        s3_ckpt = ckpt_dir / "stage3.ckpt"
        s4_ckpt = ckpt_dir / "stage4.ckpt"

        if only_stage4:
            skip_seeds = skip_graphs = skip_tasks = True

        logger.info("=" * 60)
        logger.info("Video DeepResearch Task Generation")
        logger.info("=" * 60)
        logger.info("Output dir: %s", root)
        logger.info("Target tasks: %d", self.config.workflow.target_tasks)

        # ── Stage 1 ──
        if skip_seeds:
            logger.info("[Stage 1] skipped — reading %s", s1_path)
        else:
            cfg1 = self.config.stage1_seed_generation
            SeedGenerator(
                num_seeds=cfg1.num_seeds,
                batch_size=cfg1.batch_size,
                workers=cfg1.workers,
            ).run(s1_path, checkpoint_path=s1_ckpt)

        # ── Stage 2 ──
        if skip_graphs:
            logger.info("[Stage 2] skipped — reading %s", s2_path)
        else:
            cfg2 = self.config.stage2_graph_construction
            GraphConstructor(
                graph_depth=cfg2.graph_depth,
                depth_distribution=cfg2.depth_distribution,
                frames_per_video=cfg2.frames_per_video,
                video_duration_min=cfg2.video_duration_min,
                video_duration_max=cfg2.video_duration_max,
                ytsearch_topk=cfg2.ytsearch_topk,
                frame_caption_workers=cfg2.frame_caption_workers,
                graph_workers=cfg2.graph_workers,
                caption_max_chars=cfg2.caption_max_chars,
                enrich_enabled=cfg2.enrich_enabled,
                enrich_search_topk=cfg2.enrich_search_topk,
                enrich_visit_topk=cfg2.enrich_visit_topk,
                enrich_workers=cfg2.enrich_workers,
            ).run(s1_path, s2_path, checkpoint_path=s2_ckpt)

        # ── Stage 3 ──
        s3_accepted: List[dict] = []
        if skip_tasks:
            logger.info("[Stage 3] skipped — reading %s", s3_path)
            s3_accepted = load_jsonl_safe(s3_path)
        else:
            cfg3 = self.config.stage3_task_generation
            TaskGenerationStage(
                workers=cfg3.workers,
                max_agent_turns=cfg3.max_agent_turns,
                oversampling_factor=cfg3.oversampling_factor,
                max_qa_retries=cfg3.max_qa_retries,
                self_check=cfg3.self_check,
                require_frame_evidence=cfg3.require_frame_evidence,
                target_tasks=self.config.workflow.target_tasks,
            ).run(
                s2_path, s3_path, reject_path=s3_reject_path,
                checkpoint_path=s3_ckpt,
            )
            s3_accepted = load_jsonl_safe(s3_path)

        # ── Stage 4 ──
        cfg4 = self.config.stage4_difficulty_enhancement
        final: List[dict]
        if skip_stage4 or not cfg4.enabled:
            logger.info(
                "[Stage 4] skipped (enabled=%s, skip_flag=%s)", cfg4.enabled, skip_stage4,
            )
            final = s3_accepted
        else:
            DifficultyEnhancementStage(
                workers=cfg4.workers,
                max_rounds=cfg4.max_rounds,
                max_agent_turns=cfg4.max_agent_turns,
            ).run(
                s3_path, s4_path, reject_path=s4_reject_path,
                checkpoint_path=s4_ckpt,
            )
            final = load_jsonl_safe(s4_path)

        self._write_summary(root, start, final)
        return final

    # ── Summary ─────────────────────────────────────────────────────

    def _write_summary(self, root: Path, start: datetime, final: List[dict]) -> None:
        elapsed = datetime.now() - start
        s1 = load_jsonl_safe(root / STAGE1_FILENAME)
        s2 = load_jsonl_safe(root / STAGE2_FILENAME)
        s3_accepted = load_jsonl_safe(root / STAGE3_FILENAME)
        s3_reject = load_jsonl_safe(root / STAGE3_REJECT_FILENAME)
        s4_reject = load_jsonl_safe(root / STAGE4_REJECT_FILENAME)

        reject_reasons: dict = {}
        for r in s3_reject:
            reason = r.get("reject_reason", "unknown")
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        s4_reasons: dict = {}
        for r in s4_reject:
            reason = r.get("reject_reason", "unknown")
            s4_reasons[reason] = s4_reasons.get(reason, 0) + 1

        summary = {
            "workflow": "video_task_generation",
            "target_tasks": self.config.workflow.target_tasks,
            "stage1_seeds": len(s1),
            "stage2_graphs": len(s2),
            "stage3_accept": len(s3_accepted),
            "stage3_reject": len(s3_reject),
            "stage3_reject_reasons": reject_reasons,
            "stage4_accept": len(final),
            "stage4_reject": len(s4_reject),
            "stage4_reject_reasons": s4_reasons,
            "target_reached": len(final) >= self.config.workflow.target_tasks,
            "elapsed_seconds": elapsed.total_seconds(),
            "config": self.config.model_dump(),
        }
        (root / SUMMARY_FILENAME).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        logger.info("=" * 60)
        logger.info("Pipeline Summary")
        logger.info("=" * 60)
        logger.info("  Stage 1 seeds     : %d", summary["stage1_seeds"])
        logger.info("  Stage 2 graphs    : %d", summary["stage2_graphs"])
        logger.info("  Stage 3 accept    : %d", summary["stage3_accept"])
        logger.info("  Stage 3 reject    : %d", summary["stage3_reject"])
        for reason, cnt in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            logger.info("    - %-30s %d", reason, cnt)
        logger.info("  Stage 4 accept    : %d", summary["stage4_accept"])
        logger.info("  Stage 4 reject    : %d", summary["stage4_reject"])
        for reason, cnt in sorted(s4_reasons.items(), key=lambda x: -x[1]):
            logger.info("    - %-30s %d", reason, cnt)
        logger.info("  Target reached    : %s", summary["target_reached"])
        logger.info("  Elapsed           : %s", elapsed)
        logger.info("=" * 60)
