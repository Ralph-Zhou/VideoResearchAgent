"""Stage 1 — Video-oriented seed entity generation.

Generate a diverse pool of seed entities suitable as YouTube queries,
using a category × region × era batched LLM prompt (similar to the browsecomp
URL generation stage, but tailored to video-friendly categories).
"""

from __future__ import annotations

import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Tuple

from video_task_generation.prompts import (
    ERA_MODIFIERS,
    REGION_MODIFIERS,
    SEED_ENTITY_CATEGORIES,
    SEED_LANGUAGES,
    seed_generation_prompt_for,
)
from video_task_generation.shared.checkpoint import StageCheckpoint
from video_task_generation.shared.jsonl_utils import append_jsonl_to_path, load_jsonl_safe
from video_task_generation.shared.llm_client import call_llm

logger = logging.getLogger(__name__)


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")


def detect_language(name: str) -> str:
    """Heuristic: return 'zh' if the string has any CJK/Kana char, else 'en'."""
    if not name:
        return "en"
    return "zh" if _CJK_RE.search(name) else "en"


def _build_slots() -> List[Tuple[str, str, str, str, str]]:
    """Return all (language, category, subcategory, region, era) combos.

    The language dimension is enforced at the LLM prompt level so each slot
    deterministically produces one of {zh, en} seed records.
    """
    out: List[Tuple[str, str, str, str, str]] = []
    for lang in SEED_LANGUAGES:
        for cat, subs in SEED_ENTITY_CATEGORIES.items():
            for sub in subs:
                out.append((lang, cat, sub, "不限", "不限"))
        for cat, subs in SEED_ENTITY_CATEGORIES.items():
            for sub in subs:
                region = random.choice(REGION_MODIFIERS)
                out.append((lang, cat, sub, region, "不限"))
        for cat, subs in SEED_ENTITY_CATEGORIES.items():
            for sub in subs:
                era = random.choice(ERA_MODIFIERS)
                out.append((lang, cat, sub, "不限", era))
    return out


class SeedGenerator:
    """Produce a JSONL of ``{name, category, subcategory, region, era}`` records."""

    _EXCLUSION_SAMPLE_SIZE = 20

    def __init__(self, num_seeds: int = 80, batch_size: int = 25, workers: int = 8):
        self.num_seeds = num_seeds
        self.batch_size = batch_size
        self.workers = workers

    def _generate_batch(
        self,
        language: str,
        category: str,
        subcategory: str,
        region: str,
        era: str,
        existing: Set[str],
    ) -> List[str]:
        sample_exist = (
            random.sample(sorted(existing), min(self._EXCLUSION_SAMPLE_SIZE, len(existing)))
            if existing else []
        )
        if sample_exist:
            if language == "zh":
                exclusion_clause = f"- 不要重复这些已经生成过的 entity: {', '.join(sample_exist)}\n"
            else:
                exclusion_clause = f"- Do NOT repeat any of these already-generated entities: {', '.join(sample_exist)}\n"
        else:
            exclusion_clause = ""

        template = seed_generation_prompt_for(language)
        prompt = template.format(
            batch_size=self.batch_size,
            category=category,
            subcategory=subcategory,
            region=region,
            era=era,
            exclusion_clause=exclusion_clause,
        )
        try:
            resp = call_llm([{"role": "user", "content": prompt}])
        except Exception as exc:
            logger.warning("seed batch LLM failed (%s/%s/%s): %s", language, category, subcategory, exc)
            return []
        raw = [
            ln.strip().strip("-").strip("•").strip("·").strip("*").strip()
            for ln in resp.split("\n") if ln.strip()
        ]
        filtered: List[str] = []
        for name in raw:
            if not (2 <= len(name) <= 80):
                continue
            if name.lower() in existing:
                continue
            # language compliance filter: the LLM occasionally drifts.
            detected = detect_language(name)
            if detected != language:
                continue
            filtered.append(name)
        return filtered[: self.batch_size]

    def run(
        self,
        output_path: Path,
        checkpoint_path: Optional[Path] = None,
    ) -> List[dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing = load_jsonl_safe(output_path)
        existing_names: Set[str] = {r["name"].strip().lower() for r in existing if "name" in r}
        logger.info(
            "[Stage 1] already have %d seed entities (target=%d)",
            len(existing_names), self.num_seeds,
        )
        if len(existing_names) >= self.num_seeds:
            return []

        ckpt = StageCheckpoint(checkpoint_path) if checkpoint_path else None
        all_slots = _build_slots()
        random.shuffle(all_slots)

        # Roughly estimate how many slots are needed from batch_size, then
        # oversample by a factor of 1.5
        needed = self.num_seeds - len(existing_names)
        estimate = max(4, int(needed / max(1, self.batch_size) * 1.8))
        slots = all_slots[: estimate]

        # Drop slots already processed (keyed on the checkpoint key)
        if ckpt is not None:
            slots = [s for s in slots if not ckpt.is_processed(_slot_key(s))]

        logger.info("[Stage 1] running %d slots with %d workers", len(slots), self.workers)

        new_records: List[dict] = []
        seen = set(existing_names)

        def _one(slot: Tuple[str, str, str, str, str]):
            lang, cat, sub, region, era = slot
            names = self._generate_batch(lang, cat, sub, region, era, seen)
            return slot, names

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(_one, s): s for s in slots}
            for fut in as_completed(futures):
                slot, names = fut.result()
                lang, cat, sub, region, era = slot
                added = 0
                for name in names:
                    k = name.strip().lower()
                    if k in seen:
                        continue
                    rec = {
                        "name": name,
                        "language": lang,
                        "category": cat,
                        "subcategory": sub,
                        "region": region,
                        "era": era,
                    }
                    append_jsonl_to_path(rec, output_path)
                    new_records.append(rec)
                    seen.add(k)
                    added += 1
                if ckpt is not None:
                    ckpt.mark(_slot_key(slot), f"done:{added}")
                logger.info(
                    "[Stage 1] slot [%s] %s/%s/%s/%s → +%d (total=%d)",
                    lang, cat, sub, region, era, added, len(seen),
                )
                if len(seen) >= self.num_seeds:
                    logger.info("[Stage 1] reached target — stopping submissions")
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

        logger.info(
            "[Stage 1] done — added %d new seeds (total=%d) → %s",
            len(new_records), len(seen), output_path,
        )
        return new_records


def _slot_key(slot: Tuple[str, str, str, str, str]) -> str:
    return "|".join(slot)
