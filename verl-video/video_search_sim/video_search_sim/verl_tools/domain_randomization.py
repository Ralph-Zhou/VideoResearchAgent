"""Retrieval-domain randomization (RDR) — the implementation of mechanism 1.

Motivation
------------------------------------------------------------------
A simulated retrieval environment is too clean and closed-world. A policy
trained in it latches onto three spurious invariants: the gold video is always
retrievable, the ranking is always reliable, and the reward depends only on the
final answer. The policy therefore learns the shortcut "trust the retriever,
scan the top few hits, answer immediately", which collapses under zero-shot
transfer to live YouTube.

This module inserts one layer of per-call randomization inside the **client-side
tool process** — after ``VideoSearchTool.execute()`` receives the candidate list
from the server and before the list is rendered to the model. It perturbs
simulated retrieval into a harder superset that brackets live-retrieval
difficulty, forcing backend-agnostic robust behaviour: query rewriting,
multi-round recovery, and watching the video to confirm.

Four sub-mechanisms (the four bullets of mechanism 1)
------------------------------------------------------------------
1. Random gold sinking (breaks "closed-world retrievability"): with some
   probability the gold video sinks from the RRF top toward the middle or
   bottom of the list, or is removed from the display window entirely (gold is
   unreachable this round).
2. Hard-negative injection (breaks "scan the titles and you are done"): real
   corpus videos ranked lower in the candidate pool — retrieved by the same
   query and topically near-duplicate — are promoted into the display window,
   forcing the agent to open and watch a video to confirm.
3. Presentation randomization (breaks templated overfitting to the FineVideo
   caption style): titles and snippets are truncated, given clickbait prefixes,
   or case-jittered to approach the noisy title distribution of live YouTube.
4. Structural noise: the number of displayed results K is randomized and empty
   results are occasionally injected (simulating retrieval failure), forcing
   rerouting behaviour.

Safety invariants
------------------------------------------------------------------
* **Text only, never URLs.** Injected negative samples must use ``fake_url``
  values that genuinely exist in the corpus (they already come from the server's
  retrieval for the same query), because at training time ``watch_video`` runs
  with ``enable_remote_fallback=false`` and a fabricated URL would fail to
  resolve, breaking the search-to-watch chain. This module **only reorders,
  filters, and rewrites text**; it never synthesizes a URL.
* **Difficulty tiers.** Each call independently samples a tier
  (easy/medium/hard), which naturally gives different rollouts within a GRPO
  group different difficulty (the counterpart of mechanism 2) and leaves an
  interface for curriculum annealing (mechanism 4, which only needs
  ``tier_probs`` to change with training progress).

This module is **pure functions with no verl dependency** and can be unit-tested
in isolation. All randomness is driven by an externally supplied
``random.Random`` instance so every call is reproducible.
"""

from __future__ import annotations

import copy
import random
from typing import Any

# ---------------------------------------------------------------------------
# Default randomization config. The full tier table lives in code rather than in
# every parquet row, so per-sample ``create_kwargs`` only needs to carry a light
# switch such as {"enable": true, ...}. This avoids duplicating a large nested
# table across every training sample, which would inflate storage and memory.
# ``_resolve_config`` deep-merges per-sample overrides onto this default table.
# ---------------------------------------------------------------------------
DEFAULT_DR_CONFIG: dict[str, Any] = {
    "enable": False,            # master switch; set True only for train samples (see data prep)
    "candidate_pool": 50,       # candidates requested from the server (hard-negative source; server max_topk=50)
    "seed": 0,                  # base random seed (combined with instance_id and call index for the per-call rng)
    # Probability of sampling each difficulty tier.
    # Historical note: a static 0.34/0.33/0.33 split put a third of all calls in
    # the hard tier from step 0 (gold drop 0.30 + p_empty 0.08). Before the
    # policy had learned the basic task, the flood of unsolvable/empty-result
    # samples induced a query-retry loop, response length blew up from 30k to
    # 45k characters, entropy collapsed, validation accuracy fell from 0.67 to
    # 0.40, and the growth eventually exhausted host RAM and triggered an OOM.
    # The default now starts easy-skewed and lets ``tier_probs_for_step``
    # anneal difficulty upward (mechanism 4).
    "tier_probs": {"easy": 0.55, "medium": 0.35, "hard": 0.10},
    # Per-tier perturbation strength
    "tiers": {
        # easy: almost no perturbation — lets the policy learn the basic task
        # early in training (the starting point of the mechanism 4 curriculum)
        "easy": {
            "gold_actions": {"keep": 1.0},        # gold stays at the top
            "n_hard_neg": [0, 0],                  # no hard negatives injected
            "p_title_jitter": 0.0,
            "p_snippet_jitter": 0.0,
            "p_empty": 0.0,                        # never produce an empty result
            "k_jitter": [0, 0],                    # display count is fixed
        },
        # medium: gold occasionally sinks, a few hard negatives, moderate text jitter
        "medium": {
            "gold_actions": {"keep": 0.50, "sink_mid": 0.35, "sink_bottom": 0.13, "drop": 0.02},
            "n_hard_neg": [2, 4],
            "p_title_jitter": 0.4,
            "p_snippet_jitter": 0.4,
            "p_empty": 0.01,
            "k_jitter": [-1, 2],
        },
        # hard: gold sinks or disappears often, many hard negatives, heavy text
        # jitter, occasional empty results. This tier is what brackets real
        # YouTube difficulty and is key to zero-shot transfer.
        # drop / p_empty are well below their original values: drop=0.30 with
        # p_empty=0.08 produced many rounds with no reachable gold at all, which
        # combined with max_assistant_turns=50 to induce a query-retry loop (the
        # long "Wait... Wait, maybe..." rollouts). Gold still sinks to the
        # bottom frequently (forcing video confirmation) but is rarely removed
        # outright, keeping the reward signal intact.
        "hard": {
            "gold_actions": {"keep": 0.25, "sink_mid": 0.30, "sink_bottom": 0.33, "drop": 0.12},
            "n_hard_neg": [4, 8],
            "p_title_jitter": 0.8,
            "p_snippet_jitter": 0.8,
            "p_empty": 0.03,
            "k_jitter": [-2, 3],
        },
    },
}

# Clickbait prefix pool: pollutes clean FineVideo titles into a live-YouTube style
_CLICKBAIT_PREFIXES = [
    "MUST WATCH: ",
    "You won't believe ",
    "[VIRAL] ",
    "SHOCKING - ",
    "Top 10 ",
    "(2024) ",
    "*NEW* ",
    "INSANE!! ",
]


def _url_of(r: dict[str, Any]) -> str:
    """Return a result's url uniformly (server uses fake_url, remote path uses url)."""
    return r.get("fake_url") or r.get("url") or ""


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    """Sample one name from a weight dict {name: weight}."""
    names = list(weights.keys())
    w = [max(0.0, float(weights[n])) for n in names]
    total = sum(w)
    if total <= 0:
        return names[0]
    return rng.choices(names, weights=w, k=1)[0]


def tier_probs_for_step(
    progress: float,
    *,
    start: dict[str, float] | None = None,
    end: dict[str, float] | None = None,
) -> dict[str, float]:
    """Curriculum annealing (mechanism 4): interpolate tier_probs linearly from
    easy to hard as training progresses.

    With a fully static tier_probs, the hard tier occupied a third of all calls
    from step 0 and the policy collapsed before learning the basic task. This
    function raises difficulty smoothly with ``progress`` (0 to 1, typically
    ``global_step / total_steps`` clamped to [0, 1]) from ``start`` (easy-skewed)
    to ``end`` (hard-skewed).

    A caller — data preprocessing re-baking the parquet per epoch, or the tool
    side injecting per step — only needs to place the return value into
    ``tier_probs`` in create_kwargs; no other logic in this module changes.
    """
    start = start or {"easy": 0.70, "medium": 0.25, "hard": 0.05}
    end = end or {"easy": 0.20, "medium": 0.40, "hard": 0.40}
    p = min(1.0, max(0.0, float(progress)))
    keys = ("easy", "medium", "hard")
    raw = {k: (1.0 - p) * float(start.get(k, 0.0)) + p * float(end.get(k, 0.0)) for k in keys}
    total = sum(raw.values()) or 1.0
    return {k: v / total for k, v in raw.items()}


def _resolve_config(user_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge per-sample overrides onto ``DEFAULT_DR_CONFIG``.

    A per-sample config usually passes only ``{"enable": True,
    "candidate_pool": 50}``. To change the difficulty distribution or the
    strength of one tier, override ``tier_probs`` / ``tiers`` in create_kwargs.
    """
    cfg = copy.deepcopy(DEFAULT_DR_CONFIG)
    for k, v in (user_cfg or {}).items():
        if k == "tiers" and isinstance(v, dict):
            for tier_name, tier_override in v.items():
                cfg["tiers"].setdefault(tier_name, {}).update(tier_override or {})
        elif k == "tier_probs" and isinstance(v, dict):
            cfg["tier_probs"].update(v)
        else:
            cfg[k] = v
    return cfg


def _mangle_title(title: str, rng: random.Random) -> str:
    """Rewrite a title at random: clickbait prefix / uppercase / truncate / exclamation."""
    if not title:
        return title
    op = rng.choice(["clickbait", "upper", "truncate", "exclaim"])
    if op == "clickbait":
        return rng.choice(_CLICKBAIT_PREFIXES) + title
    if op == "upper":
        return title.upper()
    if op == "truncate":
        cut = max(8, int(len(title) * rng.uniform(0.4, 0.8)))
        return title[:cut].rstrip() + "..."
    # exclaim
    return title.rstrip(".") + rng.choice([" !!", " ??", " 🔥", " | FULL VIDEO"])


def _mangle_snippet(snippet: str, rng: random.Random) -> str:
    """Rewrite a snippet at random: truncate / lowercase / hard cut."""
    if not snippet:
        return snippet
    op = rng.choice(["truncate", "lower", "hardcut"])
    if op == "truncate":
        cut = max(10, int(len(snippet) * rng.uniform(0.3, 0.7)))
        return snippet[:cut].rstrip() + "..."
    if op == "lower":
        return snippet.lower()
    # hardcut: mimics the very short descriptions live YouTube truncates to
    return snippet[: rng.randint(15, 40)].rstrip() + "..."


def _dedup_by_url(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by url, keeping first-seen order."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in items:
        u = _url_of(r)
        if u and u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out


def apply_domain_randomization(
    results: list[dict[str, Any]],
    *,
    gold_urls: set[str],
    config: dict[str, Any],
    rng: random.Random,
    display_topk: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply retrieval-domain randomization to the server's candidate list and
    return (display list, metrics).

    Parameters
    ----------
    results : list[dict]
        The RRF-sorted candidate pool returned by the server, up to
        ``candidate_pool`` entries long. Each entry carries at least
        ``fake_url``/``url``, ``title``, and ``snippet``.
    gold_urls : set[str]
        URLs of the gold answer videos for this question, passed through from
        data preprocessing into create_kwargs. When empty, hard-negative
        injection and text jitter still apply; only targeted gold sinking is
        unavailable.
    config : dict
        Per-sample randomization config, deep-merged onto ``DEFAULT_DR_CONFIG``.
    rng : random.Random
        Externally supplied random source, keeping every call reproducible. The
        caller seeds with (seed, instance_id, call_idx) so that different
        rollouts of the same question land in different difficulty tiers.
    display_topk : int
        Target number of results rendered to the model (its requested
        max_results).

    Returns
    -------
    (display, metrics)
        display: the perturbed result list actually rendered to the model, with
                 text reordered, filtered, and rewritten.
        metrics: monitoring information for this perturbation (tier, gold
                 action, number of injected negatives), used for the mechanism 4
                 statistics such as the share of watch calls hitting gold frames.
    """
    cfg = _resolve_config(config)
    pool = list(results)

    if not pool:
        # Empty candidate pool: nothing to perturb, return as-is so the trajectory survives
        return [], {"applied": False, "reason": "empty_pool"}

    # ---- 1. Sample the difficulty tier ----
    tier_name = _weighted_choice(rng, cfg["tier_probs"])
    tier = cfg["tiers"][tier_name]

    # ---- 2. Structural noise: occasional empty result (simulates retrieval
    #         failure, forcing query rewriting or rerouting) ----
    if rng.random() < float(tier.get("p_empty", 0.0)):
        return [], {"applied": True, "tier": tier_name, "empty_injected": True}

    # ---- 3. Structural noise: randomized display count K ----
    lo, hi = tier.get("k_jitter", [0, 0])
    k = display_topk + rng.randint(int(lo), int(hi))
    k = max(1, min(k, len(pool)))

    # ---- 4. Split the candidate pool by gold membership (each side keeps its
    #         RRF relative order) ----
    gold_hits = [r for r in pool if _url_of(r) in gold_urls]
    non_gold = [r for r in pool if _url_of(r) not in gold_urls]

    # ---- 5. Decide the gold action (one action applies to every gold hit) ----
    gold_to_place: list[tuple[int, dict[str, Any]]] = []  # (target rank, result dict)
    gold_dropped = False
    gold_action = "none"
    if gold_hits:
        gold_action = _weighted_choice(rng, tier.get("gold_actions", {"keep": 1.0}))
        if gold_action == "drop":
            # Remove gold from the display window entirely: no gold is reachable
            # this round, forcing multi-round recovery
            gold_dropped = True
        else:
            # keep -> top; sink_mid -> middle; sink_bottom -> bottom
            base_rank = {
                "keep": 0,
                "sink_mid": max(1, k // 2),
                "sink_bottom": max(0, k - 1),
            }.get(gold_action, 0)
            for j, g in enumerate(gold_hits):
                # Place multiple gold hits consecutively, with +-1 of noise so the
                # position is not perfectly fixed
                rank = base_rank + j + rng.randint(-1, 1)
                gold_to_place.append((max(0, rank), g))

    # ---- 6. Hard-negative injection: promote real corpus videos from beyond the
    #         first screen of the candidate pool ----
    n_lo, n_hi = tier.get("n_hard_neg", [0, 0])
    n_hard = rng.randint(int(n_lo), int(n_hi))
    head_non_gold = non_gold[:k]            # non-gold that would be displayed anyway (first screen)
    tail_non_gold = non_gold[k:]            # real hard negatives beyond it (topically near-duplicate)

    fillers = list(head_non_gold)
    injected_hard_neg = 0
    if n_hard > 0 and tail_non_gold:
        n_hard = min(n_hard, len(tail_non_gold))
        chosen = rng.sample(tail_non_gold, n_hard)
        injected_hard_neg = len(chosen)
        # Insert hard negatives at random positions in the first half and push the
        # same number of weak entries out the back, keeping length ~k
        for neg in chosen:
            pos = rng.randint(0, max(0, min(len(fillers), k // 2)))
            fillers.insert(pos, neg)
        fillers = fillers[:k]

    # ---- 7. Assemble the display list: fillers first, then gold at its target rank ----
    display = fillers[:k]
    for target_rank, g in sorted(gold_to_place, key=lambda x: x[0]):
        pos = max(0, min(target_rank, len(display)))
        display.insert(pos, g)
    display = _dedup_by_url(display)[:k]

    # ---- 8. Presentation randomization: rewrite title/snippet text only, never
    #         url, duration, or thumbnail ----
    out: list[dict[str, Any]] = []
    p_title = float(tier.get("p_title_jitter", 0.0))
    p_snippet = float(tier.get("p_snippet_jitter", 0.0))
    for r in display:
        r = dict(r)  # shallow copy so the server's returned object is not mutated
        if p_title > 0 and rng.random() < p_title:
            r["title"] = _mangle_title(r.get("title", ""), rng)
        if p_snippet > 0 and rng.random() < p_snippet:
            r["snippet"] = _mangle_snippet(r.get("snippet", "") or r.get("description", ""), rng)
        out.append(r)

    # ---- 9. Collect monitoring metrics ----
    gold_ranks = [i for i, r in enumerate(out) if _url_of(r) in gold_urls]
    metrics = {
        "applied": True,
        "tier": tier_name,
        "k": k,
        "gold_action": gold_action,
        "gold_dropped": gold_dropped,            # whether gold left the display window this round
        "gold_display_ranks": gold_ranks,        # final position of gold in the display list
        "injected_hard_neg": injected_hard_neg,  # number of real hard negatives injected
        "num_displayed": len(out),
    }
    return out, metrics
