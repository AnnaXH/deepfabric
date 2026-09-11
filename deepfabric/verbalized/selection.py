"""Tail-biased selection strategies for verbalized sampling (spec §4.3).

Never take top-1: the point of verbalized sampling is to move off the mode. A
strategy produces a *ranking* over the verbalised candidates plus the number of
leading entries that actually satisfy the strategy (``eligible``). The sampler
walks the ranking, skipping near-duplicates, until it has kept ``select``
candidates or run out of eligible ones - so dedup has backfill headroom without
escaping the strategy's intent.

Every strategy keeps at least one candidate: when nothing qualifies, the
median-probability candidate is used so a seam never yields zero samples.
"""

import random

from dataclasses import dataclass

from .config import VerbalizedConfig
from .schemas import VerbalizedCandidate

# Floor applied to selection weights so every candidate stays reachable.
_WEIGHT_EPSILON = 1e-9


@dataclass
class Ranking:
    """A strategy's preference order over candidates.

    Attributes:
        ordered: All candidates, most-preferred first.
        eligible: How many leading entries satisfy the strategy. The sampler
            never keeps a candidate beyond this point.
    """

    ordered: list[VerbalizedCandidate]
    eligible: int


def normalize_probabilities(candidates: list[VerbalizedCandidate]) -> list[float]:
    """Normalise verbalised probabilities to sum to 1.

    Args:
        candidates: Candidates carrying (possibly uncalibrated) probabilities.

    Returns:
        Normalised probabilities; a uniform distribution when the reported
        values carry no signal (all zero or empty).
    """
    if not candidates:
        return []
    raw = [max(0.0, float(c.probability)) for c in candidates]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [value / total for value in raw]


def _median_index(probabilities: list[float]) -> int:
    """Index of the median-probability candidate (the always-safe fallback)."""
    order = sorted(range(len(probabilities)), key=lambda i: probabilities[i])
    return order[len(order) // 2]


def _weighted_order(
    candidates: list[VerbalizedCandidate], weights: list[float], rng: random.Random
) -> list[VerbalizedCandidate]:
    """Order candidates by weighted sampling without replacement.

    Uses the Efraimidis-Spirakis key method: drawing ``key = u ** (1 / w)`` and
    sorting descending is equivalent to repeated weighted draws without
    replacement, and yields a full ranking in one pass.
    """
    keyed = []
    for candidate, weight in zip(candidates, weights, strict=True):
        safe_weight = max(weight, _WEIGHT_EPSILON)
        key = rng.random() ** (1.0 / safe_weight)
        keyed.append((key, candidate))
    keyed.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in keyed]


def _tail_weights(probabilities: list[float], config: VerbalizedConfig) -> list[float]:
    """Build tail-biased selection weights from normalised probabilities.

    ``temperature`` flattens the distribution (``p ** (1/T)``, ``T > 1`` moves
    toward uniform) and ``tune_diversity`` blends that flattened distribution
    toward its inversion (``1 - p``), so 0 keeps the model's own ordering and 1
    prefers the tail outright.
    """
    temperature = config.effective_temperature
    flattened = [p ** (1.0 / temperature) for p in probabilities]
    flat_total = sum(flattened) or 1.0
    flattened = [value / flat_total for value in flattened]

    lam = config.inversion_weight
    if lam <= 0:
        return flattened

    inverted = [max(0.0, 1.0 - p) for p in probabilities]
    inv_total = sum(inverted)
    inverted = (
        [value / inv_total for value in inverted]
        if inv_total > 0
        else [1.0 / len(probabilities)] * len(probabilities)
    )
    return [(1.0 - lam) * f + lam * i for f, i in zip(flattened, inverted, strict=True)]


def _rank_tail_weighted(
    candidates: list[VerbalizedCandidate], config: VerbalizedConfig, rng: random.Random
) -> Ranking:
    """Sample the full ordering with tail-biased weights."""
    probabilities = normalize_probabilities(candidates)
    weights = _tail_weights(probabilities, config)
    return Ranking(ordered=_weighted_order(candidates, weights, rng), eligible=len(candidates))


def _rank_threshold(candidates: list[VerbalizedCandidate], config: VerbalizedConfig) -> Ranking:
    """Keep candidates at or below ``prob_threshold``, rarest first."""
    probabilities = normalize_probabilities(candidates)
    indices = sorted(range(len(candidates)), key=lambda i: candidates[i].probability)
    qualifying = [i for i in indices if candidates[i].probability <= config.prob_threshold]

    if qualifying:
        rest = [i for i in indices if i not in set(qualifying)]
        ordered = [candidates[i] for i in qualifying + rest]
        return Ranking(ordered=ordered, eligible=len(qualifying))

    # Nothing qualifies - fall back to the median-probability candidate.
    median = _median_index(probabilities)
    rest = [i for i in indices if i != median]
    return Ranking(ordered=[candidates[median]] + [candidates[i] for i in rest], eligible=1)


def _rank_top_p_complement(
    candidates: list[VerbalizedCandidate], config: VerbalizedConfig
) -> Ranking:
    """Drop the smallest top set covering the probability core, keep the rest.

    The complement is ordered by descending probability: the most plausible
    candidate that is still outside the mode comes first.
    """
    probabilities = normalize_probabilities(candidates)
    by_desc = sorted(range(len(candidates)), key=lambda i: probabilities[i], reverse=True)

    core: list[int] = []
    cumulative = 0.0
    for i in by_desc:
        if cumulative >= config.top_p:
            break
        core.append(i)
        cumulative += probabilities[i]

    complement = [i for i in by_desc if i not in set(core)]
    if complement:
        ordered = [candidates[i] for i in complement] + [candidates[i] for i in core]
        return Ranking(ordered=ordered, eligible=len(complement))

    # The core covers everything - fall back to the median-probability candidate.
    median = _median_index(probabilities)
    rest = [i for i in by_desc if i != median]
    return Ranking(ordered=[candidates[median]] + [candidates[i] for i in rest], eligible=1)


def rank_candidates(
    candidates: list[VerbalizedCandidate],
    config: VerbalizedConfig,
    rng: random.Random | None = None,
) -> Ranking:
    """Rank candidates according to the configured selection strategy.

    Args:
        candidates: Verbalised candidates to rank.
        config: Verbalized sampling configuration.
        rng: Optional RNG for reproducible tail-weighted sampling.

    Returns:
        A :class:`Ranking` over all candidates.

    Raises:
        ValueError: If the selection strategy is unknown.
    """
    if not candidates:
        return Ranking(ordered=[], eligible=0)

    if config.selection == "tail_weighted":
        # Bandit: candidate selection, not a security function
        default_rng = random.Random(config.seed)  # noqa: S311  # nosec
        return _rank_tail_weighted(candidates, config, rng or default_rng)
    if config.selection == "threshold":
        return _rank_threshold(candidates, config)
    if config.selection == "top_p_complement":
        return _rank_top_p_complement(candidates, config)

    msg = f"Unknown selection strategy: {config.selection}"
    raise ValueError(msg)


def select_candidates(
    candidates: list[VerbalizedCandidate],
    config: VerbalizedConfig,
    rng: random.Random | None = None,
) -> list[VerbalizedCandidate]:
    """Select up to ``select`` candidates, tail-biased, without dedup.

    Args:
        candidates: Verbalised candidates to select from.
        config: Verbalized sampling configuration.
        rng: Optional RNG for reproducible tail-weighted sampling.

    Returns:
        The selected candidates (at least one, unless there were none).
    """
    ranking = rank_candidates(candidates, config, rng)
    return ranking.ordered[: min(config.select, max(ranking.eligible, 1))]
