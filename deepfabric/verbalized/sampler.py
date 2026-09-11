"""The shared verbalized-sampling block (spec §4).

Instead of asking the model for one instance, ask it to verbalise a
*distribution* of ``k`` candidates with probabilities, then sample away from the
mode. This module owns that flow and is reused by every seam:

1. one structured call returning a :func:`distribution_model` instance,
2. tail-biased selection (:mod:`.selection`),
3. near-duplicate rejection behind the swappable similarity interface
   (:mod:`.similarity`).

The sampler is payload-agnostic: a seam passes the payload schema and a
``text_of`` function that renders a payload for the similarity measure.
"""

import logging
import random

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .config import VerbalizedConfig
from .schemas import distribution_model, to_candidates
from .selection import normalize_probabilities, rank_candidates
from .similarity import SimilarityMeasure, build_similarity, semantic_diversity

logger = logging.getLogger(__name__)


@dataclass
class SelectedCandidate:
    """One candidate that survived selection and dedup.

    Attributes:
        payload: The candidate payload (an instance of the seam's schema).
        approach: The model's short tag for how this candidate differs.
        probability: The verbalised probability, as reported.
        normalized_probability: The probability after normalising the set.
        rank: 0-based position in the selection order.
    """

    payload: Any
    approach: str
    probability: float
    normalized_probability: float
    rank: int


@dataclass
class SamplingResult:
    """Outcome of one verbalized-sampling round.

    Attributes:
        selected: Surviving candidates, most-preferred first.
        generated: How many candidates the model verbalised.
        duplicates_dropped: How many candidates dedup rejected.
        diversity: Realised diversity of the selected set, or None when not measured.
        measure: Name of the similarity measure used, or None when dedup is off.
    """

    selected: list[SelectedCandidate] = field(default_factory=list)
    generated: int = 0
    duplicates_dropped: int = 0
    diversity: float | None = None
    measure: str | None = None

    @property
    def payloads(self) -> list[Any]:
        """The selected payloads, in selection order."""
        return [candidate.payload for candidate in self.selected]


def default_text_of(payload: Any) -> str:
    """Render a payload as text for the similarity measure.

    Args:
        payload: Candidate payload.

    Returns:
        A text rendering; Pydantic models are serialised to JSON.
    """
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    return str(payload)


class VerbalizedSampler:
    """Verbalises a candidate distribution and selects off-mode candidates."""

    def __init__(
        self,
        config: VerbalizedConfig,
        llm: Any,
        similarity: SimilarityMeasure | None = None,
        rng: random.Random | None = None,
    ):
        """Initialise the sampler.

        Args:
            config: Verbalized sampling configuration for this seam.
            llm: LLM client exposing ``generate_async(prompt, schema, **kwargs)``.
            similarity: Optional similarity measure; built from config when omitted.
            rng: Optional RNG for reproducible tail-weighted selection.
        """
        self.config = config
        self.llm = llm
        self._similarity = similarity
        # Bandit: candidate selection, not a security function
        self.rng = rng or random.Random(config.seed)  # noqa: S311  # nosec

    @property
    def similarity(self) -> SimilarityMeasure:
        """The similarity measure used for dedup (built lazily)."""
        if self._similarity is None:
            self._similarity = build_similarity(self.config.dedup)
        return self._similarity

    async def sample(
        self,
        prompt: str,
        payload_model: type[BaseModel],
        text_of: Callable[[Any], str] = default_text_of,
        **llm_kwargs,
    ) -> SamplingResult:
        """Verbalise a distribution and select candidates from its tail.

        Args:
            prompt: Fully built distribution prompt for this seam.
            payload_model: Schema of a single candidate's payload.
            text_of: Renders a payload as text for the similarity measure.
            **llm_kwargs: Extra arguments forwarded to the LLM client.

        Returns:
            The selection outcome, including realised diversity.

        Raises:
            ValueError: If verbalized sampling is disabled, the expansion mode is
                unsupported, or the model returned no usable candidates. The
                message is phrased as a validation error so the caller's normal
                sample-retry path applies.
        """
        if not self.config.enabled:
            msg = "Validation error: verbalized sampler invoked while disabled"
            raise ValueError(msg)
        if self.config.expansion != "single_pass":
            msg = (
                f"verbalized.expansion='{self.config.expansion}' is not yet supported; "
                "use 'single_pass'"
            )
            raise ValueError(msg)

        schema = distribution_model(payload_model)
        distribution = await self.llm.generate_async(prompt=prompt, schema=schema, **llm_kwargs)

        candidates = to_candidates(distribution)
        if not candidates:
            msg = "Validation error: verbalized sampling returned no candidates"
            raise ValueError(msg)

        probabilities = normalize_probabilities(candidates)
        normalized_by_id = {id(c): p for c, p in zip(candidates, probabilities, strict=True)}

        ranking = rank_candidates(candidates, self.config, self.rng)
        pool = ranking.ordered[: max(ranking.eligible, 1)]

        texts = [text_of(candidate.payload) for candidate in pool]
        dedup_enabled = self.config.dedup.enabled and len(pool) > 1
        if dedup_enabled:
            await self.similarity.prepare(texts)

        selected: list[SelectedCandidate] = []
        kept_texts: list[str] = []
        duplicates = 0

        for candidate, text in zip(pool, texts, strict=True):
            if len(selected) >= self.config.select:
                break
            if (
                dedup_enabled
                and kept_texts
                and await self.similarity.is_duplicate(text, kept_texts)
            ):
                duplicates += 1
                continue
            selected.append(
                SelectedCandidate(
                    payload=candidate.payload,
                    approach=candidate.approach,
                    probability=candidate.probability,
                    normalized_probability=normalized_by_id[id(candidate)],
                    rank=len(selected),
                )
            )
            kept_texts.append(text)

        if not selected:
            # Dedup collapsed the pool - keep the most-preferred candidate so a
            # seam never yields zero.
            best = pool[0]
            selected.append(
                SelectedCandidate(
                    payload=best.payload,
                    approach=best.approach,
                    probability=best.probability,
                    normalized_probability=normalized_by_id[id(best)],
                    rank=0,
                )
            )
            kept_texts.append(texts[0])

        diversity = (
            await semantic_diversity(kept_texts, self.similarity) if len(kept_texts) > 1 else None
        )

        logger.debug(
            "Verbalized sampling: %d verbalised, %d selected, %d duplicates dropped",
            len(candidates),
            len(selected),
            duplicates,
        )

        return SamplingResult(
            selected=selected,
            generated=len(candidates),
            duplicates_dropped=duplicates,
            diversity=diversity,
            measure=self.similarity.name if dedup_enabled else None,
        )
