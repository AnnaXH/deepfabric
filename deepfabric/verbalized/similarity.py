"""Candidate similarity and dedup for verbalized sampling (spec §4.4).

Warning:
    **The metric here is provisional.** The *structural role* is stable - the
    sampler needs some operational similarity function to reject near-duplicate
    candidates - but the method behind the interface is under review and is
    expected to be swapped. Everything metric-specific is kept behind
    :class:`SimilarityMeasure` (``similarity`` / ``is_duplicate``) so nothing in
    ``sampler.py`` has to change when it is.

Two implementations ship today, both described by the source paper:

- :class:`EmbeddingSimilarity` - cosine similarity of embeddings
  (``text-embedding-3-small`` by default). Semantic diversity is then
  ``1 - mean pairwise cosine``, negatives clipped to 0.
- :class:`LexicalSimilarity` - ROUGE-L F1, used as a graceful fallback when no
  embedding provider is configured (a warning, never a hard failure).

There are two distinct uses of similarity, deliberately kept separate:

- **Operational (in-loop):** :meth:`SimilarityMeasure.is_duplicate`, cheap and
  pairwise, run during candidate selection.
- **Evaluation (offline):** :func:`semantic_diversity` and :func:`distinct_n`,
  used to report whether a phase actually raised diversity.
"""

import asyncio
import logging
import math
import os
import re

from abc import ABC, abstractmethod
from collections.abc import Sequence

import openai

from .config import DedupConfig

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"\w+")

# A diversity measure needs at least a pair of texts.
_MIN_TEXTS_FOR_PAIRWISE = 2

# Cache of measures keyed by serialised config, so the embedding cache (and the
# HTTP client) survive across the per-sample builders that request one.
_MEASURE_CACHE: dict[str, "SimilarityMeasure"] = {}


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


def rouge_l(a: str, b: str) -> float:
    """ROUGE-L F1 between two texts.

    Args:
        a: First text.
        b: Second text.

    Returns:
        F1 score in [0, 1] over the longest common subsequence of word tokens.
    """
    tokens_a, tokens_b = tokenize(a), tokenize(b)
    if not tokens_a or not tokens_b:
        return 1.0 if not tokens_a and not tokens_b else 0.0

    # Rolling-row LCS: O(len(a) * len(b)) time, O(len(b)) space.
    previous = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        current = [0]
        for j, token_b in enumerate(tokens_b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current

    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(tokens_a)
    recall = lcs / len(tokens_b)
    return 2 * precision * recall / (precision + recall)


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Distinct-N: the ratio of unique n-grams to total n-grams.

    Args:
        texts: Texts to measure.
        n: N-gram size.

    Returns:
        Ratio in [0, 1]; 0.0 when there are no n-grams.
    """
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        tokens = tokenize(text)
        for i in range(len(tokens) - n + 1):
            unique.add(tuple(tokens[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(unique) / total


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors, clipped to [0, 1]."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return min(1.0, max(0.0, dot / (norm_a * norm_b)))


class SimilarityMeasure(ABC):
    """Narrow interface the sampler depends on.

    Implementations may be swapped freely; ``sampler.py`` only ever calls
    ``prepare``, ``similarity`` and ``is_duplicate``.

    Attributes:
        threshold: Similarity above which two candidates count as duplicates.
    """

    name: str = "base"

    def __init__(self, threshold: float):
        self.threshold = threshold

    async def prepare(self, texts: Sequence[str]) -> None:  # noqa: ARG002, B027
        """Optional hook to pre-compute state for a candidate set.

        Args:
            texts: All texts that will be compared in this round.
        """

    @abstractmethod
    async def similarity(self, a: str, b: str) -> float:
        """Return similarity between two texts in [0, 1]."""

    async def is_duplicate(self, candidate: str, kept: Sequence[str]) -> bool:
        """Whether a candidate is a near-duplicate of anything already kept.

        Args:
            candidate: Text of the candidate under consideration.
            kept: Texts of candidates already accepted.

        Returns:
            True when max similarity to any kept candidate exceeds the threshold.
        """
        for existing in kept:
            if await self.similarity(candidate, existing) > self.threshold:
                return True
        return False


class LexicalSimilarity(SimilarityMeasure):
    """ROUGE-L F1 similarity; the no-embedding-provider fallback."""

    name = "lexical_rouge_l"

    async def similarity(self, a: str, b: str) -> float:
        """Return ROUGE-L F1 between two texts."""
        return rouge_l(a, b)


class EmbeddingSimilarity(SimilarityMeasure):
    """Cosine similarity of embeddings, with a lexical safety net.

    Embeddings are cached per instance, so a candidate set is embedded once.
    If the embedding call fails (no credentials, transient API error), the
    measure degrades to :class:`LexicalSimilarity` with a warning rather than
    failing the generation run.
    """

    name = "embedding_cosine"

    def __init__(
        self,
        threshold: float,
        model: str,
        client: object | None = None,
        fallback: SimilarityMeasure | None = None,
    ):
        super().__init__(threshold)
        self.model = model
        self._client = client
        self._fallback = fallback or LexicalSimilarity(threshold)
        self._cache: dict[str, list[float]] = {}
        self._degraded = False
        self._lock = asyncio.Lock()

    def _get_client(self):
        """Return the async embedding client, creating it on first use."""
        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    async def _embed(self, texts: Sequence[str]) -> None:
        """Embed and cache any texts not already cached."""
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if not missing or self._degraded:
            return
        async with self._lock:
            missing = [text for text in missing if text not in self._cache]
            if not missing:
                return
            try:
                response = await self._get_client().embeddings.create(
                    model=self.model, input=list(missing)
                )
            except Exception as e:  # noqa: BLE001 - any failure degrades, never fails the run
                self._degraded = True
                logger.warning(
                    "Embedding similarity unavailable (%s); falling back to %s",
                    e,
                    self._fallback.name,
                )
                return
            for text, item in zip(missing, response.data, strict=False):
                self._cache[text] = list(item.embedding)

    async def prepare(self, texts: Sequence[str]) -> None:
        """Embed a whole candidate set in one call."""
        await self._embed(texts)

    async def similarity(self, a: str, b: str) -> float:
        """Return cosine similarity of the two texts' embeddings."""
        await self._embed([a, b])
        if a not in self._cache or b not in self._cache:
            return await self._fallback.similarity(a, b)
        return cosine_similarity(self._cache[a], self._cache[b])


def build_similarity(config: DedupConfig) -> SimilarityMeasure:
    """Build the similarity measure described by a dedup config.

    Falls back to the lexical measure with a warning when no embedding provider
    is available, rather than failing.

    Args:
        config: Dedup configuration.

    Returns:
        A cached :class:`SimilarityMeasure` instance.
    """
    cache_key = config.model_dump_json()
    cached = _MEASURE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if config.embedding_provider == "openai" and os.getenv("OPENAI_API_KEY"):
        measure: SimilarityMeasure = EmbeddingSimilarity(
            threshold=config.max_pairwise_cosine,
            model=config.embedding_model,
            fallback=LexicalSimilarity(config.max_lexical_similarity),
        )
    else:
        logger.warning(
            "No embedding provider configured for verbalized dedup; "
            "falling back to lexical similarity (ROUGE-L)"
        )
        measure = LexicalSimilarity(config.max_lexical_similarity)

    _MEASURE_CACHE[cache_key] = measure
    return measure


async def mean_pairwise_similarity(texts: Sequence[str], measure: SimilarityMeasure) -> float:
    """Mean similarity over all unordered pairs of texts.

    Args:
        texts: Texts to compare (fewer than two yields 0.0).
        measure: Similarity measure to use.

    Returns:
        Mean pairwise similarity in [0, 1].
    """
    if len(texts) < _MIN_TEXTS_FOR_PAIRWISE:
        return 0.0
    await measure.prepare(texts)
    scores = [
        await measure.similarity(texts[i], texts[j])
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]
    return sum(scores) / len(scores)


async def semantic_diversity(texts: Sequence[str], measure: SimilarityMeasure) -> float:
    """Diversity of a set of texts: ``1 - mean pairwise similarity``.

    With :class:`EmbeddingSimilarity` this is the paper's semantic diversity
    (1 - mean pairwise cosine, negatives clipped to 0); with the lexical
    fallback it is the ROUGE-L analogue.

    Args:
        texts: Texts to measure (fewer than two yields 0.0).
        measure: Similarity measure to use.

    Returns:
        Diversity in [0, 1].
    """
    if len(texts) < _MIN_TEXTS_FOR_PAIRWISE:
        return 0.0
    return max(0.0, 1.0 - await mean_pairwise_similarity(texts, measure))
