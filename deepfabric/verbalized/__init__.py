"""Verbalized sampling: ask the model for a distribution, sample off the mode.

Mode collapse makes an aligned model return the single most stereotypical
output for a fixed prompt. Verbalized sampling is a training-free mitigation:
the model verbalises ``k`` candidates with probabilities and the sampler selects
away from the mode, with near-duplicates rejected.

This package is the shared block (spec §4) reused by every seam that opts in via
its own ``diversity.verbalized`` config. It is self-contained: no dependency on
the generation pipeline.
"""

from .config import DedupConfig, DiversityConfig, VerbalizedConfig
from .sampler import SamplingResult, SelectedCandidate, VerbalizedSampler, default_text_of
from .schemas import CandidateDistribution, VerbalizedCandidate, distribution_model, to_candidates
from .selection import Ranking, normalize_probabilities, rank_candidates, select_candidates
from .similarity import (
    EmbeddingSimilarity,
    LexicalSimilarity,
    SimilarityMeasure,
    build_similarity,
    distinct_n,
    mean_pairwise_similarity,
    rouge_l,
    semantic_diversity,
)

__all__ = [
    "CandidateDistribution",
    "DedupConfig",
    "DiversityConfig",
    "EmbeddingSimilarity",
    "LexicalSimilarity",
    "Ranking",
    "SamplingResult",
    "SelectedCandidate",
    "SimilarityMeasure",
    "VerbalizedCandidate",
    "VerbalizedConfig",
    "VerbalizedSampler",
    "build_similarity",
    "default_text_of",
    "distinct_n",
    "distribution_model",
    "mean_pairwise_similarity",
    "normalize_probabilities",
    "rank_candidates",
    "rouge_l",
    "select_candidates",
    "semantic_diversity",
    "to_candidates",
]
