"""Configuration models for verbalized sampling (spec §4.1).

These models describe the reusable ``verbalized`` config block. The block is
never a top-level key; a stage attaches an instance under its own
``diversity`` section, e.g. ``generation.diversity.verbalized``.

The models live inside this package (rather than in ``deepfabric/config.py``)
so the package stays self-contained and importable without pulling in the rest
of the configuration system.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_PAIRWISE_COSINE = 0.85
DEFAULT_MAX_LEXICAL_SIMILARITY = 0.8

# Upper bound for the flattening temperature reached at tune_diversity == 1.0.
MAX_TAIL_TEMPERATURE = 5.0


class DedupConfig(BaseModel):
    """Operational (in-loop) near-duplicate rejection settings.

    Warning:
        The method, embedding model and thresholds here are provisional (spec
        §0 / §4.4). They sit behind the ``SimilarityMeasure`` interface in
        ``deepfabric.verbalized.similarity`` and are expected to be replaced.
    """

    enabled: bool = Field(default=True, description="Reject near-duplicate candidates")
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        description="Embedding model used for semantic similarity (provisional)",
    )
    embedding_provider: Literal["openai"] = Field(
        default="openai",
        description="Provider supplying embeddings. Falls back to a lexical measure when unavailable.",
    )
    max_pairwise_cosine: float = Field(
        default=DEFAULT_MAX_PAIRWISE_COSINE,
        ge=0.0,
        le=1.0,
        description="Reject a candidate whose max cosine to any kept candidate exceeds this (provisional)",
    )
    max_lexical_similarity: float = Field(
        default=DEFAULT_MAX_LEXICAL_SIMILARITY,
        ge=0.0,
        le=1.0,
        description=(
            "Threshold used by the lexical fallback measure (ROUGE-L F1). Separate from "
            "max_pairwise_cosine because the two scores are not on the same scale."
        ),
    )


class VerbalizedConfig(BaseModel):
    """Reusable verbalized-sampling block attached per stage.

    Example:
        ```yaml
        generation:
          diversity:
            verbalized:
              enabled: true
              k: 5
              select: 3
        ```
    """

    enabled: bool = Field(default=False, description="Master switch for this seam")
    k: int = Field(default=5, ge=1, le=50, description="Candidates the model verbalises")
    select: int = Field(default=3, ge=1, description="How many candidates to keep (<= k)")
    selection: Literal["tail_weighted", "threshold", "top_p_complement"] = Field(
        default="tail_weighted", description="Selection strategy (see selection.py)"
    )
    prob_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Keep candidates at or below this reported probability (selection='threshold')",
    )
    top_p: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Probability core dropped by selection='top_p_complement'",
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Flattening temperature over verbalised probabilities (>1 flattens toward uniform)",
    )
    tune_diversity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional 0..1 diversity dial; higher blends selection weights toward the tail",
    )
    expansion: Literal["single_pass", "two_pass"] = Field(
        default="single_pass",
        description=(
            "How selected candidates are materialised. 'single_pass': the distribution call "
            "returns full payloads. 'two_pass' is not yet supported."
        ),
    )
    sizing: Literal["budget", "multiplier"] = Field(
        default="budget",
        description=(
            "How selected candidates map to dataset size at the generation seam. "
            "'budget' (default): hold the existing target total, using select purely as an "
            "over-generate-then-choose factor (one sample committed per node-slot). "
            "'multiplier': commit every selected candidate, so totals grow with select."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Token budget for the distribution call. Defaults to the stage's max_tokens scaled "
            "by k (capped), since one call must now carry k full candidates."
        ),
    )
    keep_probabilities: bool = Field(
        default=True,
        description="Record the candidate's approach tag and verbalised probability in sample metadata",
    )
    seed: int | None = Field(
        default=None,
        description="Optional RNG seed making tail-weighted selection reproducible",
    )
    dedup: DedupConfig = Field(default_factory=DedupConfig, description="Near-duplicate rejection")

    @model_validator(mode="after")
    def validate_configuration(self) -> "VerbalizedConfig":
        """Validate internal consistency of the block."""
        if self.select > self.k:
            raise ValueError(
                f"verbalized.select ({self.select}) must be <= verbalized.k ({self.k})"
            )
        if self.enabled and self.expansion == "two_pass":
            raise ValueError(
                "verbalized.expansion='two_pass' is not yet supported; use 'single_pass'"
            )
        return self

    @property
    def effective_temperature(self) -> float:
        """Flattening temperature after applying ``tune_diversity``."""
        if self.tune_diversity is None:
            return self.temperature
        return 1.0 + self.tune_diversity * (MAX_TAIL_TEMPERATURE - 1.0)

    @property
    def inversion_weight(self) -> float:
        """How far selection weights are blended toward an inverted distribution."""
        return self.tune_diversity or 0.0


class DiversityConfig(BaseModel):
    """Per-stage ``diversity`` section holding the verbalized block."""

    verbalized: VerbalizedConfig = Field(
        default_factory=VerbalizedConfig, description="Verbalized sampling configuration"
    )
