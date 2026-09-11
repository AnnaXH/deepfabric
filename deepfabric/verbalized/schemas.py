"""Candidate-distribution schemas for verbalized sampling (spec §4.2).

The model is asked to verbalise a *distribution* of candidates rather than a
single instance. Each candidate carries a short ``approach`` tag describing how
it differs, a verbalised ``probability``, and the seam-specific ``payload``.

Two shapes are provided:

- :class:`VerbalizedCandidate` / :class:`CandidateDistribution` - the generic,
  payload-agnostic shape used internally (after parsing) and by the selection
  strategies.
- :func:`distribution_model` - builds a *typed* distribution model for a
  concrete payload schema (e.g. ``Conversation``). Structured-output providers
  need a concrete payload type, so this is what is handed to the LLM.

Probabilities are never required to sum to 1 (models rarely comply); they are
normalised downstream and treated as a ranking signal only, never as calibrated
sampling weights.
"""

from functools import cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

APPROACH_DESCRIPTION = (
    "Short tag naming how this candidate differs from the others (e.g. 'beginner framing', "
    "'edge case', 'contrarian take'). A few words, not a sentence."
)
PROBABILITY_DESCRIPTION = (
    "Your estimate of how likely this candidate is as a response to the request, between 0 and 1. "
    "Typical/stereotypical candidates score high, unusual ones score low. The values need not sum to 1."
)


def _coerce_probability(value: Any) -> Any:
    """Clamp a verbalised probability into [0, 1], tolerating loose model output.

    Probabilities are a ranking signal, not ground truth, so an out-of-range or
    percentage-shaped value is clamped rather than rejected - a whole candidate
    set should not be lost because one number came back as ``1.4``.
    """
    if isinstance(value, str):
        text = value.strip()
        is_percentage = text.endswith("%")
        try:
            value = float(text.rstrip("%"))
        except ValueError:
            return value
        if is_percentage:
            value /= 100.0
    if isinstance(value, int | float) and not isinstance(value, bool):
        return min(1.0, max(0.0, float(value)))
    return value


class _CandidateBase(BaseModel):
    """Shared candidate fields, minus the seam-specific payload."""

    model_config = ConfigDict(extra="forbid")

    approach: str = Field(description=APPROACH_DESCRIPTION)
    probability: float = Field(description=PROBABILITY_DESCRIPTION)

    _clamp_probability = field_validator("probability", mode="before")(_coerce_probability)


class VerbalizedCandidate(_CandidateBase):
    """One verbalised candidate with an untyped payload."""

    payload: Any = Field(default=None, description="Seam-specific payload")


class CandidateDistribution(BaseModel):
    """A verbalised distribution of candidates."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[VerbalizedCandidate] = Field(
        default_factory=list, description="The verbalised candidates"
    )


@cache
def distribution_model(payload_model: type[BaseModel]) -> type[BaseModel]:
    """Build a distribution schema whose payload is a concrete model.

    Structured-output providers require a concrete payload type, so the generic
    ``payload: Any`` shape cannot be sent to the LLM directly.

    Args:
        payload_model: Pydantic model describing one candidate's payload.

    Returns:
        A Pydantic model with a single ``candidates`` field.

    Example:
        >>> from deepfabric.schemas import Conversation
        >>> model = distribution_model(Conversation)
        >>> model.__name__
        'ConversationCandidateDistribution'
    """
    candidate_model = create_model(
        f"{payload_model.__name__}Candidate",
        __base__=_CandidateBase,
        payload=(payload_model, Field(description="The full candidate for this approach")),
    )
    return create_model(
        f"{payload_model.__name__}CandidateDistribution",
        __config__=None,
        __base__=BaseModel,
        candidates=(
            list[candidate_model],  # type: ignore[valid-type]
            Field(description="The verbalised candidates, one per distinct approach"),
        ),
    )


def to_candidates(distribution: Any) -> list[VerbalizedCandidate]:
    """Convert a typed distribution instance into generic candidates.

    Args:
        distribution: Instance produced by a :func:`distribution_model` schema
            (or any object exposing ``candidates``).

    Returns:
        Generic candidates preserving the payload objects as-is.
    """
    raw = getattr(distribution, "candidates", None) or []
    return [
        VerbalizedCandidate(
            approach=getattr(item, "approach", ""),
            probability=getattr(item, "probability", 0.0),
            payload=getattr(item, "payload", None),
        )
        for item in raw
    ]
