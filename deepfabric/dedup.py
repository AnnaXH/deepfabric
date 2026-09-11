"""Exact (hash-based) deduplication of topic nodes.

Every node is expanded independently, with no cross-node awareness, so aligned
models routinely re-derive the same subtopic in different branches ("Best
practices", "Common pitfalls"). Without deduplication each of those collapses
survives as its own node, wasting generation budget and narrowing real topical
coverage below what the node count suggests.

This module owns the pieces shared by :class:`~deepfabric.tree.Tree` and
:class:`~deepfabric.graph.Graph`:

- ``NormaliseConfig`` / ``normalise_topic`` - the normalisation ladder applied
  before hashing, from safest (``casefold``) to most aggressive
  (``strip_stopwords``, off by default).
- ``DedupConfig`` - the ``topics.dedup`` configuration surface.
- ``DedupStats`` - the counters surfaced through progress events.

Detection is exact only: two topics are duplicates when their normalised forms
hash identically. Semantic (near-duplicate) detection is deliberately not
implemented - it would add an embedding dependency and a threshold to tune.
"""

import hashlib
import unicodedata

from typing import Literal

from pydantic import BaseModel, Field

from .constants import (
    DEDUP_DEFAULT_BACKFILL_MAX_ATTEMPTS,
    DEDUP_MAX_BACKFILL_ATTEMPTS,
    DEDUP_STOPWORDS,
)

OnDuplicate = Literal["merge_as_connection", "drop", "keep"]
TopicMode = Literal["tree", "graph"]

# A repeated subtopic means different things per mode: a graph can express a
# shared node, a tree cannot.
MODE_DEFAULT_ON_DUPLICATE: dict[str, OnDuplicate] = {
    "graph": "merge_as_connection",
    "tree": "drop",
}


class NormaliseConfig(BaseModel):
    """Normalisation applied to a topic before it is hashed.

    Flags are independent and ordered safest to most aggressive. Only
    ``strip_stopwords`` can plausibly merge genuinely distinct topics, so it is
    the only one that defaults off.
    """

    casefold: bool = Field(default=True, description='Lowercase ("REST API" == "rest api")')
    whitespace: bool = Field(default=True, description="Trim and collapse internal whitespace runs")
    strip_punctuation: bool = Field(
        default=True, description='Drop punctuation ("REST-API auth." == "rest api auth")'
    )
    strip_stopwords: bool = Field(
        default=False,
        description=(
            "AGGRESSIVE and language-specific: drop filler words so "
            '"auth for APIs" == "auth of APIs". Can merge distinct topics.'
        ),
    )


class BackfillConfig(BaseModel):
    """Re-request the shortfall left behind by collapsed duplicates.

    Off by default: without a more diverse candidate generator, re-asking the
    same model tends to re-roll the same collapsed subtopics.
    """

    enabled: bool = Field(default=False, description="Re-request subtopics lost to deduplication")
    max_attempts: int = Field(
        default=DEDUP_DEFAULT_BACKFILL_MAX_ATTEMPTS,
        ge=0,
        le=DEDUP_MAX_BACKFILL_ATTEMPTS,
        description="Maximum extra generation rounds per node (0 disables; guarantees termination)",
    )


class DedupConfig(BaseModel):
    """Configuration for topic-node deduplication (``topics.dedup``)."""

    exact: bool = Field(
        default=True,
        description="Collapse exact (post-normalisation) duplicate topics. False = legacy output.",
    )
    normalise: NormaliseConfig = Field(
        default_factory=NormaliseConfig, description="Normalisation applied before hashing"
    )
    on_duplicate: OnDuplicate | None = Field(
        default=None,
        description=(
            "What to do with a detected duplicate: merge_as_connection (graph only), "
            "drop, or keep (detect and count only). None resolves to the per-mode default."
        ),
    )
    backfill: BackfillConfig = Field(
        default_factory=BackfillConfig, description="Optional regeneration of collapsed subtopics"
    )

    def resolve_for_mode(self, mode: TopicMode) -> "DedupConfig":
        """Return a copy with ``on_duplicate`` resolved to a concrete action.

        ``None`` means "per-mode default": ``merge_as_connection`` for graphs,
        ``drop`` for trees. Resolving at load time keeps the effective value
        explicit in logs and events.

        Args:
            mode: The topic mode the config is being used for.

        Returns:
            A copy of this config with ``on_duplicate`` set.

        Raises:
            ValueError: If ``merge_as_connection`` is requested for tree mode,
                which cannot express a shared node.
        """
        action = self.on_duplicate or MODE_DEFAULT_ON_DUPLICATE[mode]
        if mode == "tree" and action == "merge_as_connection":
            raise ValueError(
                "dedup.on_duplicate='merge_as_connection' is only available in graph mode; "
                "trees cannot express a shared node. Use 'drop' or 'keep'."
            )
        return self.model_copy(update={"on_duplicate": action})


class DedupStats(BaseModel):
    """Counters describing what deduplication did during a build."""

    duplicates_detected: int = Field(
        default=0, description="Candidate subtopics whose hash already existed"
    )
    duplicates_collapsed: int = Field(
        default=0, description="Duplicates that did not become a new node"
    )
    connections_added_via_merge: int = Field(
        default=0, description="Edges to an existing node created in place of a duplicate (graph)"
    )
    backfill_requests: int = Field(
        default=0, description="Extra generation rounds spent replacing collapsed duplicates"
    )
    expansions: int = Field(default=0, description="Node expansions that produced subtopics")
    subtopics_added: int = Field(default=0, description="Subtopics that became new nodes/paths")

    @property
    def effective_degree(self) -> float:
        """Average number of subtopics actually kept per expansion."""
        if self.expansions == 0:
            return 0.0
        return round(self.subtopics_added / self.expansions, 3)

    def as_event(self) -> dict:
        """Serialise the counters for inclusion in a progress event."""
        return {
            **self.model_dump(),
            "effective_degree": self.effective_degree,
        }


def _strip_punctuation(text: str) -> str:
    """Replace Unicode punctuation with spaces, leaving symbols (C++, C#) intact."""
    return "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)


def _strip_stopwords(text: str) -> str:
    """Drop stopword tokens, unless that would empty the topic entirely."""
    tokens = text.split()
    kept = [token for token in tokens if token.casefold() not in DEDUP_STOPWORDS]
    return " ".join(kept or tokens)


def normalise_topic(topic: str, config: NormaliseConfig | None = None) -> str:
    """Apply the configured normalisation ladder to a topic string.

    Args:
        topic: Raw topic text as returned by the model.
        config: Normalisation flags; defaults are used when omitted.

    Returns:
        The normalised topic. Two topics are exact duplicates when their
        normalised forms are equal.
    """
    config = config or NormaliseConfig()
    text = topic

    if config.casefold:
        text = text.casefold()
    if config.strip_punctuation:
        text = _strip_punctuation(text)
    if config.strip_stopwords:
        text = _strip_stopwords(text)
    if config.whitespace:
        text = " ".join(text.split())

    return text


def hash_topic(topic: str, config: NormaliseConfig | None = None) -> str:
    """Return the SHA256 hex digest of a topic's normalised form."""
    return hashlib.sha256(normalise_topic(topic, config).encode("utf-8")).hexdigest()


def coerce_dedup_config(value: "DedupConfig | dict | None") -> DedupConfig:
    """Accept a DedupConfig, a plain dict (as parsed from YAML), or None."""
    if value is None:
        return DedupConfig()
    if isinstance(value, DedupConfig):
        return value
    return DedupConfig.model_validate(value)
