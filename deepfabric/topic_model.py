from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, NamedTuple

from .dedup import (
    DedupConfig,
    DedupStats,
    OnDuplicate,
    TopicMode,
    coerce_dedup_config,
    hash_topic,
)


class TopicPath(NamedTuple):
    """A topic path with its associated unique identifier."""

    path: list[str]
    topic_id: str


class Topic(NamedTuple):
    """A unique topic with its UUID and content.

    Used for generation where we iterate over unique topics (by UUID)
    rather than paths. This deduplicated view is essential for graphs
    where multiple paths can lead to the same topic node.
    """

    uuid: str
    topic: str  # The topic text/content


class TopicModel(ABC):
    """Abstract base class for topic models like Tree and Graph.

    Besides the traversal contract, this base owns exact deduplication: the
    ``normalised hash -> node id`` index, the O(1) lookup used at insertion
    time, and the ``on_duplicate`` dispatch. Tree and Graph share one
    implementation rather than each rolling their own.
    """

    #: Mode used to resolve per-mode dedup defaults; graphs override this.
    dedup_mode: TopicMode = "tree"

    def _init_dedup(self, dedup: "DedupConfig | dict | None" = None) -> None:
        """Initialise the dedup config, hash index and counters.

        Called from ``__init__`` of each concrete model.

        Args:
            dedup: Dedup configuration (model, plain dict, or None for defaults).
        """
        self.dedup: DedupConfig = coerce_dedup_config(dedup).resolve_for_mode(self.dedup_mode)
        self.dedup_stats: DedupStats = DedupStats()
        self._topic_hash_index: dict[str, Any] = {}

    def _ensure_dedup(self) -> None:
        """Initialise dedup state lazily for models built without ``_init_dedup``."""
        if not hasattr(self, "_topic_hash_index"):
            self._init_dedup()

    @property
    def dedup_action(self) -> OnDuplicate:
        """The resolved action to take when a duplicate is detected."""
        self._ensure_dedup()
        action = self.dedup.on_duplicate
        # resolve_for_mode() guarantees this is set; the fallback is for safety.
        return action or "drop"

    def dedup_hash(self, topic: str) -> str:
        """Return the normalised hash used for duplicate detection."""
        self._ensure_dedup()
        return hash_topic(topic, self.dedup.normalise)

    def dedup_lookup(self, topic: str) -> Any | None:
        """Return the id of an existing node with the same normalised topic.

        O(1): a dict lookup against the hash index, never a rescan of all nodes.

        Args:
            topic: Candidate topic text.

        Returns:
            The id registered for that topic, or None when it is new or when
            exact dedup is disabled.
        """
        self._ensure_dedup()
        if not self.dedup.exact:
            return None
        return self._topic_hash_index.get(self.dedup_hash(topic))

    def dedup_register(self, topic: str, node_id: Any) -> str:
        """Register a topic in the hash index; the first registration wins.

        Args:
            topic: Topic text of the node being inserted.
            node_id: Identifier to return from future lookups.

        Returns:
            The normalised hash the topic was registered under.
        """
        self._ensure_dedup()
        topic_hash = self.dedup_hash(topic)
        self._topic_hash_index.setdefault(topic_hash, node_id)
        return topic_hash

    def dedup_unregister(self, topic: str, node_id: Any) -> None:
        """Drop a topic from the index when its node is removed."""
        self._ensure_dedup()
        topic_hash = self.dedup_hash(topic)
        if self._topic_hash_index.get(topic_hash) == node_id:
            del self._topic_hash_index[topic_hash]

    def dedup_rebuild_index(self, topics: Iterable[tuple[str, Any]]) -> None:
        """Rebuild the hash index from scratch (e.g. after loading from disk).

        Args:
            topics: ``(topic, node_id)`` pairs in insertion order.
        """
        self._ensure_dedup()
        self._topic_hash_index = {}
        for topic, node_id in topics:
            self.dedup_register(topic, node_id)

    @abstractmethod
    async def build_async(self) -> None:
        """Asynchronously build the topic model."""
        raise NotImplementedError

    def build(self) -> None:  # pragma: no cover - legacy compatibility
        """Deprecated synchronous entry point kept for legacy compatibility."""
        msg = "TopicModel.build() is no longer supported. Use build_async() instead."
        raise RuntimeError(msg)

    @abstractmethod
    def get_all_paths(self) -> list[list[str]]:
        """Returns all the paths in the topic model."""
        raise NotImplementedError

    @abstractmethod
    def get_all_paths_with_ids(self) -> list[TopicPath]:
        """Returns all paths with their unique identifiers.

        Returns:
            List of TopicPath namedtuples containing (path, topic_id).
            The topic_id is a stable identifier for the leaf node of each path.
        """
        raise NotImplementedError

    @abstractmethod
    def get_unique_topics(self) -> list[Topic]:
        """Returns deduplicated topics by UUID.

        For generation, we iterate over unique topics rather than paths.
        This is important for graphs where multiple paths can lead to the
        same topic node - we only want to generate one sample per unique topic.

        Returns:
            List of Topic namedtuples containing (uuid, topic).
            Each UUID appears exactly once.
        """
        raise NotImplementedError

    def get_path_by_id(self, topic_id: str) -> list[str] | None:
        """Look up a path by its topic_id.

        Args:
            topic_id: The unique identifier for a topic path.

        Returns:
            The path list if found, None otherwise.
        """
        for topic_path in self.get_all_paths_with_ids():
            if topic_path.topic_id == topic_id:
                return topic_path.path
        return None
