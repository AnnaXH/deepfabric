import asyncio
import hashlib
import json
import time
import warnings

from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    DEFAULT_MAX_TOKENS,
    MAX_RETRY_ATTEMPTS,
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_MODEL,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .dedup import DedupConfig
from .exceptions import TreeError
from .llm import LLMClient
from .metrics import trace
from .prompts import TreePromptBuilder
from .schemas import TopicList
from .stream_simulator import simulate_stream
from .topic_model import Topic, TopicModel, TopicPath

warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings:.*")

if TYPE_CHECKING:  # only for type hints to avoid runtime cycles
    from .progress import ProgressReporter

UPPER_DEGREE = 50
UPPER_DEPTH = 10


class ValidationResult(TypedDict, total=False):
    valid: bool
    total_tree_paths: int
    total_requested_paths: int
    recommended_batch_size: int


class TreeConfig(BaseModel):
    """Configuration for constructing a topic tree."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic tree"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    degree: int = Field(
        default=TOPIC_TREE_DEFAULT_DEGREE,
        ge=1,
        le=UPPER_DEGREE,
        description="The branching factor of the tree",
    )
    depth: int = Field(
        default=TOPIC_TREE_DEFAULT_DEPTH,
        ge=1,
        le=UPPER_DEPTH,
        description="The depth of the tree",
    )
    provider: str = Field(
        default="ollama",
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model_name: str = Field(
        default=TOPIC_TREE_DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=TOPIC_TREE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )
    max_concurrent: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Maximum concurrent LLM calls during tree expansion (helps avoid rate limits)",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        description="Maximum tokens for topic generation LLM calls",
    )
    dedup: DedupConfig = Field(
        default_factory=DedupConfig,
        description="Exact duplicate-topic detection and handling",
    )


class TreeValidator:
    """TreeValidator validates and calculates unique paths in a tree structure."""

    def __init__(self, degree: int, depth: int):
        self.degree = degree
        self.depth = depth

    def calculate_paths(self) -> int:
        """Calculate total number of paths in the tree."""
        return self.degree**self.depth

    def validate_configuration(self, num_steps: int, batch_size: int) -> ValidationResult:
        """
        Validates the tree configuration and provides recommendations if invalid.

        Args:
            num_steps: Number of steps requested.
            batch_size: Batch size per step.

        Returns:
            A ValidationResult dict containing validity, totals, and recommendations.
        """
        total_requested_paths = num_steps * batch_size
        total_tree_paths = self.calculate_paths()

        print(f"Total tree paths available: {total_tree_paths}")
        print(f"Total requested paths: {total_requested_paths}")

        result: ValidationResult = {
            "valid": total_requested_paths <= total_tree_paths,
            "total_tree_paths": total_tree_paths,
            "total_requested_paths": total_requested_paths,
        }

        if not result["valid"]:
            print(
                "Requested paths (%d) exceed available tree paths (%d).",
                total_requested_paths,
                total_tree_paths,
            )
            result["recommended_batch_size"] = min(batch_size, total_tree_paths)

        return result


class Tree(TopicModel):
    """A class to represent and build a hierarchical topic tree."""

    def __init__(self, **kwargs):
        """Initialize the Tree with the given parameters."""
        try:
            self.config = TreeConfig.model_validate(kwargs)
        except Exception as e:
            raise TreeError(f"Invalid tree configuration: {str(e)}") from e  # noqa: TRY003

        # Initialize from config
        self.topic_prompt = self.config.topic_prompt
        self.model_system_prompt = self.config.topic_system_prompt
        self.degree = self.config.degree
        self.depth = self.config.depth
        self.temperature = self.config.temperature
        self.provider = self.config.provider
        self.model_name = self.config.model_name
        self.max_concurrent = self.config.max_concurrent
        self.max_tokens = self.config.max_tokens

        # Exact-duplicate detection: hash index, counters and resolved on_duplicate
        try:
            self._init_dedup(self.config.dedup)
        except ValueError as e:
            raise TreeError(str(e)) from e
        self.dedup_register(self.topic_prompt, self._path_to_id([self.topic_prompt]))

        # Initialize LLM client
        llm_kwargs = {}
        if self.config.base_url:
            llm_kwargs["base_url"] = self.config.base_url

        self.llm_client = LLMClient(
            provider=self.provider,
            model_name=self.model_name,
            **llm_kwargs,
        )

        # Progress reporter for streaming feedback (set by topic_manager)
        self.progress_reporter: ProgressReporter | None = None

        trace(
            "tree_created",
            {
                "provider": self.provider,
                "model_name": self.model_name,
                "degree": self.degree,
                "depth": self.depth,
            },
        )

        # Derived attributes
        self.system_prompt = self.config.topic_system_prompt
        self.tree_paths: list[list[str]] = []
        self.failed_generations: list[dict[str, Any]] = []

    async def build_async(self, model_name: str | None = None):
        """Build the complete topic tree.

        Args:
            model_name: Optional model name to override the configured model

        Yields:
            dict: Progress events with event type and associated data
        """
        if model_name:
            self.model_name = model_name

        yield {
            "event": "build_start",
            "model_name": self.model_name,
            "depth": self.config.depth,
            "degree": self.config.degree,
        }

        def _raise_if_build_failed():
            """Check if build failed completely and raise appropriate error."""
            if len(self.tree_paths) == 0 and self.failed_generations:
                error_msg = f"Tree build failed completely: all {len(self.failed_generations)} generation attempts failed. No topic paths created."
                raise RuntimeError(error_msg)

        try:
            async for event in self._build_subtree_generator(
                [self.config.topic_prompt],
                self.config.topic_system_prompt,
                self.config.depth,
                self.config.degree,
                1,
            ):
                yield event

            # Check if build was completely unsuccessful (no paths generated)
            _raise_if_build_failed()

            trace(
                "tree_built",
                {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "total_paths": len(self.tree_paths),
                    "failed_generations": len(self.failed_generations),
                    "success": len(self.tree_paths) > 0,
                    "duplicates_collapsed": self.dedup_stats.duplicates_collapsed,
                    "effective_degree": self.dedup_stats.effective_degree,
                },
            )

            yield {
                "event": "build_complete",
                "total_paths": len(self.tree_paths),
                "failed_generations": len(self.failed_generations),
                **self.dedup_stats.as_event(),
            }
        except Exception as e:
            yield {"event": "error", "error": str(e)}
            # Save partial results before re-raising
            if self.tree_paths:
                self.save("partial_tree.jsonl")
            raise

    def get_all_paths(self) -> list[list[str]]:
        """Returns all the paths in the topic model."""
        return self.tree_paths

    @staticmethod
    def _path_to_id(path: list[str]) -> str:
        """Compute a deterministic topic ID from a tree path."""
        return hashlib.sha256(json.dumps(path).encode()).hexdigest()[:16]

    def _add_path(self, path: list[str]) -> None:
        """Add a path to the tree.

        Args:
            path: The topic path to add.
        """
        self.tree_paths.append(path)

    def get_all_paths_with_ids(self) -> list[TopicPath]:
        """Returns all paths with their unique identifiers.

        Returns:
            List of TopicPath namedtuples containing (path, topic_id).
            The topic_id is computed deterministically from the path content.
        """
        return [TopicPath(path=path, topic_id=self._path_to_id(path)) for path in self.tree_paths]

    def get_unique_topics(self) -> list[Topic]:
        """Returns all leaf topics with computed IDs.

        Paths are distinct by construction (each is a different route through
        the tree), so this collapses nothing on its own — it is a one-Topic-per-
        path view, not a deduplication step. Duplicate topic strings are removed
        at insertion time by the hash index instead.

        Returns:
            List of Topic namedtuples containing (uuid, topic).
        """
        return [
            Topic(uuid=self._path_to_id(path), topic=path[-1] if path else "")
            for path in self.tree_paths
        ]

    async def get_subtopics(
        self, system_prompt: str, node_path: list[str], num_subtopics: int
    ) -> list[str]:
        """Generate subtopics using structured generation."""

        # Determine domain based on system prompt or path content
        domain = self._detect_domain(system_prompt, node_path)

        prompt = TreePromptBuilder.build_expansion_prompt(
            topic_path=node_path,
            num_subtopics=num_subtopics,
            system_prompt=system_prompt if system_prompt else "",
            domain=domain,
        )

        try:
            # Always use non-streaming for reliable structured output
            topic_response = await self.llm_client.generate_async(
                prompt=prompt,
                schema=TopicList,
                max_retries=MAX_RETRY_ATTEMPTS,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # Fire-and-forget: simulate streaming for TUI preview (non-blocking)
            simulate_stream(
                self.progress_reporter,
                topic_response.model_dump_json(indent=2),
                source="tree_generation",
            )

            # Extract subtopics — accept whatever the LLM returned
            subtopics = topic_response.subtopics
            return subtopics[:num_subtopics]

        except Exception as e:
            self.failed_generations.append(
                {
                    "node_path": node_path,
                    "error": str(e),
                    "timestamp": time.time(),
                }
            )
            return []

    def _dedupe_subtopics(
        self, subtopics: list[str], node_path: list[str]
    ) -> tuple[list[str], int]:
        """Drop subtopics whose normalised form already exists in the tree.

        A tree cannot express a shared node, so a duplicate is dropped rather
        than merged. Kept subtopics are registered immediately, which also
        collapses duplicates repeated inside a single response.

        Args:
            subtopics: Subtopic strings as returned by the model.
            node_path: Path of the parent node being expanded.

        Returns:
            A tuple of (subtopics to expand, number collapsed).
        """
        kept: list[str] = []
        collapsed = 0

        for subtopic in subtopics:
            if self.dedup_lookup(subtopic) is not None:
                self.dedup_stats.duplicates_detected += 1
                if self.dedup_action != "keep":
                    collapsed += 1
                    continue
                # keep: detection only - the duplicate is still expanded
            self.dedup_register(subtopic, self._path_to_id([*node_path, subtopic]))
            kept.append(subtopic)

        return kept, collapsed

    async def _get_deduped_subtopics(
        self, system_prompt: str, node_path: list[str], n_child: int
    ) -> tuple[list[str], list[str], int]:
        """Generate subtopics for a node and remove exact duplicates.

        Collapsing duplicates leaves the node with fewer than ``n_child``
        children. That reduced degree is accepted and reported unless backfill
        is enabled, in which case only the shortfall is re-requested, capped by
        ``dedup.backfill.max_attempts`` so expansion always terminates.

        Args:
            system_prompt: System prompt for topic generation.
            node_path: Path of the parent node being expanded.
            n_child: Requested number of children.

        Returns:
            A tuple of (raw subtopics from the first call, subtopics to expand,
            number collapsed as duplicates).
        """
        self.dedup_stats.expansions += 1
        raw = await self.get_subtopics(system_prompt, node_path, n_child)
        kept, collapsed = self._dedupe_subtopics(raw, node_path)

        backfill = self.dedup.backfill
        if self.dedup.exact and backfill.enabled and raw:
            for _ in range(backfill.max_attempts):
                shortfall = n_child - len(kept)
                if shortfall <= 0:
                    break
                self.dedup_stats.backfill_requests += 1
                extra = await self.get_subtopics(system_prompt, node_path, shortfall)
                if not extra:
                    break
                more_kept, more_collapsed = self._dedupe_subtopics(extra, node_path)
                kept.extend(more_kept)
                collapsed += more_collapsed

        self.dedup_stats.duplicates_collapsed += collapsed
        self.dedup_stats.subtopics_added += len(kept)
        return raw, kept, collapsed

    def _detect_domain(self, system_prompt: str, node_path: list[str]) -> str:
        """Detect the appropriate domain for prompt examples based on context."""
        combined_text = f"{system_prompt} {' '.join(node_path)}".lower()

        if any(
            word in combined_text
            for word in ["math", "calculus", "algebra", "geometry", "equation"]
        ):
            return "educational"
        if any(
            word in combined_text
            for word in ["programming", "code", "software", "python", "algorithm"]
        ):
            return "technical"
        if any(
            word in combined_text
            for word in ["chat", "conversation", "talk", "friendly", "assistant"]
        ):
            return "conversational"
        return "general"

    async def _build_subtree_generator(
        self,
        node_path: list[str],
        system_prompt: str,
        total_depth: int,
        n_child: int,
        current_depth: int,
    ):
        """Recursively build a subtree of topics, yielding progress events.

        Args:
            node_path: Current path in the tree
            system_prompt: System prompt for topic generation
            total_depth: Maximum depth of the tree
            n_child: Number of child nodes per parent
            current_depth: Current depth in the tree

        Yields:
            dict: Progress events
        """
        yield {"event": "subtree_start", "node_path": node_path, "depth": current_depth}

        if current_depth > total_depth:
            self._add_path(node_path)
            yield {"event": "leaf_reached", "path": node_path}
            return

        generated, subtopics, duplicates_collapsed = await self._get_deduped_subtopics(
            system_prompt, node_path, n_child
        )

        event = {
            "event": "subtopics_generated",
            "parent_path": node_path,
            "count": len(subtopics),
            # Success reflects generation; an expansion fully collapsed by
            # dedup generated fine, it just added nothing new.
            "success": bool(generated),
            "duplicates_collapsed": duplicates_collapsed,
        }

        # Include error information if generation failed
        if not event["success"] and self.failed_generations:
            # Get the most recent failure
            recent_failure = self.failed_generations[-1]
            event["error"] = recent_failure.get("error", "Unknown error")

        yield event

        if not subtopics:
            self._add_path(node_path)
            yield {"event": "leaf_reached", "path": node_path}
            return

        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _expand_child(child_subtopic: str) -> None:
            async with semaphore:
                child_path = node_path + [child_subtopic]
                async for child_event in self._build_subtree_generator(
                    child_path, system_prompt, total_depth, n_child, current_depth + 1
                ):
                    await event_queue.put(child_event)

        tasks = [asyncio.create_task(_expand_child(s)) for s in subtopics]

        async def _signal_done() -> None:
            await asyncio.gather(*tasks)
            await event_queue.put(None)

        done_task = asyncio.create_task(_signal_done())

        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield event

        await done_task

    def save(self, save_path: str) -> None:
        """Save the topic tree to a file.

        Format: {"path": [...]}
        IDs are computed on-the-fly from path content, not persisted.
        """
        from pathlib import Path  # noqa: PLC0415

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            for path in self.tree_paths:
                f.write(json.dumps({"path": path}) + "\n")

        # Save failed generations if any
        if self.failed_generations:
            failed_path = save_path.replace(".jsonl", "_failed.jsonl")
            with open(failed_path, "w") as f:
                for failed in self.failed_generations:
                    f.write(json.dumps({"failed_generation": failed}) + "\n")

    def print_tree(self) -> None:
        """Print the topic tree in a readable format."""
        print("Topic Tree Structure:")
        for path in self.tree_paths:
            print(" -> ".join(path))

    def to_dict(self) -> dict[str, Any]:
        """Convert the topic tree to a dictionary representation.

        Returns:
            dict: Dictionary containing the tree structure and metadata
        """
        return {
            "tree_paths": self.tree_paths,
            "failed_generations": self.failed_generations,
            "config": {
                "topic_prompt": self.topic_prompt,
                "degree": self.degree,
                "depth": self.depth,
                "temperature": self.temperature,
                "provider": self.provider,
                "model_name": self.model_name,
            },
        }

    def from_dict_list(self, dict_list: list[dict[str, Any]]) -> None:
        """Construct the topic tree from a list of dictionaries.

        Accepts both the current format (``{"path": [...]}``) and the
        legacy format that included a ``leaf_uuid`` field (silently ignored).

        Args:
            dict_list: The list of dictionaries representing the topic tree.
        """
        # Clear existing data
        self.tree_paths = []
        self.failed_generations = []

        for d in dict_list:
            if "path" in d:
                self.tree_paths.append(d["path"])
            elif "failed_generation" in d:
                self.failed_generations.append(d["failed_generation"])

        # Rebuild the dedup index so continued building sees the loaded topics.
        self.dedup_rebuild_index(
            (topic, self._path_to_id(path[: index + 1]))
            for path in self.tree_paths
            for index, topic in enumerate(path)
        )
