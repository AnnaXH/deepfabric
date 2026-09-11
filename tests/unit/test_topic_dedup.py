"""Tests for exact (hash-based) topic-node deduplication."""

import asyncio

from unittest.mock import AsyncMock, patch

import pytest  # type: ignore

from pydantic import ValidationError

from deepfabric.config import TopicsConfig
from deepfabric.dedup import (
    BackfillConfig,
    DedupConfig,
    NormaliseConfig,
    hash_topic,
    normalise_topic,
)
from deepfabric.exceptions import TreeError
from deepfabric.graph import Graph
from deepfabric.schemas import GraphSubtopic, GraphSubtopics
from deepfabric.tree import Tree

GRAPH_PARAMS = {
    "topic_prompt": "Machine Learning",
    "provider": "openai",
    "model_name": "test-model",
    "degree": 3,
    "depth": 2,
}

TREE_PARAMS = {
    "topic_prompt": "Machine Learning",
    "provider": "openai",
    "model_name": "test-model",
    "degree": 3,
    "depth": 2,
}


def make_graph(**overrides) -> Graph:
    with patch("deepfabric.graph.LLMClient"):
        return Graph(**{**GRAPH_PARAMS, **overrides})


def make_tree(**overrides) -> Tree:
    with patch("deepfabric.tree.LLMClient"):
        return Tree(**{**TREE_PARAMS, **overrides})


def subtopics(*topics: str, connections: list[int] | None = None) -> GraphSubtopics:
    return GraphSubtopics(
        subtopics=[GraphSubtopic(topic=t, connections=connections or []) for t in topics]
    )


class TestNormalisation:
    """Tests for the normalisation ladder applied before hashing."""

    def test_casefold(self):
        assert normalise_topic("REST API") == normalise_topic("rest api")

    def test_whitespace_is_trimmed_and_collapsed(self):
        assert normalise_topic("  REST   API auth ") == "rest api auth"

    def test_punctuation_is_stripped(self):
        assert normalise_topic("REST-API auth.") == normalise_topic("rest api auth")

    def test_symbols_are_preserved(self):
        """C++ must not collapse into C: '+' is a symbol, not punctuation."""
        assert normalise_topic("C++") != normalise_topic("C")

    def test_stopwords_kept_by_default(self):
        assert normalise_topic("auth for APIs") != normalise_topic("auth of APIs")

    def test_stopwords_stripped_when_enabled(self):
        config = NormaliseConfig(strip_stopwords=True)
        assert normalise_topic("auth for APIs", config) == normalise_topic("auth of APIs", config)

    def test_stopword_only_topic_survives(self):
        """Stripping must never reduce a topic to the empty string."""
        config = NormaliseConfig(strip_stopwords=True)
        assert normalise_topic("The Of", config) == "the of"

    def test_flags_can_be_disabled(self):
        config = NormaliseConfig(casefold=False, strip_punctuation=False)
        assert normalise_topic("REST-API", config) == "REST-API"

    def test_hash_matches_normalised_form(self):
        assert hash_topic("REST-API  Auth.") == hash_topic("rest api auth")
        assert len(hash_topic("anything")) == 64  # noqa: PLR2004


class TestDedupConfig:
    """Tests for config defaults and per-mode resolution."""

    def test_defaults(self):
        config = DedupConfig()
        assert config.exact is True
        assert config.on_duplicate is None
        assert config.normalise.casefold is True
        assert config.normalise.whitespace is True
        assert config.normalise.strip_punctuation is True
        assert config.normalise.strip_stopwords is False
        assert config.backfill.enabled is False
        assert config.backfill.max_attempts == 2  # noqa: PLR2004

    def test_graph_default_is_merge(self):
        assert DedupConfig().resolve_for_mode("graph").on_duplicate == "merge_as_connection"

    def test_tree_default_is_drop(self):
        assert DedupConfig().resolve_for_mode("tree").on_duplicate == "drop"

    def test_explicit_action_is_preserved(self):
        config = DedupConfig(on_duplicate="keep")
        assert config.resolve_for_mode("graph").on_duplicate == "keep"
        assert config.resolve_for_mode("tree").on_duplicate == "keep"

    def test_merge_is_rejected_for_trees(self):
        with pytest.raises(ValueError, match="only available in graph mode"):
            DedupConfig(on_duplicate="merge_as_connection").resolve_for_mode("tree")

    def test_negative_max_attempts_rejected(self):
        with pytest.raises(ValidationError):
            BackfillConfig(max_attempts=-1)

    def test_topics_config_resolves_per_mode(self):
        graph_config = TopicsConfig(prompt="Test", mode="graph")
        tree_config = TopicsConfig(prompt="Test", mode="tree")
        assert graph_config.dedup.on_duplicate == "merge_as_connection"
        assert tree_config.dedup.on_duplicate == "drop"

    def test_topics_config_rejects_merge_for_trees(self):
        with pytest.raises(ValidationError, match="only available in graph mode"):
            TopicsConfig(prompt="Test", mode="tree", dedup={"on_duplicate": "merge_as_connection"})

    def test_topics_config_parses_nested_block(self):
        config = TopicsConfig(
            prompt="Test",
            mode="graph",
            dedup={
                "exact": False,
                "normalise": {"strip_stopwords": True},
                "backfill": {"enabled": True, "max_attempts": 1},
            },
        )
        assert config.dedup.exact is False
        assert config.dedup.normalise.strip_stopwords is True
        assert config.dedup.backfill.max_attempts == 1


class TestGraphDedup:
    """Graph mode: a duplicate becomes an edge to the existing node."""

    def test_duplicate_within_one_response_is_collapsed(self):
        graph = make_graph()
        result = graph._process_subtopics_response(
            subtopics("Neural Networks", "neural networks."), graph.root
        )
        assert result.subtopics_added == 1
        assert result.duplicates_collapsed == 1
        assert len(graph.nodes) == 2  # noqa: PLR2004

    def test_duplicate_across_branches_merges_as_connection(self):
        graph = make_graph()
        graph._process_subtopics_response(subtopics("Supervised", "Unsupervised"), graph.root)
        supervised, unsupervised = graph.nodes[1], graph.nodes[2]

        result = graph._process_subtopics_response(subtopics("Regression"), supervised)
        regression = graph.nodes[3]
        assert result.subtopics_added == 1

        # The same topic re-derived under the other branch becomes a shared node
        result = graph._process_subtopics_response(subtopics("Regression"), unsupervised)
        assert result.subtopics_added == 0
        assert result.duplicates_collapsed == 1
        assert result.connections_added_via_merge == 1
        assert len(graph.nodes) == 4  # noqa: PLR2004
        assert regression in unsupervised.children
        assert unsupervised in regression.parents
        assert not graph.has_cycle()

    def test_merge_that_would_create_a_cycle_falls_back_to_drop(self):
        graph = make_graph()
        graph._process_subtopics_response(subtopics("Supervised"), graph.root)
        supervised = graph.nodes[1]
        graph._process_subtopics_response(subtopics("Regression"), supervised)
        regression = graph.nodes[2]

        # Re-deriving an ancestor's topic would close a cycle regression -> supervised
        result = graph._process_subtopics_response(subtopics("Supervised"), regression)
        assert result.duplicates_collapsed == 1
        assert result.connections_added_via_merge == 0
        assert regression.children == []
        assert not graph.has_cycle()

    def test_duplicate_of_root_is_not_self_linked(self):
        graph = make_graph()
        result = graph._process_subtopics_response(subtopics("Machine Learning"), graph.root)
        assert result.subtopics_added == 0
        assert result.connections_added_via_merge == 0
        assert graph.root.children == []
        assert not graph.has_cycle()

    def test_on_duplicate_drop_discards_the_relationship(self):
        graph = make_graph(dedup={"on_duplicate": "drop"})
        graph._process_subtopics_response(subtopics("Supervised", "Unsupervised"), graph.root)
        unsupervised = graph.nodes[2]
        graph._process_subtopics_response(subtopics("Regression"), graph.nodes[1])

        result = graph._process_subtopics_response(subtopics("Regression"), unsupervised)
        assert result.duplicates_collapsed == 1
        assert result.connections_added_via_merge == 0
        assert unsupervised.children == []

    def test_on_duplicate_keep_counts_without_changing_output(self):
        graph = make_graph(dedup={"on_duplicate": "keep"})
        graph._process_subtopics_response(subtopics("Supervised"), graph.root)
        result = graph._process_subtopics_response(subtopics("Supervised"), graph.root)

        assert result.subtopics_added == 1
        assert result.duplicates_collapsed == 0
        assert graph.dedup_stats.duplicates_detected == 1
        assert len(graph.nodes) == 3  # noqa: PLR2004

    def test_exact_false_reproduces_legacy_behaviour(self):
        graph = make_graph(dedup={"exact": False})
        graph._process_subtopics_response(subtopics("Supervised", "supervised"), graph.root)
        result = graph._process_subtopics_response(subtopics("Supervised"), graph.root)

        assert result.subtopics_added == 1
        assert result.duplicates_collapsed == 0
        assert len(graph.nodes) == 4  # noqa: PLR2004

    def test_lookup_is_index_based(self):
        """Detection is a dict hit, not a rescan of every node."""
        graph = make_graph()
        for index in range(200):
            graph.add_node(f"Topic {index}")

        assert graph.dedup_lookup("topic 7.") == 8  # noqa: PLR2004
        assert graph.dedup_lookup("Not present") is None
        assert len(graph._topic_hash_index) == 201  # noqa: PLR2004

    def test_index_is_maintained_on_removal(self):
        graph = make_graph()
        node = graph.add_node("Supervised")
        graph.add_edge(graph.root.id, node.id)
        graph.remove_node(node.id)

        assert graph.dedup_lookup("Supervised") is None

    def test_index_is_rebuilt_on_load(self, tmp_path):
        graph = make_graph()
        node = graph.add_node("Supervised")
        graph.add_edge(graph.root.id, node.id)
        save_path = str(tmp_path / "graph.json")
        graph.save(save_path)

        with patch("deepfabric.graph.LLMClient"):
            loaded = Graph.from_json(save_path, GRAPH_PARAMS.copy())
        assert loaded.dedup_lookup("supervised") == node.id

    def test_backfill_terminates_under_max_attempts(self):
        graph = make_graph(dedup={"backfill": {"enabled": True, "max_attempts": 2}})
        graph._generate_subtopics_with_retry = AsyncMock(return_value=subtopics("Same", "Same"))

        result = asyncio.run(graph.get_subtopics_and_connections(graph.root, 3))

        # One initial round plus max_attempts backfill rounds, then it gives up
        assert graph._generate_subtopics_with_retry.await_count == 3  # noqa: PLR2004
        assert result.subtopics_added == 1
        assert graph.dedup_stats.backfill_requests == 2  # noqa: PLR2004

    def test_backfill_stops_once_the_shortfall_is_met(self):
        graph = make_graph(dedup={"backfill": {"enabled": True, "max_attempts": 2}})
        graph._generate_subtopics_with_retry = AsyncMock(
            side_effect=[subtopics("A", "A"), subtopics("B", "C")]
        )

        result = asyncio.run(graph.get_subtopics_and_connections(graph.root, 3))

        assert graph._generate_subtopics_with_retry.await_count == 2  # noqa: PLR2004
        assert result.subtopics_added == 3  # noqa: PLR2004

    def test_backfill_disabled_accepts_the_reduced_degree(self):
        graph = make_graph()
        graph._generate_subtopics_with_retry = AsyncMock(return_value=subtopics("Same", "Same"))

        result = asyncio.run(graph.get_subtopics_and_connections(graph.root, 3))

        assert graph._generate_subtopics_with_retry.await_count == 1
        assert result.subtopics_added == 1
        assert graph.dedup_stats.effective_degree == 1.0

    def test_get_unique_topics_collapses_duplicates_as_a_backstop(self):
        graph = make_graph(dedup={"exact": False})
        graph._process_subtopics_response(subtopics("Supervised", "supervised"), graph.root)
        assert len(graph.get_unique_topics()) == 2  # noqa: PLR2004

        graph.dedup.exact = True
        assert len(graph.get_unique_topics()) == 1


class TestTreeDedup:
    """Tree mode: duplicates are dropped, since a tree cannot share a node."""

    def test_duplicates_within_one_response_are_dropped(self):
        tree = make_tree()
        kept, collapsed = tree._dedupe_subtopics(
            ["Neural Networks", "neural networks.", "Decision Trees"], ["Machine Learning"]
        )
        assert kept == ["Neural Networks", "Decision Trees"]
        assert collapsed == 1

    def test_duplicates_across_parents_are_dropped(self):
        tree = make_tree()
        tree._dedupe_subtopics(["Regression"], ["Machine Learning", "Supervised"])
        kept, collapsed = tree._dedupe_subtopics(
            ["Regression", "Clustering"], ["Machine Learning", "Unsupervised"]
        )
        assert kept == ["Clustering"]
        assert collapsed == 1

    def test_duplicate_of_the_root_prompt_is_dropped(self):
        tree = make_tree()
        kept, collapsed = tree._dedupe_subtopics(["Machine Learning"], ["Machine Learning"])
        assert kept == []
        assert collapsed == 1

    def test_on_duplicate_keep_counts_without_dropping(self):
        tree = make_tree(dedup={"on_duplicate": "keep"})
        tree._dedupe_subtopics(["Regression"], ["Machine Learning"])
        kept, collapsed = tree._dedupe_subtopics(["Regression"], ["Machine Learning"])
        assert kept == ["Regression"]
        assert collapsed == 0
        assert tree.dedup_stats.duplicates_detected == 1

    def test_exact_false_reproduces_legacy_behaviour(self):
        tree = make_tree(dedup={"exact": False})
        tree._dedupe_subtopics(["Regression"], ["Machine Learning"])
        kept, collapsed = tree._dedupe_subtopics(["Regression"], ["Machine Learning"])
        assert kept == ["Regression"]
        assert collapsed == 0

    def test_merge_as_connection_is_rejected(self):
        with pytest.raises(TreeError, match="only available in graph mode"):
            make_tree(dedup={"on_duplicate": "merge_as_connection"})

    def test_paths_no_longer_contain_collapsed_duplicates(self):
        tree = make_tree(degree=2, depth=2)
        tree.get_subtopics = AsyncMock(return_value=["Supervised", "supervised"])

        asyncio.run(_drain(tree.build_async()))

        leaf_topics = [path[-1] for path in tree.tree_paths]
        assert leaf_topics == ["Supervised"]
        assert tree.dedup_stats.duplicates_collapsed > 0

    def test_backfill_terminates_under_max_attempts(self):
        tree = make_tree(dedup={"backfill": {"enabled": True, "max_attempts": 2}})
        tree.get_subtopics = AsyncMock(return_value=["Same", "Same"])

        _, kept, _ = asyncio.run(tree._get_deduped_subtopics("", ["Machine Learning"], 3))

        assert tree.get_subtopics.await_count == 3  # noqa: PLR2004
        assert kept == ["Same"]
        assert tree.dedup_stats.backfill_requests == 2  # noqa: PLR2004

    def test_backfill_stops_once_the_shortfall_is_met(self):
        tree = make_tree(dedup={"backfill": {"enabled": True, "max_attempts": 2}})
        tree.get_subtopics = AsyncMock(side_effect=[["A", "A"], ["B", "C"]])

        _, kept, _ = asyncio.run(tree._get_deduped_subtopics("", ["Machine Learning"], 3))

        assert tree.get_subtopics.await_count == 2  # noqa: PLR2004
        assert kept == ["A", "B", "C"]

    def test_index_is_rebuilt_from_loaded_paths(self):
        tree = make_tree()
        tree.from_dict_list([{"path": ["Machine Learning", "Supervised", "Regression"]}])
        assert tree.dedup_lookup("regression.") is not None


class TestDedupStats:
    """Counters surfaced through progress events."""

    def test_effective_degree_reports_the_reduced_branching(self):
        graph = make_graph()
        graph.dedup_stats.expansions = 4
        graph.dedup_stats.subtopics_added = 6
        assert graph.dedup_stats.effective_degree == 1.5  # noqa: PLR2004

    def test_effective_degree_without_expansions(self):
        assert make_graph().dedup_stats.effective_degree == 0.0

    def test_build_complete_event_carries_the_counters(self):
        graph = make_graph(degree=2, depth=1)
        graph._generate_subtopics_with_retry = AsyncMock(return_value=subtopics("Same", "Same"))

        events = asyncio.run(_collect(graph.build_async()))
        complete = next(e for e in events if e["event"] == "build_complete")

        assert complete["duplicates_collapsed"] == 1
        assert complete["connections_added_via_merge"] == 0
        assert complete["effective_degree"] == 1.0

    def test_node_expanded_event_carries_the_counters(self):
        graph = make_graph(degree=2, depth=1)
        graph._generate_subtopics_with_retry = AsyncMock(return_value=subtopics("Same", "Same"))

        events = asyncio.run(_collect(graph.build_async()))
        expanded = next(e for e in events if e["event"] == "node_expanded")

        assert expanded["subtopics_added"] == 1
        assert expanded["duplicates_collapsed"] == 1


async def _collect(async_iterable) -> list[dict]:
    return [event async for event in async_iterable]


async def _drain(async_iterable) -> None:
    async for _ in async_iterable:
        pass
