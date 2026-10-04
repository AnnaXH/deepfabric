"""Regression coverage for canonical graph ancestry lookup."""

from unittest.mock import patch

import pytest

from deepfabric.graph import Graph
from deepfabric.topic_model import TopicPath
from deepfabric.tree import Tree


@pytest.fixture
def graph():
    with patch("deepfabric.graph.LLMClient"):
        return Graph(
            topic_prompt="Root", provider="openai", model_name="test-model", degree=2, depth=3
        )


def test_interior_leaf_root_and_unknown_paths(graph):
    branch = graph.add_node("Branch")
    interior = graph.add_node("Subbranch")
    leaf = graph.add_node("Leaf")
    graph.add_edge(graph.root.id, branch.id)
    graph.add_edge(branch.id, interior.id)
    graph.add_edge(interior.id, leaf.id)

    assert graph.get_path_by_id(branch.metadata["uuid"]) == ["Root", "Branch"]
    assert graph.get_path_by_id(interior.metadata["uuid"]) == ["Root", "Branch", "Subbranch"]
    assert graph.get_path_by_id(leaf.metadata["uuid"]) == ["Root", "Branch", "Subbranch", "Leaf"]
    assert graph.get_path_by_id(graph.root.metadata["uuid"]) == ["Root"]
    assert graph.get_path_by_id("unknown") is None
    assert [topic.uuid for topic in graph.get_unique_topics()] == [
        branch.metadata["uuid"],
        interior.metadata["uuid"],
        leaf.metadata["uuid"],
    ]


def test_first_parent_wins_even_when_dfs_visits_other_parent_first(graph):
    first_branch = graph.add_node("First DFS branch")
    primary = graph.add_node("Primary parent")
    shared = graph.add_node("Shared")
    graph.add_edge(graph.root.id, first_branch.id)
    graph.add_edge(graph.root.id, primary.id)
    graph.add_edge(primary.id, shared.id)
    graph.add_edge(first_branch.id, shared.id)

    assert shared.parents == [primary, first_branch]
    assert graph.get_path_by_id(shared.metadata["uuid"]) == ["Root", "Primary parent", "Shared"]


def test_leaf_path_enumeration_is_unchanged_by_node_lookups(graph):
    branch_a = graph.add_node("A")
    branch_b = graph.add_node("B")
    interior = graph.add_node("Interior")
    shared_leaf = graph.add_node("Shared leaf")
    other_leaf = graph.add_node("Other leaf")
    graph.add_edge(graph.root.id, branch_a.id)
    graph.add_edge(graph.root.id, branch_b.id)
    graph.add_edge(branch_a.id, interior.id)
    graph.add_edge(interior.id, shared_leaf.id)
    graph.add_edge(branch_b.id, shared_leaf.id)
    graph.add_edge(branch_b.id, other_leaf.id)
    expected = [
        TopicPath(["Root", "A", "Interior", "Shared leaf"], shared_leaf.metadata["uuid"]),
        TopicPath(["Root", "B", "Shared leaf"], shared_leaf.metadata["uuid"]),
        TopicPath(["Root", "B", "Other leaf"], other_leaf.metadata["uuid"]),
    ]
    assert graph.get_all_paths_with_ids() == expected
    assert graph.get_all_paths() == [path.path for path in expected]
    for topic in graph.get_unique_topics():
        assert graph.get_path_by_id(topic.uuid) is not None
    assert graph.get_all_paths_with_ids() == expected
    assert graph.get_all_paths() == [path.path for path in expected]


def test_parent_cycles_return_accumulated_path(graph):
    node_a = graph.add_node("A")
    node_b = graph.add_node("B")
    graph.add_edge(graph.root.id, node_a.id)
    graph.add_edge(node_a.id, node_b.id)
    graph.add_edge(node_b.id, graph.root.id)

    assert graph.get_path_by_id(node_b.metadata["uuid"]) == ["Root", "A", "B"]
    assert graph.get_path_by_id(node_a.metadata["uuid"]) == ["B", "Root", "A"]

    isolated = graph.add_node("Self cycle")
    graph.add_edge(isolated.id, isolated.id)
    assert graph.get_path_by_id(isolated.metadata["uuid"]) == ["Self cycle"]


def test_uuid_index_is_invalidated_for_add_remove_subtree_and_prune(graph):
    assert graph.find_node_by_uuid(graph.root.metadata["uuid"]) is graph.root
    branch = graph.add_node("Branch")
    child = graph.add_node("Child")
    graph.add_edge(graph.root.id, branch.id)
    graph.add_edge(branch.id, child.id)
    assert graph.find_node_by_uuid(child.metadata["uuid"]) is child
    graph.remove_node(child.id)
    assert graph.find_node_by_uuid(child.metadata["uuid"]) is None

    child = graph.add_node("Replacement child")
    graph.add_edge(branch.id, child.id)
    assert graph.get_path_by_id(child.metadata["uuid"]) == ["Root", "Branch", "Replacement child"]
    graph.prune_at_level(1)
    assert graph.find_node_by_uuid(child.metadata["uuid"]) is None
    assert graph.find_node_by_uuid(branch.metadata["uuid"]) is branch
    graph.remove_subtree(branch.id)
    assert graph.find_node_by_uuid(branch.metadata["uuid"]) is None


def test_duplicate_uuid_keeps_first_match_after_removal(graph):
    first = graph.add_node("First", metadata={"uuid": "shared-uuid"})
    second = graph.add_node("Second", metadata={"uuid": "shared-uuid"})
    assert graph.find_node_by_uuid("shared-uuid") is first
    graph.remove_node(first.id)
    assert graph.find_node_by_uuid("shared-uuid") is second


def test_loaded_graph_builds_index_from_loaded_nodes(graph, tmp_path):
    branch = graph.add_node("Branch")
    leaf = graph.add_node("Leaf")
    graph.add_edge(graph.root.id, branch.id)
    graph.add_edge(branch.id, leaf.id)
    assert graph.find_node_by_uuid(branch.metadata["uuid"]) is branch
    path = tmp_path / "graph.json"
    graph.save(str(path))
    with patch("deepfabric.graph.LLMClient"):
        loaded = Graph.from_json(
            str(path),
            {"topic_prompt": "Placeholder", "provider": "openai", "model_name": "test-model"},
        )
    assert loaded.get_path_by_id(branch.metadata["uuid"]) == ["Root", "Branch"]
    assert loaded.get_all_paths_with_ids() == graph.get_all_paths_with_ids()
    assert loaded.find_node_by_uuid(loaded.root.metadata["uuid"]) is loaded.root


def test_large_graph_lookups_do_not_scan_nodes_or_enumerate_paths(graph):
    # A structural performance guard avoids machine-dependent timing thresholds.
    class NoScanNodes(dict):
        def values(self):
            raise AssertionError("Repeated lookup scanned every node")

    branches = [graph.root]
    for _ in range(4):
        next_level = []
        for parent in branches:
            for index in range(8):
                child = graph.add_node(f"{parent.id}/{index}")
                graph.add_edge(parent.id, child.id)
                next_level.append(child)
        branches = next_level
    topics = graph.get_unique_topics()
    graph.find_node_by_uuid(graph.root.metadata["uuid"])
    graph.nodes = NoScanNodes(graph.nodes)
    with patch.object(graph, "get_all_paths_with_ids", side_effect=AssertionError("DFS lookup")):
        for topic in topics:
            path = graph.get_path_by_id(topic.uuid)
            assert path[0] == "Root"
            assert path[-1] == topic.topic


def test_tree_paths_and_ids_remain_unchanged():
    with patch("deepfabric.tree.LLMClient"):
        tree = Tree(topic_prompt="Root", provider="openai", model_name="test-model")
    tree.tree_paths = [["Root", "A", "Leaf A"], ["Root", "B", "Leaf B"]]
    before = tree.get_all_paths_with_ids()
    topics = tree.get_unique_topics()
    assert [topic.uuid for topic in topics] == [path.topic_id for path in before]
    assert [tree.get_path_by_id(topic.uuid) for topic in topics] == tree.tree_paths
    assert tree.get_all_paths_with_ids() == before
    assert tree.get_all_paths() == tree.tree_paths
