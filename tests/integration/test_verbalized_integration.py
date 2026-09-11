"""Integration tests for verbalized sampling with real API calls (spec §6.4).

These exercise the parts that need credentials: embedding-backed similarity and
end-to-end generation against a live model. They follow the repo's convention
for credentialed tests - marked and skipped unless the API key is present - so
they stay out of the default offline run (`make test-unit`).

Run with: `make test-integration-openai`.

Assertions are deliberately about **direction and non-regression** (diversity
up, validity not down), never a fixed magnitude: the numbers are model-dependent
and the similarity metric itself is provisional (spec §0).
"""

import asyncio

import pytest

from deepfabric import DataSetGenerator, Graph
from deepfabric.builders import conversation_text
from deepfabric.verbalized import (
    DedupConfig,
    EmbeddingSimilarity,
    build_similarity,
    semantic_diversity,
)

from .conftest import assert_topic_build_result, requires_openai

GENERATION_SYSTEM_PROMPT = "You are a helpful assistant that generates training data."


def conversation_texts(dataset) -> list[str]:
    """Extract comparable text from each sample of a generated dataset."""
    texts = []
    for record in dataset:
        messages = record.get("messages", [])
        texts.append(
            "\n".join(m.get("content") or "" for m in messages if m.get("role") != "system")
        )
    return texts


@pytest.fixture
def shared_topic_graph(openai_config):
    """A small topic graph reused by the baseline and verbalized runs."""
    graph = Graph(
        topic_prompt="Everyday home gardening questions",
        provider=openai_config["provider"],
        model_name=openai_config["model_name"],
        temperature=openai_config["temperature"],
        degree=2,
        depth=1,
    )

    async def build_graph():
        async for _ in graph.build_async():
            pass

    asyncio.run(build_graph())
    assert_topic_build_result(
        graph.get_all_paths(), graph, min_paths=2, context="shared_topic_graph fixture"
    )
    return graph


def make_generator(openai_config, verbalized: dict | None = None) -> DataSetGenerator:
    """Build a generator with verbalized sampling off (default) or configured."""
    kwargs = {
        "provider": openai_config["provider"],
        "model_name": openai_config["model_name"],
        "generation_system_prompt": GENERATION_SYSTEM_PROMPT,
        "temperature": openai_config["temperature"],
    }
    if verbalized is not None:
        kwargs["diversity"] = {"verbalized": verbalized}
    return DataSetGenerator(**kwargs)


@requires_openai
class TestEmbeddingSimilarity:
    """The embedding-backed similarity measure against the real embedding API."""

    @pytest.mark.openai
    def test_build_similarity_uses_embeddings_when_available(self):
        measure = build_similarity(DedupConfig())
        assert isinstance(measure, EmbeddingSimilarity)

    @pytest.mark.openai
    def test_identical_text_scores_higher_than_unrelated_text(self):
        measure = build_similarity(DedupConfig())

        async def run():
            same = await measure.similarity(
                "How often should I water a fern?", "How often should I water a fern?"
            )
            different = await measure.similarity(
                "How often should I water a fern?", "Explain the Byzantine Generals problem."
            )
            return same, different

        same, different = asyncio.run(run())
        assert same > different

    @pytest.mark.openai
    def test_near_duplicates_are_rejected(self):
        measure = build_similarity(DedupConfig())

        async def run():
            kept = ["How often should I water a fern?"]
            duplicate = await measure.is_duplicate("How often should I water a fern?", kept)
            distinct = await measure.is_duplicate("What soil pH suits blueberries?", kept)
            return duplicate, distinct

        duplicate, distinct = asyncio.run(run())
        assert duplicate is True
        assert distinct is False

    @pytest.mark.openai
    def test_semantic_diversity_separates_varied_from_repetitive_sets(self):
        measure = build_similarity(DedupConfig())
        repetitive = ["How do I water a fern?"] * 4
        varied = [
            "How do I water a fern?",
            "What soil pH suits blueberries?",
            "When should I prune wisteria?",
            "Why are my tomato leaves curling?",
        ]

        async def run():
            return (
                await semantic_diversity(repetitive, measure),
                await semantic_diversity(varied, measure),
            )

        repetitive_score, varied_score = asyncio.run(run())
        assert varied_score > repetitive_score


@requires_openai
class TestVerbalizedGenerationOpenAI:
    """End-to-end generation with verbalized sampling enabled (spec §6.4)."""

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_diversity_rises_without_a_validity_regression(self, openai_config, shared_topic_graph):
        """Same topics, same seed: diversity up, schema validity not down."""
        baseline_generator = make_generator(openai_config)
        baseline = baseline_generator.create_data(
            num_steps=4, batch_size=2, topic_model=shared_topic_graph
        )

        verbalized_generator = make_generator(
            openai_config,
            {"enabled": True, "k": 5, "select": 3, "seed": 7},
        )
        verbalized = verbalized_generator.create_data(
            num_steps=4, batch_size=2, topic_model=shared_topic_graph
        )

        measure = build_similarity(DedupConfig())

        async def diversity_of(dataset):
            return await semantic_diversity(conversation_texts(dataset), measure)

        baseline_diversity = asyncio.run(diversity_of(baseline))
        verbalized_diversity = asyncio.run(diversity_of(verbalized))

        # Direction only - magnitudes are model-dependent.
        assert verbalized_diversity > baseline_diversity

        # Non-regression: no more failed samples than the baseline run.
        assert len(verbalized_generator.failed_samples) <= len(baseline_generator.failed_samples)

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_budget_mode_holds_the_dataset_size(self, openai_config, shared_topic_graph):
        """Enabling verbalized sampling must not silently inflate the dataset."""
        generator = make_generator(
            openai_config, {"enabled": True, "k": 5, "select": 3, "sizing": "budget"}
        )

        dataset = generator.create_data(num_steps=4, batch_size=2, topic_model=shared_topic_graph)

        assert len(dataset) <= 4  # noqa: PLR2004

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_candidates_are_distinct_within_a_node(self, openai_config, shared_topic_graph):
        """Multiplier mode: the samples a single node emits differ from each other."""
        generator = make_generator(
            openai_config,
            {"enabled": True, "k": 5, "select": 3, "sizing": "multiplier"},
        )

        dataset = generator.create_data(num_steps=1, batch_size=1, topic_model=shared_topic_graph)

        texts = conversation_texts(dataset)
        assert len(set(texts)) == len(texts)

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_cot_freetext_generation_stays_valid(self, openai_config, shared_topic_graph):
        """The cot/freetext reasoning style survives the distribution call."""
        generator = DataSetGenerator(
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
            generation_system_prompt=GENERATION_SYSTEM_PROMPT,
            temperature=openai_config["temperature"],
            conversation_type="cot",
            reasoning_style="freetext",
            diversity={"verbalized": {"enabled": True, "k": 4, "select": 2}},
        )

        dataset = generator.create_data(num_steps=2, batch_size=2, topic_model=shared_topic_graph)

        assert len(dataset) >= 1
        assert generator.failure_analysis["invalid_schema"] == []

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_verbalized_metadata_is_recorded(self, openai_config, shared_topic_graph):
        """Selected candidates carry their approach tag and verbalised probability."""
        generator = make_generator(openai_config, {"enabled": True, "k": 4, "select": 2})

        dataset = generator.create_data(num_steps=2, batch_size=2, topic_model=shared_topic_graph)

        recorded = [record["metadata"]["verbalized"] for record in dataset]
        assert recorded
        assert all(0.0 <= entry["probability"] <= 1.0 for entry in recorded)
        assert all(entry["approach"] for entry in recorded)


@requires_openai
class TestVerbalizedBuilderOpenAI:
    """Builder-level checks that need a live model."""

    @pytest.mark.openai
    @pytest.mark.flaky(reruns=2, reruns_delay=5)
    def test_builder_returns_distinct_conversations(self, openai_config):
        """One distribution call yields several usable, distinct conversations."""
        from deepfabric.builders import SingleShotBuilder  # noqa: PLC0415
        from deepfabric.llm import LLMClient  # noqa: PLC0415

        generator = make_generator(openai_config, {"enabled": True, "k": 5, "select": 3})
        llm = LLMClient(provider=openai_config["provider"], model_name=openai_config["model_name"])
        builder = SingleShotBuilder(llm, generator.config)

        conversations = asyncio.run(builder.generate("Caring for houseplants in winter"))

        assert isinstance(conversations, list)
        assert 1 <= len(conversations) <= 3  # noqa: PLR2004
        texts = [conversation_text(c) for c in conversations]
        assert len(set(texts)) == len(texts)
