"""Offline unit tests for expression diversity at the generation seam (spec §6).

Verbalized sampling is wired into SingleShotBuilder.generate, which returns a
list of conversations when enabled; the generator applies the sizing mode. All
LLM calls are mocked and the similarity measure is forced to the offline lexical
fallback, so nothing here touches the network.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepfabric.builders import SingleShotBuilder, conversation_text
from deepfabric.generator import DataSetGenerator, DataSetGeneratorConfig
from deepfabric.prompts import VerbalizedPromptBuilder
from deepfabric.schemas import ChatMessage, Conversation
from deepfabric.topic_model import Topic, TopicPath
from deepfabric.verbalized import distribution_model
from deepfabric.verbalized.similarity import _MEASURE_CACHE

CANDIDATE_TEXTS = [
    ("How do I water a fern?", "Keep the soil damp but never soggy."),
    ("Explain photosynthesis for a chemist.", "Light drives electron transport in photosystem II."),
    ("My cactus is shrivelling, why?", "Under-watering during dormancy is the usual culprit."),
    ("Compare hydroponics with soil.", "Hydroponics trades buffering for control."),
    ("Is tap water safe for orchids?", "Chlorine dissipates; dissolved salts are the real risk."),
]


@pytest.fixture(autouse=True)
def offline_similarity(monkeypatch):
    """Force the lexical fallback so no test can reach an embedding API."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _MEASURE_CACHE.clear()
    yield
    _MEASURE_CACHE.clear()


def make_config(**overrides):
    """Build a generator config, with verbalized sampling off unless overridden."""
    base = {
        "generation_system_prompt": "You are a helpful gardening assistant.",
        "dataset_system_prompt": "You are a helpful gardening assistant.",
        "provider": "openai",
        "model_name": "test-model",
        "sys_msg": True,
    }
    base.update(overrides)
    return DataSetGeneratorConfig(**base)


def verbalized_config(**verbalized):
    """Config dict enabling verbalized sampling with the given overrides."""
    settings = {"enabled": True, "k": 5, "select": 3, "seed": 17}
    settings.update(verbalized)
    return {"diversity": {"verbalized": settings}}


def make_conversation(question, answer):
    """Build a bare user/assistant conversation."""
    return Conversation(
        messages=[
            ChatMessage(role="user", content=question),
            ChatMessage(role="assistant", content=answer),
        ]
    )


def make_distribution(count=5, probabilities=None, first_role="user"):
    """Build a verbalised distribution of `count` distinct conversations."""
    probabilities = probabilities or [0.5, 0.2, 0.15, 0.1, 0.05][:count]
    candidates = []
    for i, (question, answer) in enumerate(CANDIDATE_TEXTS[:count]):
        conversation = make_conversation(question, answer)
        if i == 0 and first_role != "user":
            conversation.messages[0].role = first_role
        candidates.append(
            {
                "approach": f"approach-{i}",
                "probability": probabilities[i],
                "payload": conversation.model_dump(),
            }
        )
    return distribution_model(Conversation).model_validate({"candidates": candidates})


def make_llm(distribution=None, single=None):
    """Mock LLM client that answers distribution and single-shot calls alike."""
    llm = MagicMock()

    async def generate_async(*_args, **kwargs):
        schema = kwargs.get("schema")
        if schema is not None and schema is distribution_model(Conversation):
            return distribution if distribution is not None else make_distribution()
        return single if single is not None else make_conversation("Q?", "A.")

    llm.generate_async = AsyncMock(side_effect=generate_async)
    return llm


# ---------------------------------------------------------------------------
# Disabled: current behaviour, unchanged
# ---------------------------------------------------------------------------


async def test_disabled_returns_a_single_conversation():
    llm = make_llm()
    builder = SingleShotBuilder(llm, make_config())

    result = await builder.generate("Houseplants")

    assert isinstance(result, Conversation)
    assert [m.role for m in result.messages] == ["system", "user", "assistant"]


async def test_disabled_output_is_unchanged_by_the_feature():
    """Golden test: with the flag off, the emitted sample is byte-identical."""
    single = make_conversation("How do I water a fern?", "Keep the soil damp but never soggy.")
    builder = SingleShotBuilder(make_llm(single=single), make_config())

    result = await builder.generate("Houseplants")

    assert result.model_dump(exclude_none=True) == {
        "messages": [
            {"role": "system", "content": "You are a helpful gardening assistant."},
            {"role": "user", "content": "How do I water a fern?"},
            {"role": "assistant", "content": "Keep the soil damp but never soggy."},
        ],
        "question": "",
        "final_answer": "",
    }


async def test_disabled_uses_the_plain_generation_prompt():
    llm = make_llm()
    builder = SingleShotBuilder(llm, make_config())

    await builder.generate("Houseplants")

    prompt = llm.generate_async.await_args.kwargs["prompt"]
    assert "VERBALIZED SAMPLING" not in prompt
    assert llm.generate_async.await_args.kwargs["schema"] is Conversation


# ---------------------------------------------------------------------------
# Enabled: fan-out at SingleShotBuilder.generate (§6.1, §6.3)
# ---------------------------------------------------------------------------


async def test_enabled_returns_one_conversation_per_selected_candidate():
    builder = SingleShotBuilder(make_llm(), make_config(**verbalized_config(select=3)))

    result = await builder.generate("Houseplants")

    assert isinstance(result, list)
    assert len(result) == 3  # noqa: PLR2004
    assert all(isinstance(c, Conversation) for c in result)


async def test_enabled_selects_distinct_candidates():
    builder = SingleShotBuilder(make_llm(), make_config(**verbalized_config(select=3)))

    result = await builder.generate("Houseplants")

    texts = [conversation_text(c) for c in result]
    assert len(set(texts)) == len(texts)


async def test_enabled_finalizes_every_candidate():
    builder = SingleShotBuilder(make_llm(), make_config(**verbalized_config(select=2)))

    result = await builder.generate("Houseplants")

    for conversation in result:
        assert [m.role for m in conversation.messages] == ["system", "user", "assistant"]
        assert conversation.messages[0].content == "You are a helpful gardening assistant."


async def test_enabled_respects_sys_msg_false():
    config = make_config(sys_msg=False, **verbalized_config(select=2))
    builder = SingleShotBuilder(make_llm(), config)

    result = await builder.generate("Houseplants")

    for conversation in result:
        assert conversation.messages[0].role == "user"


async def test_enabled_records_verbalized_metadata():
    builder = SingleShotBuilder(make_llm(), make_config(**verbalized_config(select=2)))

    result = await builder.generate("Houseplants")

    for rank, conversation in enumerate(result):
        recorded = conversation.metadata["verbalized"]
        assert recorded["approach"].startswith("approach-")
        assert 0.0 <= recorded["probability"] <= 1.0
        assert recorded["rank"] == rank


async def test_metadata_can_be_switched_off():
    config = make_config(**verbalized_config(select=2, keep_probabilities=False))
    builder = SingleShotBuilder(make_llm(), config)

    result = await builder.generate("Houseplants")

    assert all(c.metadata is None for c in result)


async def test_distribution_prompt_keeps_the_base_prompt_and_asks_for_k():
    llm = make_llm()
    builder = SingleShotBuilder(llm, make_config(**verbalized_config(k=4, select=2)))

    await builder.generate("Houseplants")

    kwargs = llm.generate_async.await_args.kwargs
    assert kwargs["schema"] is distribution_model(Conversation)
    prompt = kwargs["prompt"]
    assert "Topic/Scenario: Houseplants" in prompt
    assert "You are a helpful gardening assistant." in prompt
    assert "generate 4 genuinely different candidates" in prompt


async def test_distribution_prompt_carries_retry_feedback():
    llm = make_llm()
    builder = SingleShotBuilder(llm, make_config(**verbalized_config(k=2, select=1)))

    await builder.generate("Houseplants", error_feedback="messages was empty")

    assert "messages was empty" in llm.generate_async.await_args.kwargs["prompt"]


@pytest.mark.parametrize(
    ("conversation_type", "reasoning_style", "expected"),
    [
        ("basic", None, "basic"),
        ("cot", "freetext", "cot-freetext"),
        ("cot", "agent", "cot-agent"),
        ("cot", None, "basic"),
    ],
)
def test_prompt_style_selection(conversation_type, reasoning_style, expected):
    assert VerbalizedPromptBuilder.style_key(conversation_type, reasoning_style) == expected


async def test_cot_freetext_gets_reasoning_specific_guidance():
    config = make_config(
        conversation_type="cot", reasoning_style="freetext", **verbalized_config(k=2, select=1)
    )
    llm = make_llm()
    builder = SingleShotBuilder(llm, config)

    await builder.generate("Houseplants")

    prompt = llm.generate_async.await_args.kwargs["prompt"]
    assert "reasoning" in prompt.lower()
    assert "VERBALIZED SAMPLING" in prompt


async def test_token_budget_scales_with_k():
    llm = make_llm()
    config = make_config(max_tokens=1000, **verbalized_config(k=4, select=2))
    builder = SingleShotBuilder(llm, config)

    await builder.generate("Houseplants")

    assert llm.generate_async.await_args.kwargs["max_tokens"] == 4000  # noqa: PLR2004


async def test_token_budget_can_be_set_explicitly():
    llm = make_llm()
    config = make_config(max_tokens=1000, **verbalized_config(k=4, select=2, max_tokens=1234))
    builder = SingleShotBuilder(llm, config)

    await builder.generate("Houseplants")

    assert llm.generate_async.await_args.kwargs["max_tokens"] == 1234  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Validation and dedup behaviour
# ---------------------------------------------------------------------------


async def test_malformed_candidate_is_dropped_not_fatal():
    distribution = make_distribution(count=3, first_role="assistant")
    config = make_config(
        **verbalized_config(k=3, select=3, selection="threshold", prob_threshold=1.0)
    )
    builder = SingleShotBuilder(make_llm(distribution), config)

    result = await builder.generate("Houseplants")

    assert len(result) == 2  # noqa: PLR2004
    assert all(c.messages[1].role == "user" for c in result)


async def test_all_candidates_invalid_raises_a_retryable_error():
    from deepfabric.utils import is_validation_error  # noqa: PLC0415

    distribution = make_distribution(count=1, first_role="assistant")
    config = make_config(**verbalized_config(k=1, select=1))
    builder = SingleShotBuilder(make_llm(distribution), config)

    with pytest.raises(ValueError, match="no verbalized candidate") as exc:
        await builder.generate("Houseplants")

    assert is_validation_error(exc.value)


async def test_duplicate_candidates_collapse_to_one():
    duplicate = CANDIDATE_TEXTS[0]
    distribution = distribution_model(Conversation).model_validate(
        {
            "candidates": [
                {
                    "approach": f"approach-{i}",
                    "probability": 0.3,
                    "payload": make_conversation(*duplicate).model_dump(),
                }
                for i in range(3)
            ]
        }
    )
    builder = SingleShotBuilder(
        make_llm(distribution), make_config(**verbalized_config(k=3, select=3))
    )

    result = await builder.generate("Houseplants")

    assert len(result) == 1


async def test_dedup_can_be_disabled():
    duplicate = CANDIDATE_TEXTS[0]
    distribution = distribution_model(Conversation).model_validate(
        {
            "candidates": [
                {
                    "approach": f"approach-{i}",
                    "probability": 0.3,
                    "payload": make_conversation(*duplicate).model_dump(),
                }
                for i in range(3)
            ]
        }
    )
    config = make_config(**verbalized_config(k=3, select=3, dedup={"enabled": False}))
    builder = SingleShotBuilder(make_llm(distribution), config)

    result = await builder.generate("Houseplants")

    assert len(result) == 3  # noqa: PLR2004


async def test_progress_reporter_receives_realised_diversity():
    reporter = MagicMock()
    config = make_config(**verbalized_config(k=5, select=3))
    builder = SingleShotBuilder(make_llm(), config, progress_reporter=reporter)

    await builder.generate("Houseplants")

    step_name, metadata = (
        reporter.emit_step_complete.call_args.args[0],
        (reporter.emit_step_complete.call_args.kwargs),
    )
    assert step_name == "Verbalized sampling"
    assert metadata["verbalized_selected"] == 3  # noqa: PLR2004
    assert metadata["verbalized_generated"] == 5  # noqa: PLR2004
    assert 0.0 <= metadata["verbalized_diversity"] <= 1.0


# ---------------------------------------------------------------------------
# Sample-count reconciliation (§6.2)
# ---------------------------------------------------------------------------


def make_topic_model(count=2):
    """Mock topic model exposing `count` unique topics."""
    topics = [f"topic-{i}" for i in range(count)]
    model = MagicMock()
    model.tree_paths = topics
    model.get_all_paths.return_value = [[t] for t in topics]
    model.get_all_paths_with_ids.return_value = [
        TopicPath(path=[t], topic_id=f"uuid-{i}") for i, t in enumerate(topics)
    ]
    model.get_unique_topics.return_value = [
        Topic(uuid=f"uuid-{i}", topic=t) for i, t in enumerate(topics)
    ]
    model.get_path_by_id.return_value = None
    return model


def make_generator(**overrides):
    """Build a generator whose LLM client returns verbalised distributions."""
    args = {
        "generation_system_prompt": "You are a helpful gardening assistant.",
        "dataset_system_prompt": "You are a helpful gardening assistant.",
        "provider": "openai",
        "model_name": "test-model",
    }
    args.update(overrides)
    with patch("deepfabric.generator.LLMClient"):
        generator = DataSetGenerator(**args)
    generator.llm_client.generate_async = make_llm().generate_async
    return generator


def test_budget_mode_holds_the_dataset_size():
    generator = make_generator(**verbalized_config(k=5, select=3, sizing="budget"))

    dataset = generator.create_data(num_steps=1, batch_size=2, topic_model=make_topic_model(2))

    assert len(dataset) == 2  # noqa: PLR2004
    # The samples really did come through the verbalized path.
    assert all("verbalized" in record["metadata"] for record in dataset)


def test_multiplier_mode_grows_the_dataset():
    generator = make_generator(**verbalized_config(k=5, select=3, sizing="multiplier"))

    dataset = generator.create_data(num_steps=1, batch_size=2, topic_model=make_topic_model(2))

    assert len(dataset) == 6  # 2 node-slots x 3 selected candidates  # noqa: PLR2004


def test_disabled_generation_produces_one_sample_per_slot():
    generator = make_generator()

    dataset = generator.create_data(num_steps=1, batch_size=2, topic_model=make_topic_model(2))

    assert len(dataset) == 2  # noqa: PLR2004
    assert all("verbalized" not in (record.get("metadata") or {}) for record in dataset)


def test_verbalized_samples_keep_topic_traceability():
    generator = make_generator(**verbalized_config(k=5, select=2, sizing="multiplier"))

    dataset = generator.create_data(num_steps=1, batch_size=2, topic_model=make_topic_model(2))

    topic_ids = {record["metadata"]["topic_id"] for record in dataset}
    assert topic_ids == {"uuid-0", "uuid-1"}


def test_agent_path_warns_that_verbalized_sampling_is_skipped(caplog):
    """Phase 2 covers the single-shot path only; the agent path says so out loud."""
    from deepfabric.builders import ConversationBuilderFactory  # noqa: PLC0415
    from deepfabric.schemas import ToolRegistry  # noqa: PLC0415

    config = make_config(
        conversation_type="cot",
        reasoning_style="agent",
        custom_tools=[
            {
                "name": "get_weather",
                "description": "Look up the weather",
                "parameters": [
                    {
                        "name": "city",
                        "type": "str",
                        "description": "City name",
                        "required": True,
                    }
                ],
                "returns": "Weather report",
                "category": "utility",
            }
        ],
        **verbalized_config(),
    )

    with caplog.at_level("WARNING"):
        ConversationBuilderFactory.create(
            config=config, llm=make_llm(), tool_registry=ToolRegistry(tools=[])
        )

    assert "will be skipped for this agent run" in caplog.text
