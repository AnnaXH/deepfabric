"""Offline unit tests for the shared verbalized-sampling package (spec §4).

Everything here runs without network access: the LLM client and the similarity
measure are stubbed. Embedding-backed behaviour is covered separately in
tests/integration/test_verbalized_integration.py.
"""

import random

from unittest.mock import AsyncMock

import pytest

from pydantic import BaseModel, ValidationError

from deepfabric.verbalized import (
    DedupConfig,
    DiversityConfig,
    LexicalSimilarity,
    SimilarityMeasure,
    VerbalizedCandidate,
    VerbalizedConfig,
    VerbalizedSampler,
    build_similarity,
    distinct_n,
    distribution_model,
    normalize_probabilities,
    rank_candidates,
    rouge_l,
    select_candidates,
    semantic_diversity,
    to_candidates,
)


class Payload(BaseModel):
    """Minimal payload schema used to exercise the typed distribution model."""

    text: str


def make_candidates(probabilities, prefix="cand"):
    """Build candidates with distinct payload text and the given probabilities."""
    return [
        VerbalizedCandidate(approach=f"{prefix}-{i}", probability=p, payload=f"{prefix} text {i}")
        for i, p in enumerate(probabilities)
    ]


class StubSimilarity(SimilarityMeasure):
    """Similarity from an explicit lookup table; identical texts score 1.0."""

    name = "stub"

    def __init__(self, threshold=0.8, scores=None):
        super().__init__(threshold)
        self.scores = scores or {}
        self.prepared: list[str] = []

    async def prepare(self, texts):
        self.prepared = list(texts)

    async def similarity(self, a, b):
        if a == b:
            return 1.0
        return self.scores.get(frozenset((a, b)), 0.0)


# ---------------------------------------------------------------------------
# Config (§4.1)
# ---------------------------------------------------------------------------


def test_config_defaults_are_disabled():
    config = VerbalizedConfig()
    assert config.enabled is False
    assert config.k == 5  # noqa: PLR2004
    assert config.select == 3  # noqa: PLR2004
    assert config.selection == "tail_weighted"
    assert config.expansion == "single_pass"
    assert config.dedup.enabled is True


def test_diversity_config_nests_verbalized_block():
    diversity = DiversityConfig()
    assert diversity.verbalized.enabled is False


def test_config_rejects_select_greater_than_k():
    with pytest.raises(ValidationError, match=r"select .* must be <= verbalized.k"):
        VerbalizedConfig(k=3, select=5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("k", 0),
        ("select", 0),
        ("prob_threshold", 1.5),
        ("temperature", 0.0),
        ("tune_diversity", 2.0),
        ("top_p", 0.0),
    ],
)
def test_config_rejects_out_of_range_values(field, value):
    with pytest.raises(ValidationError):
        VerbalizedConfig(**{field: value})


def test_config_rejects_unknown_selection_strategy():
    with pytest.raises(ValidationError):
        VerbalizedConfig(selection="most_likely")


def test_two_pass_expansion_is_not_yet_supported():
    with pytest.raises(ValidationError, match="not yet supported"):
        VerbalizedConfig(enabled=True, expansion="two_pass")


def test_two_pass_is_accepted_while_disabled():
    # The value stays in the config surface; it only errors when a seam enables it.
    assert VerbalizedConfig(expansion="two_pass").expansion == "two_pass"


def test_tune_diversity_overrides_temperature():
    assert VerbalizedConfig(temperature=2.0).effective_temperature == 2.0  # noqa: PLR2004
    assert VerbalizedConfig(temperature=2.0, tune_diversity=0.0).effective_temperature == 1.0
    assert VerbalizedConfig(tune_diversity=1.0).effective_temperature > 1.0
    assert VerbalizedConfig(tune_diversity=0.5).inversion_weight == 0.5  # noqa: PLR2004
    assert VerbalizedConfig().inversion_weight == 0.0


# ---------------------------------------------------------------------------
# Schemas (§4.2)
# ---------------------------------------------------------------------------


def test_probabilities_need_not_sum_to_one():
    candidates = make_candidates([0.9, 0.9, 0.9])
    assert [c.probability for c in candidates] == [0.9, 0.9, 0.9]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(1.4, 1.0), (-0.2, 0.0), ("0.25", 0.25), ("40%", 0.4)],
)
def test_probability_is_clamped_not_rejected(raw, expected):
    assert VerbalizedCandidate(approach="a", probability=raw).probability == expected


def test_distribution_model_is_typed_and_cached():
    model = distribution_model(Payload)
    assert model is distribution_model(Payload)

    schema = model.model_json_schema()
    assert "candidates" in schema["properties"]
    candidate_schema = schema["$defs"]["PayloadCandidate"]
    assert set(candidate_schema["properties"]) == {"approach", "probability", "payload"}


def test_distribution_model_validates_payload_type():
    model = distribution_model(Payload)
    with pytest.raises(ValidationError):
        model.model_validate({"candidates": [{"approach": "a", "probability": 0.5, "payload": 7}]})


def test_to_candidates_preserves_payload_objects():
    model = distribution_model(Payload)
    parsed = model.model_validate(
        {"candidates": [{"approach": "a", "probability": 0.5, "payload": {"text": "hello"}}]}
    )
    candidates = to_candidates(parsed)
    assert len(candidates) == 1
    assert isinstance(candidates[0].payload, Payload)
    assert candidates[0].payload.text == "hello"


def test_to_candidates_handles_empty_distribution():
    assert to_candidates(distribution_model(Payload)(candidates=[])) == []


# ---------------------------------------------------------------------------
# Selection strategies (§4.3)
# ---------------------------------------------------------------------------


def test_normalize_probabilities_sums_to_one():
    probabilities = normalize_probabilities(make_candidates([0.4, 0.2, 0.2]))
    assert probabilities == pytest.approx([0.5, 0.25, 0.25])


def test_normalize_probabilities_falls_back_to_uniform():
    probabilities = normalize_probabilities(make_candidates([0.0, 0.0]))
    assert probabilities == pytest.approx([0.5, 0.5])
    assert normalize_probabilities([]) == []


def test_tail_weighted_is_reproducible_with_a_seed():
    candidates = make_candidates([0.5, 0.25, 0.15, 0.07, 0.03])
    config = VerbalizedConfig(k=5, select=3, seed=1234)
    first = select_candidates(candidates, config, random.Random(7))  # noqa: S311
    second = select_candidates(candidates, config, random.Random(7))  # noqa: S311
    assert [c.approach for c in first] == [c.approach for c in second]
    assert len(first) == 3  # noqa: PLR2004


def test_tail_weighted_moves_off_the_mode():
    """Over many draws the mode must not dominate the way its probability implies."""
    candidates = make_candidates([0.9, 0.04, 0.03, 0.02, 0.01])
    config = VerbalizedConfig(k=5, select=1, temperature=4.0, tune_diversity=1.0)
    rng = random.Random(0)  # noqa: S311
    draws = 200
    picks = [select_candidates(candidates, config, rng)[0].approach for _ in range(draws)]
    # A faithful sampler would pick the 0.9-probability candidate ~180 times.
    assert picks.count("cand-0") < draws / 2


def test_higher_tune_diversity_picks_the_mode_less_often():
    candidates = make_candidates([0.7, 0.1, 0.1, 0.05, 0.05])

    def mode_share(tune):
        config = VerbalizedConfig(k=5, select=1, tune_diversity=tune)
        rng = random.Random(11)  # noqa: S311
        return sum(
            select_candidates(candidates, config, rng)[0].approach == "cand-0" for _ in range(300)
        )

    assert mode_share(1.0) < mode_share(0.0)


def test_threshold_keeps_only_low_probability_candidates():
    candidates = make_candidates([0.5, 0.25, 0.15, 0.07, 0.03])
    config = VerbalizedConfig(k=5, select=3, selection="threshold", prob_threshold=0.15)
    selected = select_candidates(candidates, config)
    assert [c.approach for c in selected] == ["cand-4", "cand-3", "cand-2"]
    assert all(c.probability <= 0.15 for c in selected)  # noqa: PLR2004


def test_threshold_falls_back_to_median_when_nothing_qualifies():
    candidates = make_candidates([0.5, 0.25, 0.15, 0.07, 0.03])
    config = VerbalizedConfig(k=5, select=3, selection="threshold", prob_threshold=0.001)
    selected = select_candidates(candidates, config)
    assert [c.approach for c in selected] == ["cand-2"]


def test_top_p_complement_drops_the_probability_core():
    candidates = make_candidates([0.5, 0.25, 0.15, 0.07, 0.03])
    config = VerbalizedConfig(k=5, select=3, selection="top_p_complement", top_p=0.9)
    selected = select_candidates(candidates, config)
    assert [c.approach for c in selected] == ["cand-3", "cand-4"]


def test_top_p_complement_falls_back_when_core_covers_everything():
    candidates = make_candidates([0.5, 0.5])
    config = VerbalizedConfig(k=2, select=2, selection="top_p_complement", top_p=1.0)
    selected = select_candidates(candidates, config)
    assert len(selected) == 1


@pytest.mark.parametrize("strategy", ["tail_weighted", "threshold", "top_p_complement"])
def test_every_strategy_keeps_at_least_one(strategy):
    candidates = make_candidates([1.0])
    config = VerbalizedConfig(k=1, select=1, selection=strategy)
    assert len(select_candidates(candidates, config)) == 1


@pytest.mark.parametrize("strategy", ["tail_weighted", "threshold", "top_p_complement"])
def test_every_strategy_never_exceeds_select(strategy):
    candidates = make_candidates([0.4, 0.3, 0.2, 0.05, 0.05])
    config = VerbalizedConfig(k=5, select=2, selection=strategy, prob_threshold=1.0)
    assert len(select_candidates(candidates, config)) <= 2  # noqa: PLR2004


@pytest.mark.parametrize("strategy", ["tail_weighted", "threshold", "top_p_complement"])
def test_ranking_covers_every_candidate(strategy):
    candidates = make_candidates([0.4, 0.3, 0.2, 0.05, 0.05])
    ranking = rank_candidates(candidates, VerbalizedConfig(k=5, select=3, selection=strategy))
    assert sorted(c.approach for c in ranking.ordered) == sorted(c.approach for c in candidates)
    assert 1 <= ranking.eligible <= len(candidates)


def test_ranking_handles_no_candidates():
    ranking = rank_candidates([], VerbalizedConfig())
    assert ranking.ordered == []
    assert ranking.eligible == 0
    assert select_candidates([], VerbalizedConfig()) == []


def test_unknown_strategy_raises():
    config = VerbalizedConfig()
    object.__setattr__(config, "selection", "nonsense")
    with pytest.raises(ValueError, match="Unknown selection strategy"):
        rank_candidates(make_candidates([0.5]), config)


# ---------------------------------------------------------------------------
# Similarity and dedup (§4.4) - lexical/stub only, no API calls
# ---------------------------------------------------------------------------


def test_rouge_l_bounds():
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert rouge_l("alpha beta gamma", "delta epsilon zeta") == 0.0
    assert 0.0 < rouge_l("the cat sat on the mat", "the cat sat on a rug") < 1.0
    assert rouge_l("", "") == 1.0
    assert rouge_l("something", "") == 0.0


def test_distinct_n():
    assert distinct_n(["a b c", "a b c"], n=2) == 0.5  # noqa: PLR2004
    assert distinct_n(["a b c", "d e f"], n=2) == 1.0
    assert distinct_n(["a"], n=2) == 0.0


async def test_semantic_diversity_direction():
    measure = LexicalSimilarity(0.8)
    identical = await semantic_diversity(["the cat sat", "the cat sat"], measure)
    varied = await semantic_diversity(["the cat sat", "quantum tunnelling explained"], measure)
    assert identical == 0.0
    assert varied > identical
    assert await semantic_diversity(["only one"], measure) == 0.0


async def test_is_duplicate_uses_the_threshold():
    measure = StubSimilarity(threshold=0.8, scores={frozenset(("a", "b")): 0.9})
    assert await measure.is_duplicate("a", ["b"]) is True
    assert await measure.is_duplicate("a", ["c"]) is False
    assert await measure.is_duplicate("a", []) is False


async def test_lexical_measure_flags_near_duplicates():
    measure = LexicalSimilarity(0.8)
    assert await measure.is_duplicate("the cat sat on the mat", ["the cat sat on the mat"]) is True
    assert await measure.is_duplicate("the cat sat on the mat", ["rocket fuel chemistry"]) is False


def test_build_similarity_falls_back_without_embedding_provider(monkeypatch, caplog):  # noqa: ARG001
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    measure = build_similarity(DedupConfig(max_lexical_similarity=0.42))
    assert isinstance(measure, LexicalSimilarity)
    assert measure.threshold == 0.42  # noqa: PLR2004


def test_build_similarity_caches_by_config(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = DedupConfig(max_lexical_similarity=0.51)
    assert build_similarity(config) is build_similarity(DedupConfig(max_lexical_similarity=0.51))


# ---------------------------------------------------------------------------
# Sampler end-to-end with a mocked LLM (§4)
# ---------------------------------------------------------------------------


def make_llm(payload_texts, probabilities=None):
    """Mock LLM client returning a verbalised distribution of Payload candidates."""
    probabilities = probabilities or [0.5, 0.3, 0.2, 0.1, 0.05][: len(payload_texts)]
    model = distribution_model(Payload)
    distribution = model.model_validate(
        {
            "candidates": [
                {"approach": f"approach-{i}", "probability": p, "payload": {"text": text}}
                for i, (text, p) in enumerate(zip(payload_texts, probabilities, strict=True))
            ]
        }
    )
    llm = AsyncMock()
    llm.generate_async = AsyncMock(return_value=distribution)
    return llm


async def test_sampler_returns_selected_payloads():
    llm = make_llm(["alpha one", "beta two", "gamma three", "delta four"])
    config = VerbalizedConfig(enabled=True, k=4, select=2, seed=3)
    sampler = VerbalizedSampler(config, llm, similarity=StubSimilarity())

    result = await sampler.sample("prompt", Payload, text_of=lambda p: p.text)

    assert len(result.selected) == 2  # noqa: PLR2004
    assert result.generated == 4  # noqa: PLR2004
    assert result.duplicates_dropped == 0
    assert all(isinstance(p, Payload) for p in result.payloads)
    assert llm.generate_async.await_count == 1
    assert llm.generate_async.await_args.kwargs["schema"] is distribution_model(Payload)


async def test_sampler_forwards_llm_kwargs():
    llm = make_llm(["alpha one", "beta two"])
    sampler = VerbalizedSampler(
        VerbalizedConfig(enabled=True, k=2, select=1), llm, StubSimilarity()
    )

    await sampler.sample("prompt", Payload, text_of=lambda p: p.text, temperature=0.9)

    assert llm.generate_async.await_args.kwargs["temperature"] == 0.9  # noqa: PLR2004


async def test_sampler_drops_near_duplicates_and_backfills():
    llm = make_llm(["alpha", "alpha-copy", "beta", "gamma"], [0.1, 0.2, 0.3, 0.4])
    measure = StubSimilarity(threshold=0.8, scores={frozenset(("alpha", "alpha-copy")): 0.95})
    config = VerbalizedConfig(
        enabled=True, k=4, select=2, selection="threshold", prob_threshold=1.0
    )
    sampler = VerbalizedSampler(config, llm, similarity=measure)

    result = await sampler.sample("prompt", Payload, text_of=lambda p: p.text)

    texts = [p.text for p in result.payloads]
    assert len(texts) == 2  # noqa: PLR2004
    assert not ("alpha" in texts and "alpha-copy" in texts)
    assert result.duplicates_dropped == 1
    assert measure.prepared  # the candidate set was prepared in one batch


async def test_sampler_keeps_one_when_everything_is_a_duplicate():
    llm = make_llm(["same", "same", "same"], [0.4, 0.3, 0.3])
    config = VerbalizedConfig(enabled=True, k=3, select=3)
    sampler = VerbalizedSampler(config, llm, similarity=StubSimilarity(threshold=0.5))

    result = await sampler.sample("prompt", Payload, text_of=lambda p: p.text)

    assert len(result.selected) == 1


async def test_sampler_respects_dedup_disabled():
    llm = make_llm(["same", "same"], [0.5, 0.5])
    config = VerbalizedConfig(enabled=True, k=2, select=2, dedup=DedupConfig(enabled=False))
    sampler = VerbalizedSampler(config, llm, similarity=StubSimilarity(threshold=0.5))

    result = await sampler.sample("prompt", Payload, text_of=lambda p: p.text)

    assert len(result.selected) == 2  # noqa: PLR2004
    assert result.measure is None


async def test_sampler_reports_realised_diversity():
    llm = make_llm(["alpha one", "beta two"], [0.5, 0.5])
    config = VerbalizedConfig(enabled=True, k=2, select=2)
    sampler = VerbalizedSampler(config, llm, similarity=LexicalSimilarity(0.8))

    result = await sampler.sample("prompt", Payload, text_of=lambda p: p.text)

    assert result.diversity is not None
    assert 0.0 <= result.diversity <= 1.0
    assert result.measure == "lexical_rouge_l"


async def test_sampler_records_probabilities():
    llm = make_llm(["alpha", "beta"], [0.6, 0.2])
    config = VerbalizedConfig(
        enabled=True, k=2, select=2, selection="threshold", prob_threshold=1.0
    )
    sampler = VerbalizedSampler(config, llm, similarity=StubSimilarity())

    result = await sampler.sample("prompt", Payload, text_of=lambda p: p.text)

    by_approach = {c.approach: c for c in result.selected}
    assert by_approach["approach-0"].probability == 0.6  # noqa: PLR2004
    assert by_approach["approach-0"].normalized_probability == pytest.approx(0.75)
    assert [c.rank for c in result.selected] == [0, 1]


async def test_sampler_raises_validation_style_error_on_empty_distribution():
    llm = make_llm([])
    llm.generate_async = AsyncMock(return_value=distribution_model(Payload)(candidates=[]))
    sampler = VerbalizedSampler(VerbalizedConfig(enabled=True), llm, StubSimilarity())

    with pytest.raises(ValueError, match="no candidates") as exc:
        await sampler.sample("prompt", Payload)

    # Phrased so the generator's sample-retry path treats it as retryable.
    from deepfabric.utils import is_validation_error  # noqa: PLC0415

    assert is_validation_error(exc.value)


async def test_sampler_refuses_to_run_while_disabled():
    sampler = VerbalizedSampler(VerbalizedConfig(enabled=False), make_llm(["a"]), StubSimilarity())
    with pytest.raises(ValueError, match="disabled"):
        await sampler.sample("prompt", Payload)


async def test_sampler_default_text_of_handles_models():
    llm = make_llm(["alpha one", "beta two"], [0.5, 0.5])
    config = VerbalizedConfig(enabled=True, k=2, select=2)
    sampler = VerbalizedSampler(config, llm, similarity=LexicalSimilarity(0.99))

    result = await sampler.sample("prompt", Payload)

    assert len(result.selected) == 2  # noqa: PLR2004
