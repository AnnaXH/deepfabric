"""Provider-free stop/resume tests with real clients, retries, and slot coordination."""
# ruff: noqa: PLR2004, ARG002, SIM117

import asyncio
import json

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from google.genai.errors import ClientError

from deepfabric.checkpoint import read_jsonl, validate_checkpoint
from deepfabric.constants import ERROR_DATASET_FILENAME
from deepfabric.exceptions import ProviderBlockedError
from deepfabric.generator import DataSetGenerator
from deepfabric.llm.client import LLMClient
from deepfabric.llm.token_usage import record_usage
from deepfabric.run_report import RunReport, reporting

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
    "required": ["name", "value"],
    "additionalProperties": False,
}
MODES = [
    ("independent", None),
    ("bulk", "full"),
    ("bulk", "seed"),
    ("sequential", "full"),
    ("sequential", "seed"),
]


def topics():
    return SimpleNamespace(
        get_unique_topics=lambda: [SimpleNamespace(uuid=key, topic=key) for key in ("a", "b")],
        get_path_by_id=lambda key: ["Root", key],
    )


def quota_error():
    return openai.RateLimitError(
        "Credit balance exhausted",
        response=httpx.Response(429, request=httpx.Request("POST", "https://provider.invalid")),
        body={"code": "credit_balance_exhausted", "message": "Credit balance exhausted"},
    )


class Model:
    def __init__(self, mode, scope, fail_at, error=None):
        self.mode, self.scope, self.fail_at = mode, scope, fail_at
        self.error = error or quota_error()
        self.calls = 0
        self.recovered = False

    async def __call__(self, prompt, schema, **kwargs):
        self.calls += 1
        if not self.recovered and self.calls >= self.fail_at:
            raise self.error
        record_usage("openai", "test", {"prompt_tokens": 10, "completion_tokens": 2})
        props = schema.model_json_schema()["properties"]
        if "items" in props:
            items = [f"seed-{self.calls}-{i}" for i in range(3)]
            if self.scope == "full":
                items = [{"name": seed, "value": 1} for seed in items]
            return json.dumps({"items": items})
        if "name" not in props:
            return '{"value": 1}'
        return json.dumps({"name": f"sample-{self.calls}", "value": 1})


def make_engine(
    tmp_path,
    model,
    mode="independent",
    scope=None,
    checkpoint=10,
    provider="openai",
    retries=1,
    custom=True,
):
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=model),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=model),
    ):
        client = LLMClient(provider, "test", rate_limit_config={"max_retries": retries})
        assert client.async_model is model
    options = {} if scope is None else {"shared_scope": scope}
    if scope == "seed" and custom:
        options["seed_field"] = "name"
    with patch("deepfabric.generator.LLMClient", return_value=client):
        return DataSetGenerator(
            generation_system_prompt="test",
            provider=provider,
            model_name="test",
            output_schema=SCHEMA if custom else None,
            topic_sample_mode=mode,
            checkpoint_interval=checkpoint,
            checkpoint_path=str(tmp_path),
            output_save_as=str(tmp_path / "samples.jsonl"),
            sample_retries=0,
            **options,
        )


async def collect_stop(engine, *, topic_model, batch_size=1, report=None):
    events = []
    with reporting(report), pytest.raises(ProviderBlockedError) as caught:
        async for event in engine.create_data_with_events_async(
            num_samples=6, batch_size=batch_size, topic_model=topic_model
        ):
            if isinstance(event, dict):
                events.append(event)
    assert events[-1]["event"] == "generation_error"
    assert not any(e["event"] == "generation_complete" for e in events)
    return caught.value, events[-1]


@pytest.mark.parametrize("mode,scope", MODES)
async def test_stop_preserves_accepted_work_and_ordinary_resume(tmp_path, monkeypatch, mode, scope):
    monkeypatch.chdir(tmp_path)
    # Seed collections are followed by completions: stop after one accepted completion.
    fail_at = 3 if mode == "bulk" and scope == "seed" else 2
    model = Model(mode, scope, fail_at)
    engine = make_engine(tmp_path, model, mode, scope)
    report = RunReport()
    error, event = await collect_stop(engine, topic_model=topics(), report=report)
    assert engine.stop_requested
    assert engine.get_all_failures() == []
    assert event["failed_samples"] == event["new_failures"] == 0
    assert event["total_samples"] > 0
    assert event["unaccounted"] == 6 - event["total_samples"]
    assert "--resume" in str(error)
    metadata = json.loads(engine._checkpoint_metadata_path.read_text())
    validate_checkpoint(metadata, engine._checkpoint_samples_path, engine._checkpoint_failures_path)
    accepted_before = read_jsonl(engine._checkpoint_samples_path)
    assert len(engine._completed) == len(accepted_before) == event["total_samples"]
    assert read_jsonl(tmp_path / ERROR_DATASET_FILENAME) == accepted_before
    result = report.snapshot(engine.llm_client.token_usage.snapshot(), "failed")
    assert result["status"] == "stopped"
    assert result["summary"]["accepted_samples"] == len(accepted_before)
    assert result["summary"]["failed_samples"] == 0
    assert result["summary"]["requests"] == model.calls
    failed_usage = result["token_usage"]["records"][-1]
    assert failed_usage["tokens"]["total_tokens"] is None
    # Ordinary resume must not need --retry-failed, including a blocked collection.
    model.recovered = True
    assert engine.load_checkpoint()
    dataset = await engine.create_data_async(num_samples=6, batch_size=1, topic_model=topics())
    assert len(dataset) == len(engine._completed) == 6
    assert engine.get_all_failures() == []
    assert list(dataset)[: len(accepted_before)] == accepted_before


@pytest.mark.parametrize("mode,scope", MODES)
async def test_first_call_stop_keeps_empty_valid_checkpoint_and_bulk_budget(
    tmp_path, monkeypatch, mode, scope
):
    monkeypatch.chdir(tmp_path)
    model = Model(mode, scope, 1)
    engine = make_engine(tmp_path, model, mode, scope)
    _, event = await collect_stop(engine, topic_model=topics(), batch_size=2)
    assert model.calls == 1
    assert event["unaccounted"] == 6
    assert not engine._completed
    assert not engine.get_all_failures()
    assert engine.load_checkpoint()
    if mode == "bulk":
        assert all(count == 0 for count in engine._collection_attempts.values())
    model.recovered = True
    dataset = await engine.create_data_async(num_samples=6, batch_size=2, topic_model=topics())
    assert len(dataset) == 6


@pytest.mark.parametrize("mode,scope", MODES)
async def test_basic_conversation_paths_propagate_provider_stop(tmp_path, monkeypatch, mode, scope):
    monkeypatch.chdir(tmp_path)
    model = Model(mode, scope, 1)
    engine = make_engine(tmp_path, model, mode, scope, custom=False)
    _, event = await collect_stop(engine, topic_model=topics())
    assert model.calls == 1
    assert event["unaccounted"] == 6
    assert not engine.get_all_failures()


@pytest.mark.parametrize("custom", [False, True])
async def test_bulk_seed_completion_stop_keeps_seeds_without_settling_slots(
    tmp_path, monkeypatch, custom
):
    monkeypatch.chdir(tmp_path)
    model = Model("bulk", "seed", 2)
    engine = make_engine(tmp_path, model, "bulk", "seed", custom=custom)
    _, event = await collect_stop(engine, topic_model=topics())
    assert model.calls == 2
    assert event["unaccounted"] == 6
    assert not engine.get_all_failures()
    assert engine.load_checkpoint()
    assert len(engine._topic_seeds["a"]) == 3


@pytest.mark.parametrize("checkpoint", [None, 10])
async def test_no_topic_stop_and_resume_or_partial_output(tmp_path, monkeypatch, checkpoint):
    monkeypatch.chdir(tmp_path)
    model = Model("independent", None, 2)
    engine = make_engine(tmp_path, model, checkpoint=checkpoint)
    error, event = await collect_stop(engine, topic_model=None)
    assert event["total_samples"] == 1
    assert event["unaccounted"] == 5
    assert len(read_jsonl(tmp_path / ERROR_DATASET_FILENAME)) == 1
    if checkpoint is None:
        assert "Checkpoint resume is unavailable" in str(error)
        assert not engine.has_checkpoint()
        assert len(engine._samples) == 1
    else:
        model.recovered = True
        assert engine.load_checkpoint()
        dataset = await engine.create_data_async(num_samples=6, batch_size=1)
        assert len(dataset) == 6


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
async def test_exhausted_429_budget_stops_scheduling_not_just_one_slot(
    tmp_path, monkeypatch, provider
):
    monkeypatch.chdir(tmp_path)
    if provider == "gemini":
        error = ClientError(
            429,
            {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded"}},
        )
    else:
        error = quota_error()
        error.body = {"message": "Generic rate limit"}
    model = Model("independent", None, 1, error)
    engine = make_engine(tmp_path, model, provider=provider, retries=2)
    with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()):
        error, event = await collect_stop(engine, topic_model=None)
    assert model.calls == 3
    assert error.context["reason"] == "rate_limit_retries_exhausted"
    assert event["unaccounted"] == 6
    assert not engine.get_all_failures()


@pytest.mark.parametrize(
    "provider,status,metadata",
    [
        ("openrouter", 402, {"limit_source": "openrouter_credits"}),
        ("openrouter", 403, {"limit_source": "openrouter_workspace_budget"}),
        (
            "openrouter",
            402,
            {
                "limit_source": "openrouter_in_flight_budget",
                "reason": "in_flight_budget_exhausted",
            },
        ),
        ("openrouter", 429, {}),
        ("ollama", 429, {}),
    ],
)
@pytest.mark.parametrize("mode,scope", MODES)
async def test_router_and_raw_ollama_stop_then_ordinary_resume(
    tmp_path, monkeypatch, provider, status, metadata, mode, scope
):
    monkeypatch.chdir(tmp_path)
    response = httpx.Response(status, request=httpx.Request("POST", "https://provider.invalid"))
    cls = {429: openai.RateLimitError, 403: openai.PermissionDeniedError}.get(
        status, openai.APIStatusError
    )
    error = cls(
        "Provider blocked this request",
        response=response,
        body={
            "code": status,
            "message": "Provider blocked this request",
            "metadata": metadata,
        },
    )
    model = Model(mode, scope, 1, error)
    engine = make_engine(tmp_path, model, mode, scope, provider=provider, retries=1)
    with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()):
        _, event = await collect_stop(engine, topic_model=topics(), batch_size=2)
    assert event["unaccounted"] == 6
    assert event["failed_samples"] == 0
    assert not engine._completed
    assert engine.load_checkpoint()
    model.recovered = True
    dataset = await engine.create_data_async(num_samples=6, batch_size=2, topic_model=topics())
    assert len(dataset) == 6
    assert not engine.get_all_failures()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"patterns": ["ignore previous instructions"]},
        {"reasons": ["moderation"], "flagged_input": "prompt injection"},
    ],
)
async def test_openrouter_403_related_topic_failures_do_not_stop_other_topics(
    tmp_path, monkeypatch, metadata
):
    monkeypatch.chdir(tmp_path)

    class RelatedTopicsModel:
        calls = 0

        async def __call__(self, prompt, schema, **kwargs):
            self.calls += 1
            if self.calls <= 3:
                raise openai.PermissionDeniedError(
                    "Request blocked by content guardrail",
                    response=httpx.Response(
                        403, request=httpx.Request("POST", "https://provider.invalid")
                    ),
                    body={
                        "code": 403,
                        "message": "Request blocked by content guardrail",
                        "metadata": metadata,
                    },
                )
            return json.dumps({"name": f"unrelated-{self.calls}", "value": 1})

    model = RelatedTopicsModel()
    engine = make_engine(tmp_path, model, provider="openrouter")
    dataset = await engine.create_data_async(num_samples=6, batch_size=1, topic_model=topics())
    assert len(dataset) == 3
    assert len(engine.get_all_failures()) == 3
    assert len(engine._completed) == 6
    assert not engine.stop_requested
    assert model.calls == 6


async def test_concurrent_stop_drains_accepted_outcome_and_cancels_inflight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class ConcurrentModel:
        calls = 0

        async def __call__(self, prompt, schema, **kwargs):
            self.calls += 1
            call = self.calls
            if call == 1:
                # Both topic workers enter before the first outcome is accepted.
                await entered.wait()
                return '{"name": "accepted", "value": 1}'
            if call == 2:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            raise quota_error()

    model = ConcurrentModel()
    engine = make_engine(tmp_path, model)
    _, event = await asyncio.wait_for(
        collect_stop(engine, topic_model=topics(), batch_size=2), timeout=5
    )
    assert cancelled.is_set()
    assert model.calls == 3
    assert event["total_samples"] == 1
    assert event["unaccounted"] == 5
    assert not engine.get_all_failures()
    assert engine.load_checkpoint()
    assert len(engine._completed) == 1
