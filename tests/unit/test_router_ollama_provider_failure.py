"""OpenRouter/Ollama provider policy and wrapped SDK error handling."""
# ruff: noqa: PLR2004, SIM117

from unittest.mock import AsyncMock, Mock, patch

import httpx
import openai
import pytest

from outlines.exceptions import normalize_provider_exception
from pydantic import BaseModel

from deepfabric.exceptions import ProviderBlockedError
from deepfabric.llm.client import LLMClient
from deepfabric.llm.provider_failure import provider_failure


class Record(BaseModel):
    name: str


def api_error(status, metadata=None, message="Provider rejected request", retry_after="0"):
    response = httpx.Response(
        status,
        headers={"retry-after": retry_after},
        request=httpx.Request("POST", "https://provider.invalid"),
    )
    cls = {429: openai.RateLimitError, 403: openai.PermissionDeniedError}.get(
        status, openai.APIStatusError
    )
    return cls(
        message,
        response=response,
        body={
            "code": status,
            "message": message,
            "metadata": metadata or {},
        },
    )


def make_client(provider, model, retries=2):
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=Mock()),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=model),
    ):
        client = LLMClient(provider, "test", rate_limit_config={"max_retries": retries})
        assert client.async_model is model
    return client


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"limit_source": "openrouter_credits"},
        {"limit_source": "openrouter_key_limit"},
        {"limit_source": "openrouter_credits", "reason": "weight_exceeds_budget"},
    ],
)
async def test_openrouter_nontransient_402_stops_and_displays_provider_error(metadata):
    model = AsyncMock(side_effect=api_error(402, metadata, "This request cannot fit your budget"))
    client = make_client("openrouter", model)
    with pytest.raises(ProviderBlockedError) as caught:
        await client.generate_async("prompt", Record)
    assert model.await_count == 1
    assert "This request cannot fit your budget" in str(caught.value)
    assert caught.value.context["provider_metadata"] == metadata
    if metadata:
        assert metadata["limit_source"] in str(caught.value)


@pytest.mark.parametrize(
    "source",
    [
        "openrouter_guardrail_budget",
        "openrouter_workspace_budget",
        "openrouter_key_limit",
    ],
)
async def test_budget_403_does_not_require_scope_and_reset_fields(source):
    model = AsyncMock(side_effect=api_error(403, {"limit_source": source}))
    client = make_client("openrouter", model)
    with pytest.raises(ProviderBlockedError) as caught:
        await client.generate_async("prompt", Record)
    assert model.await_count == 1
    assert caught.value.context["reason"] == "spend_limit"


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"reasons": ["content policy"], "flagged_input": "related topic"},
        {"patterns": ["ignore previous instructions"]},
        {"limit_source": "unknown_source", "scope": "workspace", "resets_at": "tomorrow"},
        {"limit_source": ["malformed"]},
    ],
)
async def test_unknown_or_content_403_does_not_stop_or_retry_even_after_repetition(metadata):
    # A graph subtree may share blocked content; other branches can still work.
    error = api_error(403, metadata, "Content guardrail blocked network security topic")
    model = AsyncMock(side_effect=[error, error, '{"name": "unrelated branch"}'])
    client = make_client("openrouter", model)
    for _ in range(2):
        with pytest.raises(openai.PermissionDeniedError):
            await client.generate_async("related topic", Record)
    assert (await client.generate_async("other topic", Record)).name == "unrelated branch"
    assert model.await_count == 3


@pytest.mark.parametrize("recover", [False, True])
async def test_temporary_inflight_402_retries_then_stops_or_recovers(recover):
    metadata = {
        "limit_source": "openrouter_in_flight_budget",
        "reason": "in_flight_budget_exhausted",
    }
    error = api_error(402, metadata)
    effects = [error, '{"name": "accepted"}'] if recover else error
    model = AsyncMock(side_effect=effects)
    client = make_client("openrouter", model, retries=1)
    with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()) as sleep:
        if recover:
            assert (await client.generate_async("prompt", Record)).name == "accepted"
        else:
            with pytest.raises(ProviderBlockedError) as caught:
                await client.generate_async("prompt", Record)
            assert caught.value.context["reason"] == "temporary_budget_retries_exhausted"
    assert model.await_count == 2
    assert sleep.await_count == 1


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
@pytest.mark.parametrize("retries", [0, 2])
async def test_raw_429_uses_api_budget_then_latches_workflow_stop(provider, retries):
    model = AsyncMock(side_effect=api_error(429, message="Quota or rate limit reached"))
    client = make_client(provider, model, retries)
    with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ProviderBlockedError) as caught:
            await client.generate_async("prompt", Record)
    assert model.await_count == retries + 1
    assert caught.value.context["attempts"] == retries + 1
    with pytest.raises(ProviderBlockedError):
        await client.generate_async("another node", Record)
    assert model.await_count == retries + 1


def test_generic_openrouter_429_does_not_infer_daily_exhaustion():
    error = api_error(429, {"error_type": "rate_limit_exceeded"})
    failure = provider_failure(error, "openrouter")
    assert failure.immediate_reason is None
    assert failure.retryable_blocker


def test_inflight_source_is_sufficient_without_reason_or_header():
    error = api_error(402, {"limit_source": "openrouter_in_flight_budget"})
    error.response.headers.clear()
    failure = provider_failure(error, "openrouter")
    assert failure.retryable_blocker
    assert failure.immediate_reason is None


def test_ollama_outlines_wrapper_preserves_status_and_retry_after():
    wrapped = normalize_provider_exception(api_error(429), "openai")
    failure = provider_failure(wrapped, "ollama")
    assert failure.status == 429
    assert failure.retryable_blocker
    assert failure.retry_after == "0"
    client = make_client("ollama", AsyncMock())
    assert client.retry_handler.should_retry(wrapped)
    assert client.retry_handler.detector.is_rate_limit_error(wrapped, "ollama")
    assert client.retry_handler.detector.extract_quota_info(wrapped, "ollama").retry_after == 0


def test_openrouter_inflight_retry_after_uses_existing_delay_cap():
    error = api_error(
        402,
        {
            "limit_source": "openrouter_in_flight_budget",
            "reason": "in_flight_budget_exhausted",
        },
        retry_after="120",
    )
    client = make_client("openrouter", AsyncMock())
    assert client.retry_handler.calculate_delay(0, error) == 60
    client.retry_handler.config.respect_retry_after = False
    client.retry_handler.config.jitter = False
    assert client.retry_handler.calculate_delay(0, error) == 1


@pytest.mark.parametrize(
    "status,metadata,expected_calls",
    [
        (402, {"limit_source": "openrouter_key_limit"}, 1),
        (403, {"limit_source": "openrouter_workspace_budget"}, 1),
        (429, {}, 2),
        (
            402,
            {"limit_source": "openrouter_in_flight_budget", "reason": "in_flight_budget_exhausted"},
            2,
        ),
    ],
)
def test_openrouter_real_adapter_with_mock_http(monkeypatch, status, metadata, expected_calls):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            status,
            headers={"retry-after": "0"},
            json={
                "error": {
                    "code": status,
                    "message": "Router blocked this request",
                    "metadata": metadata,
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = LLMClient(
            "openrouter", "test", http_client=http_client, rate_limit_config={"max_retries": 1}
        )
        with patch("deepfabric.llm.retry_handler.time.sleep"):
            with pytest.raises(ProviderBlockedError, match="Router blocked this request"):
                client.generate("prompt", Record)
    assert len(requests) == expected_calls
    assert str(requests[0].url).startswith("https://openrouter.ai/")
