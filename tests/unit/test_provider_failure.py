"""Workflow classification uses SDK fields rather than broad quota word matches."""
# ruff: noqa: PLR2004, ARG002, SIM117

from unittest.mock import AsyncMock, Mock, patch

import anthropic
import httpx
import openai
import pytest

from google.genai.errors import ClientError
from pydantic import BaseModel

from deepfabric.exceptions import ProviderBlockedError
from deepfabric.llm.client import LLMClient
from deepfabric.llm.provider_failure import provider_failure
from deepfabric.utils import is_validation_error


class Record(BaseModel):
    name: str


def sdk_error(
    provider,
    status,
    *,
    code=None,
    error_type=None,
    message="Provider rejected request",
    details=None,
):
    error = {"message": message}
    if code:
        error["code"] = code
    if error_type:
        error["type"] = error_type
    if details:
        error["details"] = details
    if provider == "gemini":
        return ClientError(
            status,
            {
                "error": {
                    **error,
                    "code": status,
                    "status": "RESOURCE_EXHAUSTED" if status == 429 else "INVALID_ARGUMENT",
                }
            },
        )
    response = httpx.Response(
        status,
        headers={"retry-after": "0"},
        request=httpx.Request("POST", "https://provider.invalid"),
    )
    sdk = openai if provider == "openai" else anthropic
    classes = {
        400: sdk.BadRequestError,
        401: sdk.AuthenticationError,
        403: sdk.PermissionDeniedError,
        404: sdk.NotFoundError,
        429: sdk.RateLimitError,
    }
    return classes.get(status, sdk.APIStatusError)(message, response=response, body=error)


def make_client(provider, model, retries=2):
    with (
        patch("deepfabric.llm.client.make_outlines_model", return_value=Mock()),
        patch("deepfabric.llm.client.make_async_outlines_model", return_value=model),
    ):
        client = LLMClient(provider, "test", rate_limit_config={"max_retries": retries})
        assert client.async_model is model
    return client


@pytest.mark.parametrize(
    "code",
    [
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    ],
)
async def test_openai_quota_stops_without_retry_and_latches_peer_calls(code):
    model = AsyncMock(side_effect=sdk_error("openai", 429, code=code))
    client = make_client("openai", model)
    for _ in range(2):
        with pytest.raises(ProviderBlockedError) as caught:
            await client.generate_async("prompt", Record)
    assert model.await_count == 1
    assert caught.value.context["provider_code"] == code
    assert caught.value.context["attempts"] == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("status", [401, 403, 404])
async def test_configured_provider_access_failure_stops_immediately(provider, status):
    model = AsyncMock(side_effect=sdk_error(provider, status))
    client = make_client(provider, model)
    with pytest.raises(ProviderBlockedError) as caught:
        await client.generate_async("prompt", Record)
    assert model.await_count == 1
    assert caught.value.context["reason"] == "provider_access"


@pytest.mark.parametrize(
    "status,details,message",
    [
        (429, {"error_code": "enforced_spend_limit_reached"}, "API usage threshold reached"),
        (400, None, "You have reached your specified API usage limits. Access resumes tomorrow."),
        (
            429,
            None,
            "You have reached your specified workspace API usage limits. Access resumes tomorrow.",
        ),
        (
            400,
            None,
            "You have reached your specified workspace API usage limits. Access resumes tomorrow.",
        ),
        (402, None, "Billing payment failed"),
    ],
)
async def test_anthropic_spend_and_billing_errors_stop_immediately(status, details, message):
    model = AsyncMock(side_effect=sdk_error("anthropic", status, details=details, message=message))
    client = make_client("anthropic", model)
    with pytest.raises(ProviderBlockedError, match=message):
        await client.generate_async("prompt", Record)
    assert model.await_count == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("retries", [0, 2])
async def test_ambiguous_429_stops_only_after_configured_api_retries(provider, retries):
    # The word 'quota' is not sufficient evidence of persistent exhaustion.
    model = AsyncMock(side_effect=sdk_error(provider, 429, message="Temporary quota throttle"))
    client = make_client(provider, model, retries)
    with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()) as sleep:
        with pytest.raises(ProviderBlockedError) as caught:
            await client.generate_async("prompt", Record)
    assert model.await_count == retries + 1
    assert sleep.await_count == retries
    assert caught.value.context["reason"] == "rate_limit_retries_exhausted"


@pytest.mark.parametrize(
    "message",
    [
        "Your project has exceeded its monthly spending cap.",
        "Your prepaid funds are depleted.",
        "quotaMetric generate_requests_per_model_per_day, limit: 0",
        "Rolling 10-minute spend limit exceeded",
    ],
)
async def test_unverified_gemini_signatures_do_not_skip_retry_budget(message):
    model = AsyncMock(side_effect=sdk_error("gemini", 429, message=message))
    client = make_client("gemini", model, retries=1)
    with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(ProviderBlockedError):
            await client.generate_async("prompt", Record)
    assert model.await_count == 2


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
async def test_transient_429_recovers_without_stopping(provider):
    model = AsyncMock(side_effect=[sdk_error(provider, 429), '{"name": "accepted"}'])
    client = make_client(provider, model)
    with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()):
        record = await client.generate_async("prompt", Record)
    assert record.name == "accepted"
    client.retry_handler.raise_if_stopped()
    assert model.await_count == 2


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("status", [408, 409, 500, 529])
async def test_sdk_transient_statuses_keep_retry_coverage_without_workflow_stop(provider, status):
    error = sdk_error(provider, status, message="Provider rejected request")
    model = AsyncMock(side_effect=error)
    client = make_client(provider, model, retries=1)
    with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(type(error)):
            await client.generate_async("prompt", Record)
    assert model.await_count == 2
    client.retry_handler.raise_if_stopped()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
async def test_request_specific_400_does_not_latch_workflow_stop(provider):
    error = sdk_error(
        provider, 400, error_type="invalid_request_error", message="Invalid input schema"
    )
    model = AsyncMock(side_effect=[error, '{"name": "accepted"}'])
    client = make_client(provider, model)
    with pytest.raises(type(error)):
        await client.generate_async("bad prompt", Record)
    assert (await client.generate_async("good prompt", Record)).name == "accepted"


def test_sync_retry_path_uses_same_stop_contract():
    model = Mock(side_effect=sdk_error("openai", 429))
    with patch("deepfabric.llm.client.make_outlines_model", return_value=model):
        client = LLMClient("openai", "test", rate_limit_config={"max_retries": 1})
    with patch("deepfabric.llm.retry_handler.time.sleep"):
        with pytest.raises(ProviderBlockedError):
            client.generate("prompt", Record)
    assert model.call_count == 2
    with pytest.raises(ProviderBlockedError):
        client.generate("prompt", Record)
    assert model.call_count == 2


def test_openai_structured_type_does_not_imply_other_provider_quota():
    error = sdk_error("openai", 429, error_type="insufficient_quota")
    error.body = {"error": error.body}
    assert provider_failure(error, "openai").immediate_reason == "quota_exhausted"
    assert provider_failure(error, "openrouter").immediate_reason is None
    assert provider_failure(error, "ollama").immediate_reason is None


def test_generic_gemini_429_is_retryable_without_resource_exhausted_text():
    client = make_client("gemini", AsyncMock())
    assert client.retry_handler.should_retry(Exception("HTTP 429"))


async def test_gemini_structured_invalid_api_key_is_a_shared_400_blocker():
    error = sdk_error(
        "gemini",
        400,
        details=[
            {
                "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                "reason": "API_KEY_INVALID",
                "domain": "googleapis.com",
            }
        ],
    )
    model = AsyncMock(side_effect=error)
    client = make_client("gemini", model)
    with pytest.raises(ProviderBlockedError) as caught:
        await client.generate_async("prompt", Record)
    assert caught.value.context["provider_code"] == "API_KEY_INVALID"
    assert model.await_count == 1


def test_workflow_error_is_never_retried_as_validation():
    assert not is_validation_error(ProviderBlockedError("invalid schema in provider error"))


def test_429_in_validation_content_is_not_a_provider_rate_limit():
    error = ValueError("validation error: expected string, input_value=429")
    assert provider_failure(error, "openai") is None
    assert provider_failure(error, "gemini") is None


def test_real_openai_transport_does_not_retry_confirmed_exhaustion(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "credit_balance_exhausted",
                    "type": "insufficient_quota",
                    "message": "Credit balance exhausted",
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = LLMClient(
            "openai", "test", http_client=http_client, rate_limit_config={"max_retries": 3}
        )
        with pytest.raises(ProviderBlockedError):
            client.generate("prompt", Record)
    assert len(requests) == 1
    assert len(client.token_usage.snapshot()["records"]) == 1


@pytest.mark.parametrize("status", [400, 429])
async def test_real_anthropic_transport_stops_for_spend_caps(monkeypatch, status):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "message": (
                        "You have reached your specified API usage limits"
                        if status == 400
                        else "Monthly usage threshold reached"
                    ),
                    "details": {"error_code": "enforced_spend_limit_reached"},
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        client = LLMClient(
            "anthropic", "test", http_client=http_client, rate_limit_config={"max_retries": 3}
        )
        with pytest.raises(ProviderBlockedError):
            await client.generate_async("prompt", Record)
    assert len(requests) == 1


def test_gemini_factory_disables_sdk_retries_for_outer_budget(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    with patch("deepfabric.llm.client.genai.Client") as factory:
        client = LLMClient("gemini", "test")
        assert client.async_model is not None
    assert factory.call_count == 2
    for call in factory.call_args_list:
        assert call.kwargs["http_options"].retry_options.attempts == 1
