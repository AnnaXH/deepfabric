"""Exercise Ollama retries through real SDK/Outlines adapters with mocked HTTP."""
# ruff: noqa: PLR2004

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from outlines.exceptions import BadRequestError
from pydantic import BaseModel

from deepfabric.exceptions import ProviderBlockedError
from deepfabric.llm.client import LLMClient


class Record(BaseModel):
    name: str


def response(status):
    if status != 200:
        return httpx.Response(
            status,
            headers={"retry-after": "7"},
            json={"error": {"message": "Ollama rejected request", "code": "test_error"}},
        )
    return httpx.Response(
        200,
        json={
            "id": "test",
            "object": "chat.completion",
            "created": 0,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"name": "accepted"}'},
                    "finish_reason": "stop",
                }
            ],
        },
    )


@pytest.mark.parametrize("retries", [0, 2])
@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_sync_real_adapter_stops_with_one_retry_owner(status, retries):
    requests = []

    def respond(request):
        requests.append(request)
        return response(status)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = LLMClient(
            "ollama",
            "test",
            http_client=http_client,
            rate_limit_config={"max_retries": retries, "max_delay": 10},
        )
        with patch("deepfabric.llm.retry_handler.time.sleep") as sleep:
            with pytest.raises(ProviderBlockedError, match="Ollama rejected request") as caught:
                client.generate("prompt", Record)
            with pytest.raises(ProviderBlockedError):
                client.generate("another node", Record)
    attempts = retries + 1 if status == 429 else 1
    assert len(requests) == attempts
    assert sleep.call_count == attempts - 1
    if sleep.call_count:
        sleep.assert_called_with(7)
    assert caught.value.context["attempts"] == attempts
    assert caught.value.context["status_code"] == status
    assert caught.value.context["provider_code"] == "test_error"
    assert caught.value.context["retry_after"] == "7"
    assert str(requests[0].url) == "http://localhost:11434/v1/chat/completions"


@pytest.mark.parametrize("recover", [False, True])
async def test_async_real_adapter_retries_then_stops_or_recovers(recover):
    requests = []

    def respond(request):
        requests.append(request)
        return response(200 if recover and len(requests) > 1 else 429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        with patch("deepfabric.llm.client.make_outlines_model", return_value=Mock()):
            client = LLMClient(
                "ollama",
                "test",
                http_client=http_client,
                rate_limit_config={"max_retries": 1, "max_delay": 10},
            )
        with patch("deepfabric.llm.retry_handler.asyncio.sleep", new=AsyncMock()) as sleep:
            if recover:
                assert (await client.generate_async("prompt", Record)).name == "accepted"
                assert (await client.generate_async("next node", Record)).name == "accepted"
            else:
                with pytest.raises(ProviderBlockedError, match="Ollama rejected request"):
                    await client.generate_async("prompt", Record)
                with pytest.raises(ProviderBlockedError):
                    await client.generate_async("next node", Record)
    assert len(requests) == (3 if recover else 2)
    sleep.assert_awaited_once_with(7)


async def test_wrapped_request_error_does_not_latch_stop():
    requests = []

    def respond(request):
        requests.append(request)
        return response(400 if len(requests) == 1 else 200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
        with patch("deepfabric.llm.client.make_outlines_model", return_value=Mock()):
            client = LLMClient("ollama", "test", http_client=http_client)
        with pytest.raises(BadRequestError):
            await client.generate_async("bad request", Record)
        assert (await client.generate_async("other topic", Record)).name == "accepted"
    assert len(requests) == 2
