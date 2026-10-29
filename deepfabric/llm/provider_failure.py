"""Conservative workflow-blocking signals for the reviewed provider APIs."""

import json
import re

from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

import openai

from deepfabric.exceptions import DataSetGeneratorError, ProviderBlockedError

REVIEWED_PROVIDERS = frozenset({"openai", "anthropic", "gemini", "openrouter", "ollama"})
_OPENROUTER_BUDGET_SOURCES = frozenset(
    {"openrouter_guardrail_budget", "openrouter_workspace_budget", "openrouter_key_limit"}
)
_OPENAI_QUOTA_CODES = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "organization_usage_limit_exceeded",
    }
)
_ANTHROPIC_SPEND_PREFIXES = (
    "You have reached your specified API usage limits",
    "You have reached your specified workspace API usage limits",
)


@dataclass(frozen=True)
class ProviderFailure:
    """Observed provider fields, without attributing shared traffic or unknown usage."""

    provider: str
    status: int | None
    code: str | None
    error_type: str | None
    message: str
    retry_after: str | None
    immediate_reason: str | None
    retryable_blocker: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def blocked_error(self, attempts: int) -> ProviderBlockedError:
        reason = self.immediate_reason or (
            "temporary_budget_retries_exhausted"
            if self.status == HTTPStatus.PAYMENT_REQUIRED
            else "rate_limit_retries_exhausted"
        )
        status = f" HTTP {self.status}" if self.status is not None else ""
        code = f" ({self.code or self.error_type})" if self.code or self.error_type else ""
        retry = f" Retry-After: {self.retry_after}." if self.retry_after else ""
        details = (
            f" Details: {json.dumps(self.metadata, ensure_ascii=False)}."
            if self.provider == "openrouter" and self.metadata
            else ""
        )
        return ProviderBlockedError(
            f"Generation stopped: {self.provider}{status}{code}: {self.message}.{retry}{details} "
            "Check the provider error and resolve the issue before resuming.",
            context={
                "provider": self.provider,
                "status_code": self.status,
                "provider_code": self.code,
                "provider_error_type": self.error_type,
                "provider_message": self.message,
                "retry_after": self.retry_after,
                "reason": reason,
                "attempts": attempts,
                "provider_metadata": self.metadata,
            },
        )


def unwrap_provider_exception(exception: Exception) -> Exception:
    """Outlines preserves the SDK exception separately from its normalized error."""
    seen = set()
    while id(exception) not in seen:
        seen.add(id(exception))
        original = getattr(exception, "original_exception", None)
        if original is None and isinstance(exception, DataSetGeneratorError):
            original = exception.__cause__
        if not isinstance(original, Exception):
            break
        exception = original
    return exception


def provider_failure(exception: Exception, provider: str) -> ProviderFailure | None:
    """Read structured signals; an ambiguous 429 remains retryable, including Gemini's.

    Gemini spend/daily quota message signatures have not been verified. Do not
    infer immediate exhaustion from the words 'quota', 'per_day', or a zero limit.
    Non-429 request/content errors are not automatically workflow blockers.
    """
    if provider not in REVIEWED_PROVIDERS:
        return None
    exception = unwrap_provider_exception(exception)
    if provider == "ollama" and not isinstance(exception, openai.APIStatusError):
        return None
    body = getattr(exception, "body", None)
    if not isinstance(body, dict) and provider == "gemini":
        # google-genai stores the response JSON in .details, not .response_json.
        body = getattr(exception, "details", None)
    body = body if isinstance(body, dict) else {}
    error = body.get("error", body)
    error = error if isinstance(error, dict) else {}
    status = getattr(exception, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exception, "code", None)
    if not isinstance(status, int):
        status = error.get("code")
    if not isinstance(status, int):
        # Do not mistake an ordinary validation value such as input_value=429
        # for an HTTP error. SDK statuses remain the primary signal.
        status = (
            HTTPStatus.TOO_MANY_REQUESTS
            if re.search(r"\b(?:HTTP\s+429|429\s+RESOURCE_EXHAUSTED)\b", str(exception))
            else None
        )
    if status is None:
        return None
    code = error.get("code")
    code = code if isinstance(code, str) else None
    error_type = error.get("type", error.get("status"))
    error_type = error_type if isinstance(error_type, str) else None
    message = error.get("message")
    message = message if isinstance(message, str) else str(exception)
    details = error.get("details")
    metadata = error.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if provider == "anthropic" and isinstance(details, dict):
        detail_code = details.get("error_code")
        if isinstance(detail_code, str):
            code = detail_code

    reason = None
    retryable_blocker = status == HTTPStatus.TOO_MANY_REQUESTS
    if provider == "openrouter" and status == HTTPStatus.FORBIDDEN:
        # Content and moderation guardrails may affect only related topic nodes.
        # Missing scope/reset fields must not hide a recognized budget source.
        source = metadata.get("limit_source")
        if isinstance(source, str) and source in _OPENROUTER_BUDGET_SOURCES:
            reason = "spend_limit"
    elif provider == "openrouter" and status == HTTPStatus.PAYMENT_REQUIRED:
        if metadata.get("limit_source") == "openrouter_in_flight_budget":
            retryable_blocker = True
        else:
            # The documented temporary reservation case is the exception to
            # payment failures. Do not repeatedly submit an unaffordable call.
            reason = (
                "request_cost_exceeds_budget"
                if metadata.get("reason") == "weight_exceeds_budget"
                else "credit_budget"
            )
    elif status in (401, 403, 404):
        # These clients issue generation calls against one configured model and
        # credentials. A denial or missing configured resource blocks that run.
        reason = "provider_access"
    elif provider == "gemini" and status == HTTPStatus.BAD_REQUEST and isinstance(details, list):
        # Google's structured ErrorInfo reason distinguishes an invalid API key
        # from the many request-specific INVALID_ARGUMENT responses.
        if any(
            isinstance(detail, dict)
            and detail.get("@type") == "type.googleapis.com/google.rpc.ErrorInfo"
            and detail.get("reason") == "API_KEY_INVALID"
            for detail in details
        ):
            code = "API_KEY_INVALID"
            reason = "provider_access"
    elif (
        provider == "openai"
        and status == HTTPStatus.TOO_MANY_REQUESTS
        and (code in _OPENAI_QUOTA_CODES or error_type == "insufficient_quota")
    ):
        reason = "quota_exhausted"
    elif provider == "anthropic":
        if status == HTTPStatus.PAYMENT_REQUIRED:
            reason = "billing_error"
        elif (
            status == HTTPStatus.TOO_MANY_REQUESTS and code == "enforced_spend_limit_reached"
        ) or (
            status in (HTTPStatus.BAD_REQUEST, HTTPStatus.TOO_MANY_REQUESTS)
            and message.startswith(_ANTHROPIC_SPEND_PREFIXES)
        ):
            reason = "spend_limit"

    response = getattr(exception, "response", None)
    if response is None:
        response = getattr(exception, "http_response", None)
    headers = getattr(response, "headers", None)
    retry_after: Any = headers.get("retry-after") if headers is not None else None
    return ProviderFailure(
        provider,
        status,
        code,
        error_type,
        message,
        str(retry_after) if isinstance(retry_after, str | int | float) else None,
        reason,
        retryable_blocker,
        metadata,
    )
