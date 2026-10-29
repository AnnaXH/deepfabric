# Rate Limiting

DeepFabric includes intelligent retry handling for API rate limits across all LLM providers.

## Overview

The system provides:

- **Provider-aware defaults** for OpenAI, Anthropic, Gemini, Ollama, OpenRouter
- **Exponential backoff with jitter** to prevent thundering herd
- **Retry-after header support** when providers specify wait times
- **Workflow stops** for confirmed provider access, credit, or spending blockers

## Provider Defaults

Each provider has optimized defaults:

| Provider | Max Retries | Base Delay | Max Delay |
|----------|-------------|------------|-----------|
| OpenAI | 5 | 1.0s | 60s |
| Anthropic | 5 | 1.0s | 60s |
| Gemini | 5 | 2.0s | 120s |
| Ollama | 2 | 0.5s | 5s |
| OpenRouter | 5 | 1.0s | 60s |

## Configuration

### Using Defaults

Omit rate limiting config to use provider defaults:

```yaml title="config.yaml"
generation:
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash-exp"
  # Rate limiting uses defaults automatically
```

### Custom Configuration

```yaml title="config.yaml"
generation:
  llm:
    provider: "gemini"
    model: "gemini-2.0-flash-exp"

  rate_limit:
    max_retries: 7
    base_delay: 3.0
    max_delay: 180.0
    backoff_strategy: "exponential_jitter"
    exponential_base: 2.0
    jitter: true
    respect_retry_after: true
```

## Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_retries` | int | 5 | Maximum retry attempts (0-20) |
| `base_delay` | float | 1.0 | Initial delay in seconds (0.1-60) |
| `max_delay` | float | 60.0 | Maximum delay cap in seconds (1-300) |
| `backoff_strategy` | string | exponential_jitter | See strategies below |
| `exponential_base` | float | 2.0 | Multiplier for exponential backoff (1.1-10) |
| `jitter` | bool | true | Add randomization to prevent synchronized retries |
| `respect_retry_after` | bool | true | Honor server-specified wait times |

### Backoff Strategies

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `exponential_jitter` | `delay = base * (exp_base ^ attempt) +/- 25%` | Recommended default |
| `exponential` | `delay = base * (exp_base ^ attempt)` | Predictable timing |
| `linear` | `delay = base * attempt` | Gentle increase |
| `constant` | Always use `base_delay` | Fixed intervals |

## Provider-Specific Behavior

=== "OpenAI"

    Monitors `x-ratelimit-*` headers and respects `retry-after`:

    ```yaml title="config.yaml"
    rate_limit:
      max_retries: 5
      respect_retry_after: true
    ```

=== "Anthropic"

    Uses token bucket algorithm with RPM/ITPM/OTPM limits:

    ```yaml title="config.yaml"
    rate_limit:
      max_retries: 5
      base_delay: 1.0
    ```

=== "Gemini"

    Uses bounded retries for `429` errors, including ambiguous quota errors:

    ```yaml title="config.yaml"
    rate_limit:
      max_retries: 5
      base_delay: 2.0      # Higher default
      max_delay: 120.0     # Maximum backoff delay
    ```

    !!! info "Quota classification"
        Gemini quota/spend message fragments are not treated as verified immediate-stop signals. A `429` uses the configured API retry budget, then stops the workflow if it continues. This includes daily quota and rolling spend-window errors when no verified distinguishing signal is available.

=== "Ollama"

    Minimal retries for local deployment:

    ```yaml title="config.yaml"
    rate_limit:
      max_retries: 2
      base_delay: 0.5
      max_delay: 5.0
    ```

## Python API

```python title="rate_limit_example.py"
from deepfabric import DataSetGenerator

generator = DataSetGenerator(
    generation_system_prompt="You are a helpful assistant.",
    provider="gemini",
    model_name="gemini-2.0-flash-exp",
    rate_limit={
        "max_retries": 7,
        "base_delay": 3.0,
        "max_delay": 180.0,
        "backoff_strategy": "exponential_jitter",
    }
)
```

## Retry Behavior

!!! success "Retries On"
    - `429` (rate limit)
    - `500`, `502`, `503`, `504` (server errors)
    - Timeout, connection, network errors

!!! failure "Does NOT Retry"
    - `4xx` errors (except 429)
    - Authentication failures
    - Confirmed OpenAI credit/spend/usage exhaustion
    - Confirmed Anthropic spend limits or billing failures

### Workflow stops and provider coverage

`generation.rate_limit.max_retries` controls retries of one API request: the initial
attempt plus that many retries. It is separate from `sample_retries`, which handles
invalid generated output, and the deprecated generation-level `max_retries`.
For OpenAI, Anthropic, Gemini, OpenRouter, and Ollama, SDK-internal retries are
disabled so this API retry budget is not multiplied by a second retry layer.

Confirmed OpenAI credit/spend/usage errors and Anthropic spend/billing errors stop
generation immediately. Authentication/access errors or a missing configured model
also stop the run. A potentially temporary or ambiguous `429` retries with backoff;
if it still fails after the configured budget, the workflow stops rather than
submitting the remaining sample slots. Request-specific errors and invalid output
are classified separately and do not automatically stop the entire run.

The error shows the provider's status, code/message, and available retry details.
DeepFabric cannot determine whether its traffic or another workload caused a shared
rate limit; check the provider error before changing settings or resuming.

Accepted samples are retained, and unfinished slots remain pending. With checkpointing
enabled, rerun the same generation command with `--resume` after resolving the issue;
these pending slots do not require `--retry-failed`. In Python, call
`load_checkpoint()` before generating again with the same plan. Without checkpointing,
partial results are exported to `error_dataset.jsonl`, but checkpoint resume is unavailable.
The CLI exits with status 1, and an enabled run report records `stopped`, not `completed`.

OpenRouter stops immediately on non-temporary 402 credit/budget failures and 403s
with a recognized budget `limit_source`. A request whose estimated cost exceeds
its available budget is reported as a cost/budget failure, not a model context-limit
error. The original provider message and metadata are shown when stopping.

OpenRouter 402s with `limit_source: openrouter_in_flight_budget` are potentially
temporary: they use bounded API retries and stop only if those retries are exhausted.
A generic 429 follows the same retry-then-stop policy; DeepFabric does not infer daily
free-model exhaustion from status or remaining-count headers alone. Moderation,
content guardrails, and unclassified 403s remain sample failures; repeated blocks
on related topic nodes do not establish a shared workflow blocker.

Ollama's OpenAI-SDK errors are recognized through the Outlines wrapper. A 429
uses bounded retries, then stops the workflow; 401/403/404 access failures stop
immediately. Both sync and async clients disable SDK-internal retries so
DeepFabric owns the request retry budget. Generic request errors remain sample
failures. No cloud-specific immediate-exhaustion signature or account monitoring
calls were added.

### Retry-After handling and long reset windows

The reviewed providers currently differ in how DeepFabric uses `Retry-After`:

| Provider | Retry delay behavior |
| --- | --- |
| OpenAI / Anthropic | With `respect_retry_after: true` (default), a numeric header sets the per-request delay, capped at `max_delay`. Otherwise, configured backoff is used. HTTP-date header values are not parsed. |
| Gemini | Uses configured backoff; the current quota parser does not use `Retry-After` or structured retry-delay details to schedule retries. A captured header can still appear in the final provider error. |
| OpenRouter | Numeric `Retry-After` on retryable 429s or temporary in-flight-budget 402s sets the delay when enabled, capped at `max_delay`. Otherwise configured backoff is used; HTTP-date values are not parsed. |
| Ollama | Numeric headers are read from the SDK error preserved by Outlines and set the delay when enabled, capped at `max_delay`. Otherwise configured backoff is used; HTTP-date values are not parsed. |

The live TUI events panel shows the retry attempt and actual wait, such as
`openai retry (attempt 1), backoff 30.0s`. It does not identify whether the delay
came from the header or configured backoff, or whether `max_delay` shortened it.
Run reports record the actual wait but not that source; the header-use decision
is logged at debug level. Waiting is per request, not a global pause of other workers.

A long provider reset window, such as an exhausted weekly allowance, is a reason
to stop and resume later rather than keep the invocation waiting. `Retry-After`
alone does not identify a weekly quota; inspect the provider's error and reset
details. Confirmed spending blockers already stop immediately even if a header
is present.

For an otherwise ambiguous error, **automatic stopping based on a long retry
delay is not yet implemented**. Currently, a numeric header longer than `max_delay`
is shortened to that cap, so requests may retry before the provider's indicated
wait has elapsed and stop only after the configured retry budget. `max_delay` is
a backoff cap, not a workflow waiting deadline. A follow-up should stop with the
provider's delay/reset information when the required wait is unsuitable for the
current invocation, preserving the checkpoint for later resume instead of
shortening that required wait and sending predictably premature retries.

## Best Practices

=== "High Volume (Paid Tier)"

    ```yaml title="Paid tier config"
    rate_limit:
      max_retries: 3
      base_delay: 0.5
      max_delay: 10.0
    ```

=== "Free Tier (Aggressive Limits)"

    ```yaml title="Free tier config"
    rate_limit:
      max_retries: 10
      base_delay: 5.0
      max_delay: 300.0
    ```

=== "Combined with Batch Size"

    ```yaml title="Batch size optimization"
    output:
      batch_size: 2    # Smaller batches reduce rate limit pressure
      num_samples: 20

    generation:
      rate_limit:
        max_retries: 5
        base_delay: 2.0
    ```

## Troubleshooting

### Still Hitting Rate Limits

!!! tip "Solutions"
    1. Reduce `batch_size` in dataset creation
    2. Increase `base_delay`
    3. Check your provider tier/quota

### Daily Quota Exhausted (Gemini)

Gemini returns `429 RESOURCE_EXHAUSTED` for several different limits. DeepFabric
uses the configured API retry budget, then stops the workflow if the error continues.
It does not infer daily exhaustion from a generic quota message alone.

!!! info "Options"
    - Wait until midnight Pacific time
    - Upgrade Gemini tier
    - Switch providers temporarily

### Too Many Retries

Reduce `max_retries` to fail faster:

```yaml
rate_limit:
  max_retries: 3
```
