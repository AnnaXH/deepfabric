# Phase 2 implementation — decisions not covered by the spec

Companion to `deepfabric-verbalize-spec.md` (design of record) and
`deepfabric-verbalize-phase2-implementation-note.md` (task scope). Everything
below is a choice I had to make because the spec left it open, plus the things I
deliberately did **not** build.

Scope built: **§4** (`deepfabric/verbalized/`) and **§6** (expression diversity at
`SingleShotBuilder.generate`). No Phase 1, no Phase 3, no changes to
`graph.py` / `tree.py`.

---

## 1. Package layout

| Decision | Why |
|---|---|
| Config models live in `verbalized/config.py` (spec §4 lists only `sampler.py`, `selection.py`, `schemas.py`, `similarity.py`) | Keeps the package self-contained and importable on its own. `deepfabric/config.py` now imports *from* the package, so putting the models there would have been circular. |
| Added a `DiversityConfig` wrapper holding `verbalized` | Gives the YAML the shape §4.1/§9 specify: `<stage>.diversity.verbalized`, attachable per seam. |

Files added: `config.py`, `schemas.py`, `selection.py`, `similarity.py`,
`sampler.py`, `__init__.py`.

## 2. Candidate schemas (§4.2)

- **Dropped `Field(ge=0, le=1)` from the LLM-facing `probability`.** OpenAI and
  Anthropic strict structured-output schemas reject `minimum`/`maximum`
  keywords, and a model that answers `1.4` should cost us one number, not the
  whole candidate set. A `mode="before"` validator clamps into `[0, 1]` instead,
  and converts percentage strings (`"40%"` → `0.4`). This is consistent with the
  spec's "never treat verbalised probabilities as calibrated".
- **Added `distribution_model(payload_model)`**, a cached factory building a
  *typed* distribution schema (`payload: Conversation`, etc.). The spec's
  `payload: Any` shape cannot be sent through constrained decoding, so the
  generic `VerbalizedCandidate` / `CandidateDistribution` are kept for internal
  use and `to_candidates()` converts a typed instance back to them.

## 3. Selection (§4.3)

- **Strategies return a full ranking, not just the winners.** `rank_candidates()`
  returns `Ranking(ordered, eligible)`, where `eligible` counts the leading
  entries that actually satisfy the strategy. The sampler walks the ranking and
  skips near-duplicates, so dedup has backfill headroom without escaping the
  strategy's intent (a `threshold` run never backfills with a candidate above the
  threshold). `select_candidates()` is the plain spec-shaped API on top.
- **`tail_weighted` uses both knobs.** The spec offered `p^(1/T)` *or* `(1 − p)`;
  I use both, for different controls: `temperature` flattens
  (`w ∝ p^(1/T)`), and `tune_diversity ∈ [0,1]` blends that flattened
  distribution toward its inversion. `tune_diversity` also overrides the
  temperature (`T = 1 + tune·4`), so the single dial behaves as §4.1 describes.
- Weighted sampling without replacement uses the Efraimidis–Spirakis key method
  (`key = u^(1/w)`, sort descending) — one pass yields the whole ranking.
- **`top_p_complement` needed a parameter the spec doesn't name**: added
  `top_p` (default `0.9`) for the probability core that gets dropped. The
  complement is ordered by *descending* probability, i.e. the most plausible
  non-modal candidate first — off-mode without reaching for the least coherent
  candidate.
- `threshold` orders qualifying candidates rarest-first; when nothing qualifies,
  the median-probability candidate is kept (§4.3's ≥1 rule).
- **Added `seed`** so tail-weighted selection is reproducible in tests and across
  baseline/enabled comparison runs.

## 4. Similarity and dedup (§4.4) — provisional metric, fixed interface

The metric is behind `SimilarityMeasure.similarity()` / `.is_duplicate()` exactly
as the note asks; swapping it should not touch `sampler.py`.

- **The interface is async, with an optional `prepare(texts)` batch hook.**
  Embeddings are network-bound and are best fetched once per candidate set;
  `prepare` embeds the whole set in one call, `is_duplicate` is implemented once
  on the base class.
- **Two thresholds, not one.** `max_pairwise_cosine` (0.85, spec) governs the
  embedding measure; I added `max_lexical_similarity` (0.8) for the ROUGE-L
  fallback, because a cosine and an F1 are not on the same scale and reusing one
  number would silently mis-tune the fallback.
- **Fallback is two-tier.** No embedding provider (`OPENAI_API_KEY` unset) →
  lexical from the start with a warning, per §4.1. An embedding call that fails
  mid-run → that measure degrades to lexical for the rest of the run with a
  warning, rather than failing the generation.
- **Measures are cached module-level, keyed by the serialised `DedupConfig`.**
  Builders are constructed per sample; without the cache each sample would build
  a new HTTP client and throw away the embedding cache.
- `semantic_diversity(texts, measure)` is metric-agnostic: `1 − mean pairwise
  similarity`, clipped at 0. Backed by embeddings that is the paper's measure;
  backed by the fallback it is the ROUGE-L analogue. `distinct_n` and `rouge_l`
  are exported for the offline (evaluation) use.

## 5. Phase 2 wiring (§6)

- **Fan-out is inside `SingleShotBuilder.generate`**, returning
  `list[Conversation]` when enabled (as the note resolved). The abstract
  `ConversationBuilder.generate` signature widened to
  `Conversation | list[Conversation]`; every other builder still returns one.
- **The sizing mode is applied where node-slot grouping still exists.** The note
  says `process_topic` / `_process_batch_with_retries_async` applies it, but by
  the time `_process_batch_with_retries_async` sees results they are a flat list
  with no per-prompt grouping. So `DataSetGenerator._apply_sizing()` (a generator
  method, as intended) is called from the per-prompt result loop inside
  `_generate_structured_samples_async`, which is the function
  `_process_batch_with_retries_async` delegates to.
- **`sizing: budget | multiplier` on the verbalized block, default `budget`.**
  Budget commits one sample per node-slot, holding dataset totals (the note's
  decision). Multiplier is implemented too — §6.2 describes both, the plumbing is
  identical, and it is the only way to actually consume `select > 1`. Disabled
  and budget runs are unaffected by its existence.
- **Sample-accounting check relaxed from `!=` to `<`.** In multiplier mode one
  prompt legitimately produces several samples; only a shortfall indicates a
  silent drop.
- **Token budget for the distribution call = stage `max_tokens × k`, capped at
  32 000**, overridable with `verbalized.max_tokens`. One single-pass call now
  has to carry `k` complete conversations; leaving it at the per-sample budget
  would truncate the response.
- **Candidate-level validation failures are not fatal.** A candidate whose
  conversation does not start with a `user` message is dropped and the rest are
  kept; only an empty survivor set raises. That error is phrased
  `"Validation error: ..."` so the existing `sample_retries` path retries it —
  the repo detects retryable errors by substring
  (`deepfabric/utils.py:VALIDATION_ERROR_INDICATORS`).
- **`keep_probabilities` (default true)** records
  `metadata.verbalized = {approach, probability, normalized_probability, rank}`.
  Spec §10.6 leaves this open; recording by default preserves the signal for
  later diversity-tuning work, and it is one flag to turn off. Only ever present
  when the feature is enabled, so disabled output is unchanged.
- **Tool-execution validation extracted** to `DataSetGenerator._check_tool_executions()`
  so it can run per conversation over a list. Behaviour is unchanged for the
  agent path.
- **The agent path logs a warning** when `generation.diversity.verbalized.enabled`
  is set: expression diversity is the single-shot seam only, and tool-value
  diversity is Phase 3. Better than silently ignoring the flag.
- **Custom `output_schema` mode is untouched**: it bypasses the conversation
  builder entirely, so verbalized sampling does not apply there.
- **Realised diversity per batch** is emitted twice: a
  `progress_reporter.emit_step_complete("Verbalized sampling", ...)` event
  carrying `generated / selected / duplicates_dropped / invalid / diversity /
  measure`, and a `metrics.trace("verbalized_batch", ...)` telemetry event.
  `metrics.py` itself needed no change — its `trace` helper was sufficient.
- **Prompts** (`prompts.py`, `VerbalizedPromptBuilder`): the distribution
  instructions are *appended after* the existing single-sample prompt, so every
  requirement it states (system prompt, topic path, retry feedback, "start with a
  user message") still binds each candidate. Per-style guidance exists for
  `basic`, `cot-freetext` and `cot-agent`; the guidance pushes variation in who
  is asking, what is asked, how it is asked, and the shape of the answer, rather
  than rewording.

## 6. Tests

- **Offline (run, all passing):** `tests/unit/test_verbalized.py` (59),
  `tests/unit/test_verbalized_generation.py` (28), plus 5 new config tests in
  `tests/unit/test_config.py`. Full unit suite: **686 passed** (595 before this
  work).
- **API-dependent (written, never executed):**
  `tests/integration/test_verbalized_integration.py` — 10 tests covering
  embedding-backed similarity and end-to-end diversity/non-regression. They use
  the repo's existing mechanism for credentialed tests (`@pytest.mark.openai` +
  the `requires_openai` skipif from `tests/integration/conftest.py`), so they are
  outside the offline run (`make test-unit` targets `tests/unit` only) and skip
  without a key. Verified by collection only (10 collected), not by running.
- **Offline tests cannot reach an embedding API even on a machine with a key**:
  an autouse fixture deletes `OPENAI_API_KEY` and clears the measure cache, so
  dedup always resolves to the lexical fallback.
- Assertions are directional (diversity up, validity not down); no magic
  magnitudes anywhere.

## 7. Deliberately not built

- **Phase 1 (§5) and Phase 3 (§7)** — out of scope per the note.
- **`expansion: two_pass`** — remains a valid config value; enabling it raises
  `verbalized.expansion='two_pass' is not yet supported` at config validation
  (fails fast, before any LLM call). The sampler carries the same guard.
- **Docs (§9): `docs/`, README config section, `mkdocs.yml`, example YAML.** The
  note restricts edits to the files §4 and §6 name, which excludes docs. Worth
  doing before the feature is announced — say the word and I will add a
  task-oriented page plus a runnable example config.
- **§6.4 criterion 2 numbers for the PR** (measured diversity rise, baseline vs
  enabled) need API access we do not have yet.
  `test_diversity_rises_without_a_validity_regression` is written to produce
  exactly that comparison once a key is provisioned.

## 8. Environment note (not a code change)

The project's `.venv` pointed at a conda interpreter that no longer exists on
this machine (`.../miniconda3/envs/py13`), so nothing could run. I rebuilt it on
`/usr/local/bin/python3.12` with `uv sync --extra dev`. `uv`'s managed-Python
download is blocked by the sandbox, so `uv venv --python 3.13` will fail here;
3.12 is what the lockfile resolves cleanly against.

## 9. Config surface added

```yaml
generation:
  diversity:
    verbalized:
      enabled: false          # master switch for this seam (default false)
      k: 5                    # candidates the model verbalises
      select: 3               # how many to keep (<= k, validated)
      selection: tail_weighted # tail_weighted | threshold | top_p_complement
      prob_threshold: 0.15    # selection = threshold
      top_p: 0.9              # selection = top_p_complement (added)
      temperature: 1.0        # flattening temperature over verbalised probs
      tune_diversity: null    # optional 0..1; higher = more tail bias
      expansion: single_pass  # two_pass raises "not yet supported"
      sizing: budget          # budget (hold totals) | multiplier (added)
      max_tokens: null        # defaults to stage max_tokens x k, capped (added)
      keep_probabilities: true # record approach/probability in metadata (added)
      seed: null              # reproducible tail-weighted selection (added)
      dedup:
        enabled: true
        embedding_model: text-embedding-3-small   # provisional
        embedding_provider: openai                # added
        max_pairwise_cosine: 0.85                 # provisional
        max_lexical_similarity: 0.8               # fallback threshold (added)
```
