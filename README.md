# llm-context-bench

Reproducible long-context quality and speed benchmarks for OpenAI-compatible
inference servers — `llama.cpp`, `vllm serve`, SGLang, or a hosted endpoint.
The repository locks the prompts, sampling parameters, output limits, scoring,
warm-up, and validation rules so repeated runs compare the same workload, and it
calculates every speed number itself from the server's token counts and its own
clock, so a rate means the same thing on every engine.

Two profiles are included:

- **regular** — document extraction, calculation, policy application, summarization,
  and long-context retrieval over public-domain documents.
- **coding** — implementation, debugging, SQL, code analysis, and long-context
  retrieval over real open-source code.

Both profiles provide selectable **16K, 32K, 64K, and 128K input tiers**. The
performance lane always requests exactly **1,024 output tokens**.

## Why this benchmark is strict

Artificial output such as thousands of dots can make speculative decoding look
far faster than normal generation. This runner therefore:

- uses coherent generation tasks instead of forced repeated output;
- never sets `ignore_eos` for a measured performance request;
- rejects low-entropy or heavily repeated generated text;
- requires exactly 1,024 server-reported output tokens;
- rejects prompt-cache reuse wherever the engine reports it;
- measures prefill and decode throughput from its own timestamps and the
  reported token counts instead of trusting engine-internal rates;
- excludes invalid trials from reported performance medians and records the
  reason each trial was excluded.

The bundled input fixtures are deterministic assemblies of public-domain
documents and permissively-licensed open-source code; each fixture names its
sources. Their gzip ratios are checked to prevent highly repetitive filler, and
their SHA-256 hashes are locked in the suite manifests.

## Install

Python 3.10 or newer and a running OpenAI-compatible server are required.

```bash
cd llm-context-bench
python3 -m pip install -e .
```

No runtime Python dependencies are required.

## Run

Run both profiles and all sizes:

```bash
llm-context-bench \
  --base-url http://127.0.0.1:8080 \
  --model qwen-model \
  --profile qwen-build-a \
  --suite all \
  --lane all \
  --sizes all \
  --output results/qwen-build-a.json \
  --command 'CUDA_VISIBLE_DEVICES=0 llama-server --model model.gguf --ctx-size 131072' \
  --system '1x GPU; Ubuntu 24.04; CUDA 13.0; llama.cpp b7000'
```

Run only regular performance at selected sizes:

```bash
llm-context-bench \
  --base-url http://127.0.0.1:8080 \
  --model qwen-model \
  --profile qwen-regular-check \
  --suite regular \
  --lane performance \
  --sizes 16k 64k 128k \
  --output results/qwen-regular-check.json \
  --command 'ROCR_VISIBLE_DEVICES=0,1 llama-server --model model.gguf --ctx-size 131072' \
  --system '2x GPU; Ubuntu 24.04; ROCm 7.2; llama.cpp commit abc1234'
```

`--command` and `--system` are plain strings saved unchanged in the result. The
runner does not execute the command. Use them to record the exact server launch
(`llama-server ...`, `vllm serve ...`, `python -m sglang.launch_server ...`) and
the cards, OS, driver/runtime, and engine version used for the run.

Run against a hosted endpoint (the API key is redacted from the recorded
command; prefer the environment variable):

```bash
export LLM_BENCH_API_KEY=...
llm-context-bench \
  --base-url https://api.example.com/v1 \
  --model vendor-model-id \
  --profile vendor-2026-09 \
  --suite regular --lane performance --sizes 16k 64k \
  --output results/vendor-2026-09.json \
  --command 'hosted vendor endpoint' --system 'vendor API; region eu; server-side quantisation'
```

## Engine profiles and auto-detection

The harness speaks the OpenAI chat-completions API and sends each engine only
the parameters that engine actually accepts. Anything it drops is recorded per
trial as `engine_only_params_dropped`, so the effective payload of a run is
always visible in the result.

| `--engine` | Sends additionally | Notes |
|------------|-------------------|-------|
| `llama.cpp` | `cache_prompt`, `ignore_eos`, `top_k`, `min_p` | Also the only engine that reports in-band per-request `timings` and MTP draft counters |
| `vllm` | `ignore_eos`, `top_k`, `min_p` | |
| `sglang` | `top_k`, `min_p` | |
| `openai` | nothing | Strict hosted-API payload |
| `generic` | nothing | Assumed when nothing identifies the server |
| `auto` (default) | — | Probes `/props`, `/server_info`, `/get_model_info`, then `/v1/models` `owned_by` |

`auto` never fails on an unknown server: it resolves to `generic` and runs the
locked workload with the strict payload. A warm-up response containing a
`timings` object promotes `generic` to `llama.cpp` so the full payload is used
for measured trials. Override detection with an explicit `--engine` whenever
the workload's sampling parameters matter.

Other engine-related flags:

- `--no-stream` — run the performance lane without SSE. Prefill and decode then
cannot be separated, so only the whole-request rate is reported.
- `--input-size-tolerance-percent` — allowed drift between the requested tier
  and the server's `prompt_tokens` (default 2.0). Different tokenizer and chat
  template combinations legitimately shift a "16K" prompt by a few percent; the
  measured drift is always recorded as `median_input_size_percent_from_nominal`.
- `--max-retries` — extra attempts for a request that failed before producing
  any output, e.g. a hosted provider answering 429 (default 2). Retried attempts
  are counted in `attempts`/`retry_statuses` and never contribute timings.
- `--chat-template-kwargs` — JSON object passed through untouched, e.g.
  `'{"enable_thinking": false}'`.

## Size selection and token counts

`--sizes` accepts `all` or a space-separated list from `16k 32k 64k 128k`.
The fixture bytes are identical for every model and run. Because tokenizers and
chat templates differ, the server's `usage.prompt_tokens` is authoritative.
The fixture lengths were calibrated against the pinned Qwen3 tokenizer listed
in [FIXTURES.json](FIXTURES.json), while the runner validates the complete chat
prompt against the selected tier with a ±2% tolerance (configurable with
`--input-size-tolerance-percent` for engines with another tokenizer).

If the actual count falls outside that range, the JSON retains the measurements
but marks the trial invalid and excludes it from valid medians. This prevents a
nominal 16K case that actually tokenizes to, for example, 18K from being presented
as a valid 16K result.

## Warm-up

There is one model-load warm-up before all measured work:

- nominal input: 1,024 tokens (3,500 fixed fixture characters);
- required output: 512 tokens;
- `ignore_eos: true` is used only for this warm-up, and only when the selected
  engine accepts it (`fixed_length_check_applies` records whether it did);
- no full-size 16K–128K prompt is used for warming.

Use `--no-warmup` to skip it and record that choice in the result.

## Checkpointed results

The output JSON is atomically replaced after initialization, health check,
warm-up, every case start, every trial, every case completion, and final
completion. An interrupted or failed run therefore keeps all completed parts;
you do not wait for the entire benchmark to finish before results appear.

Result files include:

- model, profile, tool version, suite hashes, and fixture hashes;
- the supplied server `command` and `system` strings;
- the resolved engine profile, how it was detected, and which request parameters
  were dropped for it;
- `metric_sources`, which states for every metric whether the harness measured
  it or the engine cannot report it at all;
- Python and platform versions;
- server health and warm-up response;
- measured prefill and decode rates, TTFT, inter-token latency, whole-request
  rate, elapsed time, MTP counters where available, and the reason each trial
  passed or failed validation;
- a flat `performance_table` with one row per input tier;
- quality responses, scores, and per-case summaries.

Large prompts are referenced by locked file and hash rather than duplicated in
every result.

## How speed is measured

The performance lane requests each prompt with `stream: true` and
`stream_options: {"include_usage": true}`, stamps the arrival of every chunk
that actually carries text, and computes all rates from those stamps and the
engine's own token counts:

| Metric | Definition |
|--------|------------|
| `ttft_s` | request sent → first token-bearing stream chunk |
| `prefill_tps` | `prompt_tokens / ttft_s` (queueing, network, prefill, first sample) |
| `decode_window_s` | first token arrival → last token arrival |
| `generation_tps` | `(completion_tokens - 1) / decode_window_s` |
| `mean_inter_token_latency_ms` | `decode_window_s / (completion_tokens - 1)` |
| `output_tps_whole_request` | `completion_tokens / whole request wall-clock` |
| `request_tps_total` | `(prompt_tokens + completion_tokens) / whole request wall-clock` |

The first generated token is excluded from `generation_tps` because it is
sampled from the prefill's final logits — the same convention llama.cpp uses for
`predicted_per_second`, which keeps llama.cpp results comparable with their
historical records.

Chunk arrival times are only used for the window; token counts always come from
`usage`, because engines batch differently (vLLM commonly flushes ~1.6 tokens
per SSE event, recorded as `tokens_per_stream_chunk`). A stream that arrives in
one flush — a buffering proxy, typically — cannot show a decode window at all, so
`stream_delivery: buffered` makes the trial invalid rather than reporting the
whole request duration as prefill speed.

At the end of a run the harness prints the same numbers as a table, one row per
input tier:

```text
suite    tier  prompt tok  prefill t/s  gen t/s  TTFT s  ITL ms  req t/s  valid
-------------------------------------------------------------------------------
regular   16k      16,750        943.9     49.0   17.74   20.42     26.5    yes
regular   32k      33,455        942.0     58.3   35.51   17.16     19.3    yes
```

## Metrics

- `prefill_tps`, `generation_tps`, `ttft_s`, `decode_window_s`,
  `mean_inter_token_latency_ms` — measured by the harness on every engine.
- `output_tps_whole_request`, `request_tps_total` — user-perceived throughput;
  on a hosted endpoint these include network time.
- `token_count_source` — `usage` when the engine reported token counts,
  `engine-timings` when only counters were available, `stream-chunk-estimate`
  as a labelled last resort.
- `prompt_cache` — cached prompt tokens discovered through whichever field this
  engine uses (`timings.cache_n`, `usage.prompt_tokens_details.cached_tokens`,
  `usage.cached_tokens`, …). When an engine reports nothing, the trial records
  `prompt_cache_not_reported_by_engine`; the per-request tag prefix is what keeps
  the prefill cold there.
- `engine_cross_check` — for engines that do report per-request timings
  (llama.cpp), the harness compares its own rates with the engine's, checks that
  the advertised rate matches the engine's own counters, and verifies that
  `prompt_tokens == prompt_n + cache_n`. This is advisory: a missing `timings`
  object never invalidates a trial. llama.cpp attaches `timings` to the final
  stream chunk, so streaming keeps the in-band counters (including MTP draft
  stats); `--no-stream` reproduces the pre-0.3.0 non-streaming behaviour.
- `prompt_tps_engine` / `generation_tps_engine` — the engine's own advertised
  rates, kept for reference only.
- `draft_acceptance_rate`, `draft_tokens`, `accepted_draft_tokens` — accepted
  speculative draft tokens over drafted tokens. Only llama.cpp exposes these
  per request over HTTP; on vLLM, SGLang, and hosted APIs they are `null`, and
  `metric_sources.speculative_decoding_stats` says so explicitly.

See the official
[`llama-server` endpoint documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
for the llama.cpp response fields the cross-check reads.

### Comparing engines honestly

Engine-reported rates and harness-measured rates are different quantities, and
hosted endpoints add network time. Every result therefore carries `engine`,
`metric_sources`, and per-trial `notes`; compare a column only between runs that
report the same source. On a hosted endpoint, `output_tps_whole_request` is the
honest user-facing number while `prefill_tps` includes the transfer of a
16K–128K prompt.

### What makes a performance trial valid

`invalid_reasons` names every gate a trial missed; `performance_valid` is true
only when all of them pass:

- `request_failed` — the HTTP request or stream did not produce output;
- `output_length_mismatch` — not exactly 1,024 generated tokens;
- `prompt_tokens_not_reported` / `input_size_outside_tolerance` — the tier claim
cannot be verified, or the prompt landed outside the configured tolerance;
- `too_few_text_tokens`, `low_unique_token_ratio`, `dominant_repeated_token`,
`highly_compressible_repetition` — degenerate or forced output;
- `prompt_cache_hit` — the engine reported reused prompt tokens;
- `stream_not_incremental` — the response arrived in a single flush, so no decode
window exists to measure.

## Repetitions

The canonical suites use one measured repetition. Use `--repetitions N` for
diagnostic repeats; the result is then marked `canonical: false`. Medians include
only trials that satisfy every performance validity rule.

## Rebuild and verify fixtures

```bash
python3 -m pip install -e ".[fixtures]"       # runtime is dependency-free
python3 tools/generate_fixtures.py           # verify committed fixtures
python3 tools/generate_fixtures.py --write   # rebuild + refresh hashes
python3 -m unittest discover -s tests -v
```

Fixture assembly is deterministic from pinned public sources: Project Gutenberg
document URLs, a fixed `microsoft/vscode` commit, and the pinned Qwen3-8B
tokenizer listed in [FIXTURES.json](FIXTURES.json). The coding source list lives
in [tools/corpus/vscode-files.json](tools/corpus/vscode-files.json); downloads are
cached under `tools/.corpus_cache/` (git-ignored). A `--write` rebuild refreshes
the suite hashes and [FIXTURES.json](FIXTURES.json).

## Try it without a GPU

`tools/mock_openai_engine.py` is a dependency-free OpenAI-compatible server that
streams SSE, reports usage, and can answer whichever engine-signature endpoint
you choose. It is what the integration tests run against:

```bash
python3 tools/mock_openai_engine.py --port 8099 --signature vllm
llm-context-bench --base-url http://127.0.0.1:8099 --model mock-model \
  --profile mock --suite regular --lane performance --sizes 16k \
  --output /tmp/mock.json --command 'mock' --system 'mock'
```

Its token counts are synthesised from prompt length, so its rates exercise the
harness; they are not engine measurements.

## License

Code and fixture assembly are licensed under the [MIT License](LICENSE).
Fixtures also bundle third-party content: public-domain texts from Project
Gutenberg (no license restrictions) and `microsoft/vscode` source (dual-licensed
MIT/Apache-2.0) with its per-file copyright headers preserved.
