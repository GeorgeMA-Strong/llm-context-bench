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

Hold eight identical prose and code requests in flight at the 8K tier and read
both per-stream and aggregate throughput:

```bash
llm-context-bench \
  --base-url http://127.0.0.1:8080 \
  --model qwen-model \
  --profile qwen-load-8 \
  --suite all --lane performance --sizes 8k \
  --concurrency 8 \
  --output results/qwen-load-8.json \
  --command 'vllm serve model --max-model-len 262144' \
  --system '4x GPU; Ubuntu 24.04; vLLM commit def5678'
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
- `--request-tag-scope` — see [Keep the prefill cold](#keep-the-prefill-cold).
- `--chat-template-kwargs` — JSON object passed through untouched, e.g.
  `'{"enable_thinking": false}'`.

## Size selection and token counts

`--sizes` accepts `all` or a space-separated list from `8k 16k 32k 64k 128k`.
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

The measured prompt also carries the chat template, the performance instruction,
and the request tag on top of the fixture, which is a fixed ~350-400 tokens. That
overhead is about 2% of a 16K tier and about 5% of an 8K tier, so an 8K run needs a
wider tolerance than a 16K run on the same engine. Measured drift on a vLLM server
using a quantised model's own tokenizer: 8K prose +2.8%, 8K code +9.2%, 16K code
+10.2% — the code fixture tokenises noticeably larger than the calibration
tokenizer, so `--input-size-tolerance-percent 12` is what makes the whole ladder
valid there.

## Warm-up

There is one model-load warm-up before all measured work:

- nominal input: 1,024 tokens (3,500 fixed fixture characters);
- required output: 512 tokens;
- `ignore_eos: true` is used only for this warm-up, and only when the selected
  engine accepts it (`fixed_length_check_applies` records whether it did);
- no full-size 8K–128K prompt is used for warming.

Use `--no-warmup` to skip it and record that choice in the result.

## Checkpointed results

The output JSON is atomically replaced after initialization, health check,
warm-up, every case start, every completed load group (one request at
concurrency 1), every case completion, and final completion. An interrupted or
failed run therefore keeps all completed parts; you do not wait for the entire
benchmark to finish before results appear.

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

At the end of a run the harness prints two tables and a legend. The first totals
the whole load, the second is one request's own experience:

```text
WHOLE LOAD - every request in the group added together
suite    tier  req  wall s  prompt tok  out tok  TOTAL PP t/s  TOTAL TG t/s  all tok t/s  req/min  valid
coding     8k    4    82.2      35,868    4,096       1,167.6          57.5        486.2     2.92    yes
regular    8k    4    86.4      33,780    4,096       1,149.3          54.1        438.3     2.78    yes

ONE STREAM - what a single request experiences (median of the group)
suite    tier  req  prompt tok/req  PP t/s  TG t/s  TTFT med s  TTFT max s  ITL ms  out t/s/req  valid
coding     8k    4           8,967   381.9    17.9       24.29       30.72   56.13         12.5    yes
regular    8k    4           8,445   371.1    16.7       23.53       29.39   60.26         12.1    yes
```

`TOTAL PP t/s` and `TOTAL TG t/s` are the summary of the whole load: every
prompt divided by the time until the slowest stream produced its first token,
and every generated token (minus each stream's first) divided by the time the
group spent generating. They are what the engine did in total - not a single
stream's rate multiplied by the request count, which would claim 4 x 17.9 = 71.6
t/s where the engine actually produced 54.1. The printed legend repeats every
definition, and `PP`/`TG` use llama.cpp's definitions so the numbers line up with
`prompt_per_second` and `predicted_per_second` records.

## Concurrency

`--concurrency N` (1-8, default 1; a run without the flag is a single in-flight
request) holds N identical performance requests in flight at once and reports
both views:

- **whole load** (`*_total` on each group) — `wall_s` (first request sent until
  the last finished), `prompt_tokens_total`, `output_tokens_total`,
  `prefill_tps_total`, `token_generation_tps_total`, `generating_window_s`,
  `tokens_tps_total`, `requests_per_minute`, plus `requests`, `valid_requests`
  and `group_valid`;
- **one stream** — the same `prefill_tps`, `generation_tps`, `ttft_s` and gates
  as a sequential run, so you can see how a single user degrades under load.

All N requests wait on a start barrier, so a "concurrent" run cannot quietly
drift into a sequential one. Group wall time runs from the first request leaving
the harness to the last response, so a straggler thread can only lower the
reported totals, never inflate them.

Each suite and each case is measured one group at a time: `--suite all
--concurrency 4` runs 4 prose requests together, waits, then 4 code requests
together - it is not a mixed 2 + 2 load.

Every copy carries its own request tag, so N simultaneous identical prompts
cannot collapse into one shared prefix-cache entry. A group is valid only when
every request in it passed every gate; `content_only_failure: true` on a group
separates "a sampled stream went degenerate" (routine at temperature 1.0, more
likely with more streams) from a measurement failure. Totals of invalid groups
stay in the JSON but are excluded from the table row.

The quality lane always stays sequential — its scores have to stay comparable.
A run with `--concurrency` other than 1 is recorded `canonical: false`, like a
run with modified repetitions.

## Keep the prefill cold

Each request sends a tag in its system prompt, and the tag now includes a
**per-run scope** (the run start time, or `--request-tag-scope`). That is not
cosmetic. Engines with automatic prefix caching key KV blocks on the leading
prompt tokens, so a tag that repeats between runs lets the second run start from
a warm prefill. Measured on a vLLM server that reports no cached-token field:
the same 8K prompt came back at 4,451 t/s prefill on a re-run against 1,310 t/s
on the first — a 3.4× lie that nothing in the response betrayed.

`measurement.request_tag_scope` is recorded, so passing it back with
`--request-tag-scope` reproduces an earlier run's exact prompts when you want
that comparison instead.

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
  `prompt_cache_not_reported_by_engine`; the run-scoped per-request tag is what
  keeps the prefill cold there.
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
honest user-facing number while `prefill_tps` includes the transfer of an
8K–128K prompt.

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

A load group adds two of its own reasons in `invalid_reason_totals`:
`concurrency_barrier_broken` (the requests cannot be claimed to have started
together) and `group_wall_unmeasured` (no group wall time was observed, so no
total can be computed).

## Repetitions

The canonical suites use one measured repetition. Use `--repetitions N` for
diagnostic repeats; the result is then marked `canonical: false`. Medians include
only trials that satisfy every performance validity rule. With
`--concurrency N`, each repetition is one load group of N simultaneous requests,
so a run issues repetitions × N requests per case.

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
