# Changelog

All notable changes to this project will be documented here.

## 0.4.0 - 2026-09-13

- `--concurrency N` (1-8, default 1 without the flag) holds N identical
  performance requests in flight at once and reports both views: whole-load
  totals (`wall_s`, `prompt_tokens_total`, `output_tokens_total`,
  `prefill_tps_total`, `token_generation_tps_total`, `generating_window_s`,
  `tokens_tps_total`, `requests_per_minute`, `group_valid`) and the unchanged
  per-request rates. Requests wait on a start barrier so a concurrent run cannot
  drift sequential, each copy carries its own request tag so identical prompts
  cannot share a prefix-cache entry, and `content_only_failure` separates "a
  sampled stream went degenerate" from a measurement failure. The quality lane
  stays sequential and a concurrent run is recorded `canonical: false`.
- The run prints two tables plus a legend: `WHOLE LOAD` with `TOTAL PP t/s` and
  `TOTAL TG t/s` - every prompt divided by the time until the slowest stream's
  first token, and every generated token minus each stream's first divided by the
  group's generating window - and `ONE STREAM` with the per-request rates. The
  totals are the engine's real output under load, not a single stream's rate
  multiplied by the request count, and PP/TG keep llama.cpp's definitions.
- Added the 8K input tier to both suites (`*-09-context-8k`) and `--sizes 8k`,
  built from the same pinned public-domain and `microsoft/vscode` sources. The
  fixed chat-template and instruction overhead is a larger share of a shorter
  prompt, so an 8K run needs a wider `--input-size-tolerance-percent` than 16K.
- Request tags now carry a per-run scope (`--request-tag-scope`, recorded as
  `measurement.request_tag_scope`). Engines with automatic prefix caching key KV
  blocks on the leading prompt tokens, so the previous deterministic tag let a
  re-run start from a warm prefill: the same 8K prompt measured 4,451 t/s on a
  second run against 1,310 t/s on the first, on an engine that reports no
  cached-token field. Pass the recorded scope back to reproduce a run's prompts.
- `generate_fixtures.py --write` no longer fails on its own path handling
  (`prompt_file` is package-relative), which is what made the 8K rebuild possible.
- `benchmark_schema_version` is 8.

## 0.3.0 - 2026-09-12

- Engine-agnostic runner: the performance lane no longer depends on llama.cpp's
  `timings` object. Every rate is now calculated by the harness from the
  server's token counts and its own monotonic clock, so llama.cpp, vLLM,
  SGLang, and hosted OpenAI-compatible endpoints are measured identically.
- The performance lane streams with SSE (`stream_options.include_usage`) and
  records `ttft_s`, `prefill_tps` (`prompt_tokens / TTFT`), `decode_window_s`,
  `generation_tps` (`(completion_tokens - 1) / decode window`),
  `mean_inter_token_latency_ms`, and `request_tps_total`, so prefill and token
  generation speed are reported separately per input tier.
- Engine profiles (`--engine auto|llama.cpp|vllm|sglang|openai|generic`) send
  only the parameters each server accepts and record what was dropped.
  `auto` identifies the server from `/props`, `/server_info`, `/get_model_info`,
  and `/v1/models` `owned_by`, then falls back to the strict OpenAI payload.
- Prompt-cache detection probes whichever field an engine uses
  (`timings.cache_n`, `usage.prompt_tokens_details.cached_tokens`,
  `usage.cached_tokens`, …) and says so when nobody reports it.
- llama.cpp's in-band counters are now an advisory cross-check
  (`engine_cross_check`), including whether an advertised rate matches the
  engine's own counters and how far it diverges from the measured one.
- Per-trial `invalid_reasons`, `notes`, `attempts`/`retry_statuses`, and
  `stream_delivery` make an unusable measurement explicit instead of silent.
  A stream delivered in one flush is rejected rather than reported as prefill.
- Added `--no-stream`, `--input-size-tolerance-percent`, and `--max-retries`
  (rate limits and connection resets on hosted endpoints).
- Added a printed `prefill t/s` / `gen t/s` / `TTFT` table per input tier and a
  flat `performance_table` in the result JSON; `benchmark_schema_version` is 7,
  and `harness_sha256` now fingerprints `engine.py` as well.
- Added `tools/mock_openai_engine.py`, a dependency-free OpenAI-compatible SSE
  server used by the integration tests.

## 0.2.0 - 2026-08-29

- Replaced synthetic fixtures with deterministic assemblies of real content:
  public-domain documents (Project Gutenberg) for **regular** and pinned
  `microsoft/vscode` source (MIT/Apache-2.0) for **coding**.
- Needles (authoritative records/functions), `TASK:` footers, quality
  checkers, and token calibration are unchanged.
- Performance instructions now pin per-profile output style: plain narrative
  prose (regular) and source code only (coding), so speculative-decoding
  acceptance differs measurably by workload domain.
- `tools/generate_fixtures.py` now deterministically rebuilds or verifies the
  fixtures from pinned public sources (`--write` refreshes hashes).

## 0.1.0 - 2026-08-29

- Initial public-ready release.
- Added deterministic synthetic regular and coding fixtures at 16K, 32K, 64K,
  and 128K tiers.
- Added quality and performance lanes with fixed 1,024-token measured output.
- Added one 1K-input/512-output model-load warm-up.
- Added atomic checkpoints after every benchmark part.
- Added server command, system description, tool version, and runtime metadata.
- Added cache, timing, fixed-length, input-size, and output-degeneracy validation.
