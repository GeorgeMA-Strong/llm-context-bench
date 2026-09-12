# Changelog

All notable changes to this project will be documented here.

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
