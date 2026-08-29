# llm-context-bench

Reproducible long-context quality and speed benchmarks for `llama.cpp` servers.
The repository locks the prompts, sampling parameters, output limits, scoring,
warm-up, and validation rules so repeated runs compare the same workload.

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
- rejects prompt-cache hits;
- recomputes llama.cpp PP and TG rates from token and timing counters;
- excludes invalid trials from reported performance medians.

The bundled input fixtures are deterministic assemblies of public-domain
documents and permissively-licensed open-source code; each fixture names its
sources. Their gzip ratios are checked to prevent highly repetitive filler, and
their SHA-256 hashes are locked in the suite manifests.

## Install

Python 3.10 or newer and a running `llama-server` are required.

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
and the cards, OS, driver/runtime, and llama.cpp version used for the run.

Set `LLM_BENCH_API_KEY` when the endpoint requires authentication. Avoid placing
real keys on the command line. Any `--api-key` argument is redacted from the
recorded benchmark command.

## Size selection and token counts

`--sizes` accepts `all` or a space-separated list from `16k 32k 64k 128k`.
The fixture bytes are identical for every model and run. Because tokenizers and
chat templates differ, the server's `usage.prompt_tokens` is authoritative.
The fixture lengths were calibrated against the pinned Qwen3 tokenizer listed
in [FIXTURES.json](FIXTURES.json), while the runner validates the complete chat
prompt against the selected tier with a ±2% tolerance.

If the actual count falls outside that range, the JSON retains the measurements
but marks the trial invalid and excludes it from valid medians. This prevents a
nominal 16K case that actually tokenizes to, for example, 18K from being presented
as a valid 16K result.

## Warm-up

There is one model-load warm-up before all measured work:

- nominal input: 1,024 tokens (3,500 fixed fixture characters);
- required output: 512 tokens;
- `ignore_eos: true` is used only for this warm-up;
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
- Python and platform versions;
- server health and warm-up response;
- PP, TG, whole-request rate, elapsed time, MTP counters, and validation details;
- quality responses, scores, and per-case summaries.

Large prompts are referenced by locked file and hash rather than duplicated in
every result.

## Metrics

- `prompt_tps_engine` — llama.cpp prompt-processing rate (PP).
- `generation_tps_engine` — llama.cpp decode rate (TG). llama.cpp excludes the
  first sampled token from this calculation; the runner uses the same formula.
- `output_tps_whole_request` — completion tokens divided by complete HTTP wall
  time, including prompt processing and request overhead.
- `draft_acceptance_rate` — accepted speculative draft tokens divided by drafted
  tokens when the server reports both counters.

Engine timing validation requires consistent `prompt_n`, `cache_n`,
`predicted_n`, usage counts, milliseconds, and reported rates. See the official
[`llama-server` endpoint documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
for the server response fields.

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

## License

Code and fixture assembly are licensed under the [MIT License](LICENSE).
Fixtures also bundle third-party content: public-domain texts from Project
Gutenberg (no license restrictions) and `microsoft/vscode` source (dual-licensed
MIT/Apache-2.0) with its per-file copyright headers preserved.
