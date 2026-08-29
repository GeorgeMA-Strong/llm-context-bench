# Proposal: Fix the regular/coding fixture divergence (MTP results are identical)

Status: proposal only — no code changed yet.
Scope: input fixtures + performance-lane task wording + validation gates.
Trigger: `regular 16K` and `coding 16K` performance runs return statistically
indistinguishable results (PP 888.8 vs 876.0 tok/s, TG 54.4 vs 53.4 tok/s,
same 1,024-token outputs) even though the profiles are supposed to be
"non-code vs code". For an MTP model, draft acceptance should differ
between prose-like and code-like generation — identical numbers are a red
flag, and they are.

## 1. Verdict

The suspicion is confirmed: **the "regular" fixture is not prose**. It is a
template-generated telemetry log whose token statistics are nearly identical
to the "coding" fixture. Both bundles are ~95% the same shape — templated
lines of random numbers plus hex digests around a small amount of real
content — so the model sees one workload, not two. Identical MTP/PP/TG
numbers are the *expected* outcome of the current fixtures, not a model or
server problem.

## 2. How the performance lane works (why fixtures decide everything)

`runner.py` (`run_suite`, performance branch):

1. `render_performance_prompt()` strips the trailing `\nTASK:` from the
   fixture and appends the suite's `performance_instruction`
   ("write ~4,000 words of prose / code … grounded in the bundle").
2. One request: cold cache (unique `request_tag` + `cache_prompt: false`),
   `max_tokens=1024`, no `ignore_eos`, locked sampling
   (temp 1.0, top_p 0.95, top_k 20, min_p 0.0, seed 3407).
3. Metrics come from the server's `timings`: PP from `prompt_n/prompt_ms`,
   TG from `(predicted_n-1)/predicted_ms`, and MTP acceptance from
   `draft_n_accepted / draft_n`.

Two consequences:

- **PP is content-blind.** It is dominated by token count (16,384 on both
  sides); 888 vs 876 is noise-level. PP cannot distinguish the profiles.
- **TG and draft acceptance are output-distribution-dependent.** What the
  MTP head must predict is the stream of 1,024 generated tokens. If both
  suites generate the same *kind* of tokens, acceptance and TG will match.

## 3. Evidence: the fixtures are the same distribution

Measured on the smallest fixtures (16k) plus a real-prose reference
(Project Gutenberg, *A Tale of Two Cities*, 33.1k-char window):

| metric                    | regular-16k | coding-16k | real prose (Dickens) |
|---------------------------|-------------|------------|----------------------|
| char entropy              | 5.14 bits   | 5.26 bits  | 4.48 bits            |
| word-bigram entropy       | 10.89       | 10.52      | 11.94                |
| zlib-9 ratio (lower = more repetitive) | **0.345** | **0.360** | 0.423       |
| digit character share     | 25.2%       | 33.2%      | 0.0%                 |
| hex digest chunks (8+ char) | 175        | 157        | 0                    |
| templated "machine" lines | **56.4%** of lines | **46.9%** of lines | ~0%     |
| unique-word ratio         | 0.465       | 0.654      | 0.355                |
| chars per nominal token   | 2.02        | 1.74       | ~4.3–5.1             |

What the "regular" fixture actually is (`tools/generate_fixtures.py`):

- 10 sections at 16k, each = one of 6 short templates + an "Evidence
  samples" block of 18 lines drawn from one fixed template
  `- <ts> actor=<24 names> service=<24 names> region=<8> state=<12>
  count=<7-digit> latency_us=<6-digit> sample=<28-hex>`.
- 3 injected "authoritative records" (the quality-lane needles) at
  ~10/50/90%.
- Closed vocabulary everywhere: 24 people, 24 services, 8 regions,
  12 states.

The "coding" fixture is the mirror image: ~10 snippets from 5 short
code templates (Python/SQL/sh) with rotated identifiers, plus `# vector=`
lines of random ints + 28-hex checksums, plus one verbatim repeated note
paragraph ("The … excerpt is synthetic. Reviewers should check …") 10×.

So:

- The regular fixture compresses ~20% better than actual Dickens prose
  (0.345 vs 0.423) — it is *more* templated than natural language.
- The two fixtures differ by <4% on every compression/entropy axis.
- Both tokenize at ~2 chars/token (numbers/hex merge aggressively), i.e.
  their token streams are dominated by digit/hex runs in fixed templates —
  exactly the kind of content an MTP head predicts with near-uniform
  success, regardless of the "profile".

## 4. Why the output side doesn't save the distinction

- Both `performance_instruction`s say the writing must be *grounded in the
  bundle*. The bundles are 95% templated numbers/hex, so the model's
  continuation is pulled into that low-entropy structured regime in both
  suites (prose-style scaffolding + telemetry-style content ≈ code-style
  scaffolding + vector-style content).
- The only output guard, `analyze_output_content()`, rejects extreme
  degeneracy (dominant token >20%, compression <0.15, <128 tokens).
  Templated-but-varied continuation passes it — it is designed to catch
  "1000 dots", not "realistic-looking telemetry".
- Inconsistent setup on top: `regular` has
  `performance_system_prompt: ""` (the system message is only the request
  tag), while `coding` has a full system prompt steering toward "realistic
  code examples". The two lanes are not even the same experiment shape.

## 5. Secondary issues found while investigating

1. `test_long_prompts_are_not_repetitive_filler` only asserts gzip ratio
   > 0.20; the fixtures sit at 0.30–0.36, far below real prose (~0.42).
   The README claim "their gzip ratios are checked to prevent highly
   repetitive filler" overstates what the check does. The gate would need
   to be much stricter to have caught this.
2. `MODEL_LOAD_WARMUP_SOURCE` is coupled to `prompts/regular-16k.txt`
   (first 3,500 chars). Fine today, but any fixture redesign must keep
   that file ≥3,500 chars of normal prose or change the constant.
3. Char-target calibration (`TARGET_CHARS`) is a guess about
   chars/token (2.02 for current content). Real prose is ~4.3–5.1
   chars/token with the pinned Qwen3-class tokenizer, so new fixtures must
   be calibrated by counting tokens, not chars, to stay inside the ±2%
   `input_size_valid` band.

## 6. Recommended solution

Goal: the two profiles must produce measurably different token
distributions **in both prefill and generation**, while keeping everything
the harness already locks (determinism, hashes, ±2% size tolerance,
answerable quality needles, cold-cache rules).

### A. Rebuild fixtures from real, public-domain corpora (the core fix)

- **Regular**: assemble from real public-domain text (Project Gutenberg
  works, e.g. Dickens + 2–3 other works or old technical manuals/RFCs — a
  mix of narrative and technical is more representative of "regular use").
  Keep the 3 authoritative AC records as needles at ~10/50/90% token
  positions and the `TASK:` footer. This turns the quality case into a
  standard needle-in-a-haystack over real text (the RULER/NIAH pattern)
  and gives the performance lane a genuinely prose-like prefill.
- **Coding**: assemble from real permissively-licensed (MIT/Apache/BSD)
  source files (a pinned small OSS tree: .py/.sql/.sh/.md), keeping the 3
  authoritative functions and `TASK:` footer. Drop the generated
  `# vector=` filler, or cap it at <20% of the bundle.
- **Reproducibility**: vendor the source files in the repo
  (`tools/corpus/regular/*.txt`, `tools/corpus/coding/*`) instead of
  downloading at build time. Record per-source SHA-256 + license/attribution
  in `FIXTURES.json`. Deterministic assembly = fixed source order, fixed
  offsets, no RNG (keep SEED 3407 for anything that still needs it).
- **Calibration**: replace `TARGET_CHARS` with token counting against the
  already-pinned Qwen3 tokenizer revision; regenerate until each tier is
  within ±2% of nominal tokens. Expect regular fixtures to roughly double
  in bytes (~70–80k chars for 16k) once content is real prose.

### B. Decouple the generation regime from the input style (fixes the MTP axis)

Draft acceptance is measured on the **generated** tokens, so the two suites
must actually generate different regimes:

- **Regular perf task**: free-running long-form prose on a general topic
  (e.g. "Write a detailed technical explanation of X, ~4,000 words"),
  *without* forcing it to imitate the bundle's style. The input still
  provides the long prefill; the output is real prose.
- **Coding perf task**: keep "write substantial code + explanation" (it
  already asks for code; keep the system prompt).
- Give **both** suites a non-empty `performance_system_prompt` so the two
  requests have the same shape.
- Keep the locked sampling/seed, fixed 1,024 output, cold-cache tagging,
  and the degenerate-output rejection unchanged.
- (Optional, later) a third "agentic/tool-call JSON" profile — it is the
  dominant real-world MTP workload — but out of scope for this pass.

### C. Strengthen fixture validation gates

Add generator/test assertions that would have caught the current fixtures:

- zlib-9 ratio band per profile (prose ~0.35–0.50; reject <0.32);
- digit-char share cap for regular (<10%);
- no single line template may occupy >5% of lines;
- repeated verbatim paragraph cap (e.g. any 12+ word span repeated >2×
  fails);
- unique-word ratio floor (regular >0.35 on a 16k window).

Tighten `test_long_prompts_are_not_repetitive_filler` accordingly and fix
the README wording to match.

### D. Locking & bookkeeping

- Bump suite `schema_version` (and `suite_id` suffix, e.g. `-v3`) since
  inputs change; `tools/generate_fixtures.py` already rewrites
  `prompt_sha256` in the suites and `FIXTURES.json`.
- Update README: content provenance (public-domain/permissive, no private
  data), the new calibration method, and the warmup-source note.
- Keep `MODEL_LOAD_WARMUP_SOURCE` valid (first 3,500 chars of the new
  regular-16k fixture must be normal prose — it will be).

### E. Verification plan (after implementation)

1. Offline: `python3 tools/generate_fixtures.py`, then
   `python3 -m unittest discover -s tests -v` — all green, including new
   variety gates and ±2% token-count checks.
2. Empirical (16k first, matching this investigation): rerun the
   performance lane on the MTP model for both suites and compare
   `median_draft_acceptance_rate` (and TG). Expected: **coding acceptance
   > regular acceptance** (code is more predictable; MTP work reports
   gains "especially pronounced on generative benchmarks like coding"),
   with PP staying roughly equal — equal PP is correct, not a bug.
3. Sanity: quality lane still 100% (needles are still embedded, checkers
   unchanged).
4. Then repeat at 32/64/128k to confirm the divergence persists with size.

## 7. Sources

- llama.cpp server docs — `timings` object (`cache_n`, `prompt_n`,
  `predicted_n`, rates) and speculative-decoding options
  (`--spec-type draft-mtp`, `--spec-draft-n-max`):
  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- RULER: What's the Real Context Size of Your Long-Context Language
  Models? (arXiv:2404.06654) — synthetic long-context design; needle
  retrieval over distractor text; task diversity matters.
- Gloeckle et al., Better & Faster Large Language Models via Multi-token
  Prediction (arXiv:2404.19737) — MTP speedups "especially pronounced on
  generative benchmarks like coding" (domain-dependent acceptance).
- Real-prose reference stats: Project Gutenberg eBook #98 (Dickens,
  public domain), 33.1k-char window: char entropy 4.48 bits, zlib-9 0.423,
  0% digits — measured for this proposal.

## 8. Open questions (need your call)

1. **Corpus choice**: which 3–5 public-domain works for "regular"?
   (Proposal: 1 narrative + 2 technical/standards + 1 scientific, so
   "regular use" isn't biased to one register.)
2. **Coding corpus**: pick a specific small MIT/Apache repo to pin, or
   keep a generated-but-more-realistic code corpus? Vendored real files
   are preferred for the MTP question.
3. **Performance task wording**: free-running prose (recommended) vs
   keep "grounded in the bundle" (weaker separation, more RULER-like)?
4. **Schema bump**: fine to invalidate all existing result JSONs
   (`suite_sha256` will change, so old/new results are not comparable)?
