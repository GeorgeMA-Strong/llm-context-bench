# Security policy

## Reporting a vulnerability

Please report security issues privately through the repository's GitHub security
advisory page. Do not include credentials, private model paths, or unredacted
benchmark results in a public issue.

## Operational notes

- Prefer `LLM_BENCH_API_KEY` over a command-line key.
- Result files intentionally save the supplied `--command` and `--system`
  strings. Remove secrets from those strings before running.
- The runner sends fixture contents and generated responses to the configured
  endpoint only. It does not upload results elsewhere.
- Coding responses are checked in a restricted child process, but this is a
  benchmark guardrail, not a hardened security sandbox. Run the tool under an
  account and machine appropriate for untrusted model output.
