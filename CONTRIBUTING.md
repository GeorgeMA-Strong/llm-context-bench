# Contributing

Thank you for improving `llm-context-bench`.

## Before opening a change

1. Install the package in editable mode: `python3 -m pip install -e .`
2. Run `python3 -m unittest discover -s tests -v`.
3. If fixture generation changed, run `python3 tools/generate_fixtures.py` and
   commit the generated prompts, suite hash updates, and `FIXTURES.json` together.
4. Confirm no benchmark records, hostnames, local paths, credentials, model
   files, or private source material are included.

Changes to prompts, sampling, warm-up behavior, validity rules, or metric
formulas change comparability. Explain those changes clearly and increment the
suite or benchmark schema when old and new results must not be compared.

Keep the runtime dependency-free unless a dependency provides a substantial,
measured benefit. Prefer deterministic tests that do not require a GPU or a
running server.
