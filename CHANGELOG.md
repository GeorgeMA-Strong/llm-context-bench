# Changelog

All notable changes to this project will be documented here.

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
