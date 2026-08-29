# Changelog

All notable changes to this project will be documented here.

## 0.1.0 - 2026-08-29

- Initial public-ready release.
- Added deterministic synthetic regular and coding fixtures at 16K, 32K, 64K,
  and 128K tiers.
- Added quality and performance lanes with fixed 1,024-token measured output.
- Added one 1K-input/512-output model-load warm-up.
- Added atomic checkpoints after every benchmark part.
- Added server command, system description, tool version, and runtime metadata.
- Added cache, timing, fixed-length, input-size, and output-degeneracy validation.
