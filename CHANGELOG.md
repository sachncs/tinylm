# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- MIT licence, contributing guide, security policy, code of conduct, citation
  file, issue and pull request templates, Dependabot configuration.
- CI workflow running the quality gate on Linux and macOS and a wheel install
  smoke test.
- `py.typed` marker and full package metadata.

### Changed
- Lower bounds on `torch`, `tokenizers` and `transformers`.
- The source distribution contains only source, tests and top-level docs.
- `plan.md` moved to `docs/status.md`.

## [0.2.0]

Final Strict-Small (3.7M) and Strict (12.9M) models, official-pipeline
evaluation, GLUE fine-tuning, probes, and the documentation site. See
[docs/paper.md](docs/paper.md).
