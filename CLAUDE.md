# CLAUDE.md

tinylm: small language models for the BabyLM 2026 benchmark. Python 3.14
(GIL build), managed with `uv`.

## Commands
- `uv sync --group official` install locked deps; `./scripts/install.sh` enable hooks.
- `./scripts/gate.sh` the quality gate: ruff format, ruff check, pyright strict,
  pytest with coverage >= 97%. CI (`.github/workflows/ci.yml`) runs this script.
- `uv build` builds sdist and wheel; `cd site && npm ci && npm run build` builds docs.

## Hooks and scripts (deterministic rules live here, not in prompts)
- `scripts/hooks/pre-commit` blocks commits that fail the gate;
  `scripts/hooks/post-commit` backs up `external/`; `scripts/install.sh` wires them.
- `scripts/gate.sh` is the single definition of "passes"; never copy its commands.

## Rules
- Test first; warnings are errors; do not weaken a check to pass.
- Never commit `data/`, `runs/`, `external/`, checkpoints. Never edit results in `runs/`.
- Report a number only from a saved run, with its seed count.
- Ask before push, tag, release, PyPI upload, or changing GitHub settings.
- Harness files live in `harness/` (root `plan.md` was moved to `docs/status.md`).
- `.python-version` is `3.14+gil`: plain `3.14` can resolve to the free-threaded
  build on machines that have it; the old value was a platform-specific string
  that broke Linux.
- The sdist must stay small: `[tool.hatch.build.targets.sdist]` is an allowlist,
  because the default included `site/node_modules` (11k files).
- Run `uv build`, `uv pip install` of the wheel, and the link check after packaging edits.
- Pin GitHub Actions by commit SHA only when the SHA was looked up, never invented;
  Dependabot keeps tags current.
