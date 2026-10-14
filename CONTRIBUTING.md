# Contributing

Thanks for helping. This project values measured claims and small, tested
changes.

## Setup

```sh
uv sync --group official   # Python 3.14, locked dependencies
./scripts/install.sh       # enables the pre-commit quality gate
```

## Rules

1. **Test first.** Write the failing test, then the code.
2. **The gate must pass:** `./scripts/gate.sh` runs `ruff format --check`,
   `ruff check`, `pyright` (strict) and `pytest` with at least 97% coverage.
   CI runs the same script.
3. **Warnings are errors** in tests; fix the cause, do not filter the warning.
4. **Numbers need evidence.** A result in the docs must come from a saved run
   under `runs/` with its config, seed and manifest. State the seed count; do
   not report a single seed as a finding.
5. Do not commit `data/`, `runs/`, `external/` or checkpoints (they are
   gitignored).
6. Add a line to `CHANGELOG.md` for any user-visible change.

## Pull requests

Branch from `main`, keep one topic per PR, fill in the PR template, and link
the issue. Reproduction steps are in [docs/reproduce.md](docs/reproduce.md).

By contributing you agree your work is released under the [MIT licence](LICENSE).
