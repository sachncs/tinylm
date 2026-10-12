# tinylm

Small language models (4M to 22M parameters) for the BabyLM 2026 benchmark,
built test-first, offline, and reproducible. The goal is the smallest model
that is accurate and precise, found by measurement rather than assumption.

## Setup

```sh
uv sync                    # Python 3.14.8 (standard, GIL build), locked deps
./scripts/install.sh       # enables the pre-commit gate and the external backup
```

## Pipeline

| Step | Command |
|------|---------|
| Download pinned data once (checksummed) | `uv run python scripts/data.py` |
| Build offline token stores | `uv run python scripts/build.py` |
| Check benchmark leakage | `uv run python scripts/leak.py data/strict data/evals/evaluation_data/full_eval/blimp_filtered` |
| One experiment | `uv run python scripts/run.py configs/proxy.toml --name v --set train.lr=0.003` |
| A queue of variants | `uv run python scripts/batch.py configs/<queue>.toml` |
| 24-run factor screen | `uv run python scripts/screen.py run` then `... analyse` |
| Summary table | `uv run python scripts/table.py runs/screen` |
| Official benchmark scores of a finished run | `uv run python scripts/official.py runs/screen/<run>` |
| Paired comparison of two runs | `uv run python scripts/compare.py runs/a runs/b blimp_fast` |

After `data.py` nothing touches the network: `Offline` blocks sockets and the
tests prove prepare, train and evaluate run with the network disabled.

## Quality gate

`scripts/gate.sh` runs `ruff format --check`, `ruff check`, strict `pyright`,
and `pytest` with warnings as errors and a 97% coverage floor. The versioned
`pre-commit` hook runs it, so a red commit is impossible; `post-commit` writes
a full `git bundle` to `~/Documents/Codex/backups/` after every commit.

## Layout

`src/tinylm/` library (one noun per module: `Token`, `Prepare`, `Train`,
`Bench`, `Bootstrap`, `Experiment`, `Screen`, `Export`, `Official`, ...),
`scripts/` thin command-line entry points, `configs/` experiment files,
`tests/` unit, property-based and end-to-end tests, `docs/` findings,
`plan.md` the plan, `docs/review.md` the engineering review,
`docs/standards.md` how results must be reported.

## Conventions

Python 3.14 only; Google style; names are judged in local context (no
`prepare.prepare`, no `TokenStore`); no leading underscores; frozen
dataclasses for every record that crosses a boundary; failures are loud and
typed; atomic, versioned, checksummed files; atomic commits, tests first.
