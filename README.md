# tinylm

Small language models (3.7M and 12.9M parameters) for the
[BabyLM 2026](https://babylm.github.io) benchmark, built test-first, offline
and reproducibly on one Apple-silicon laptop. The question: how small can a
model be and still be accurate and precise, found by measurement rather than
assumption.

**Documentation site:** https://sachncs.github.io/tinylm

## Results

| Model | Parameters | Official BLiMP | Supplement | COMPS | EWoK (fast) | GLUE mean |
|---|---|---|---|---|---|---|
| `tinylm-small-3.7M` (Strict-Small, 10 epochs) | 3.70M | 67.4 | 57.0 | 52.5 | 53.9 | 64.2 |
| `tinylm-strict-12.9M` (Strict, 10 epochs) | 12.92M | 74.8 | 60.2 | 55.8 | 54.7 | 66.1 |

These are well below the best published BabyLM systems (about 80 BLiMP on
Strict-Small); the 90 this project aimed for was **not reached**. Single final
run per size, no seed replicates. Weights are release assets (see
[Models](docs/models.md)). Everything measured, including the negative
results, is in [docs/paper.md](docs/paper.md).

What the experiments found (4 epochs, 3 seeds per arm unless noted): RMSNorm
beats Dynamic Tanh, Muon beats AdamW, a 16k vocabulary beats 4k, dropout 0.1
hurts, a recency penalty and exclusive self attention each lower bits per
character by about 0.005 with no clear BLiMP effect, and cross attention to
memory slots does not pay for its parameters.

## Setup

```sh
uv sync --group official   # Python 3.14.8 (standard, GIL build), locked deps
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

## Documentation

[Models](docs/models.md) (usage, scores, data notes) ·
[Architecture](docs/architecture.md) · [Results](docs/paper.md) ·
[Reproducing](docs/reproduce.md) · [Screen](docs/findings/screen.md) ·
[Attention](docs/findings/attention.md) · [Review](docs/review.md) ·
[Reporting standards](docs/standards.md)

## Status and licence

The MultiLingual (22M) track and the full EWoK set were not run: both datasets
are gated on Hugging Face and need an account that has accepted their terms.
The repository does not yet carry a licence file; the BabyLM corpora and the
evaluation data have their own terms.
