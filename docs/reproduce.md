# Reproducing the results

Everything runs on one Apple-silicon laptop (M3 Pro, 18 GB) with Python 3.14.
After the one-off data download nothing touches the network.

```sh
git clone https://github.com/sachncs/tinylm && cd tinylm
uv sync --group official                 # Python 3.14.8, locked dependencies
                                         # (the group adds what the reading and GLUE evaluators need)
./scripts/install.sh                     # optional: enables the commit gate
git clone https://github.com/babylm-org/babylm-eval external/babylm-eval
git -C external/babylm-eval checkout 6f825c291e2c4c78ad33b1935fd64d45f52642dc
uv run python scripts/data.py            # pinned, checksummed downloads
uv run python scripts/build.py           # offline token stores
```

## Train

```sh
# Strict-Small 3.7M (about 45 minutes)
uv run python scripts/batch.py configs/final.toml
# Strict 12.9M (about 16 hours; interrupt-safe, re-run to resume)
uv run python scripts/batch.py configs/finalstrict.toml
```

Runs checkpoint atomically and resume exactly; Ctrl-C or SIGTERM stops at a
step boundary after saving. Configuration or data changes are detected by a
fingerprint and refused on resume.

## Evaluate

```sh
uv run python scripts/official.py runs/final/small10   # BLiMP, supplement, entity tracking, COMPS
uv run python scripts/glue.py runs/final/small10       # GLUE fine-tuning
uv run python scripts/compare.py runs/a runs/b blimp_fast   # paired bootstrap
```

## Experiments in this repository

| Question | Config |
|---|---|
| Which of 20 factors matter? (24-run Plackett-Burman screen) | `scripts/screen.py` |
| Interactions of the top five factors (32-run factorial) | `scripts/follow.py` |
| Dropout, recency, exclusive and cross attention (4 epochs, 3 seeds) | `configs/confirm.toml`, `recency.toml`, `exclusive.toml`, `cross.toml` |
| Large learning rate and batch | `configs/rate.toml` |

The findings are in [Screen](findings/screen.md) and [Attention](findings/attention.md).

## Quality gate

`./scripts/gate.sh` runs the formatter check, lint, strict type checking and
the test suite (warnings are errors, coverage floor 97%). About 480 tests cover
unit, property-based and end-to-end behaviour, including negative cases.

## Hardware notes

The GPU is Apple MPS; there is no CUDA path. `torch.compile` is opt-in
(`train.compile=true`) and measured about 20% faster per step. Large batches
use `train.accumulate` micro-batches, because a batch of 256 sequences swaps
memory on an 18 GB machine.
