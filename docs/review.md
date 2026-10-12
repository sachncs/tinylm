# Principal-engineer review (2026-10-12)

Scope: whole repo after the rebuild (src, scripts, configs, tests, data flow).
Method: read the code, grep for patterns, measure on the real corpus where a
claim could be checked. Each finding has evidence and a status. Statuses:
FIXED (with commit), OPEN (planned), CHECKED (suspected, measured, no defect).

## Engineering
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| E1 | P1 | Resume never checks that config or data match the checkpoint; a changed run silently continues from foreign state. | `Checkpoint` has no version, config hash or data hash (`train.py`). | OPEN |
| E2 | P1 | Checkpoint and metrics writes are not durable: no fsync of file or directory; metrics.json is not atomic. | `grep fsync src` = 0. | OPEN |
| E3 | P1 | Results carry no provenance (git revision, dirty flag, library versions, device, config and data fingerprints), so a number cannot be traced to its code. | `Metrics` has no such fields. | OPEN |
| E4 | P2 | Output is `print` only: no structured logs, no run log file. | 1 `print` in src, 0 `logging`. | OPEN |
| E5 | P2 | 17 type/lint suppressions remain; each hides a check. | `grep -rn "pyright: ignore\|noqa"` = 17. | OPEN (justify or remove each) |
| E6 | P2 | The quality gate is a script, not enforced: a red commit is possible (already happened twice). | history. | OPEN (pre-commit hook) |
| E7 | P2 | No coverage threshold; no property-based tests (hypothesis). | pyproject. | OPEN |
| E8 | P3 | `Offline` patches process-global socket functions; not thread-safe. Documented limit. | `offline.py`. | OPEN (document) |
| E9 | P3 | Long runs in one process (queue script) leak memory and one crash kills the batch. | seen risk. | FIXED for `screen` (subprocess per run); `batch.py` OPEN |

## Optimisation
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| O1 | P1 | Training forces a GPU->CPU sync every step (`isfinite(loss)`, `float(loss)`), serialising the device and breaking overlap with data loading. | `train.py:141,150`. | OPEN |
| O2 | P1 | Attention masks are rebuilt on CPU and copied to the device on every forward; pure-causal full attention does not use the fused `is_causal` path. | `model.py:284,290`. | OPEN |
| O3 | P2 | `Prepare` holds every cleaned line of the corpus in Python lists (multiple copies); Strict needs GBs of RAM. | `prepare.py`, `tokenizer.train` copies lines. | OPEN |
| O4 | P3 | Scoring tokenises sentences one by one. | `bench.py`. | OPEN |
| O5 | P3 | bf16/fp16 gave no speed-up on MPS (measured 7.4k/8.1k vs 7.9k tokens/s): documented, no change. | earlier measurement. | CHECKED |

## Hardware acceleration
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| H1 | P1 | `device = "mps"` is trusted blindly: no availability check, and MPS CPU-fallback env var is not verified off (silent CPU fallback hides slow or wrong paths). | configs, `train.py`. | OPEN |
| H2 | P2 | Throughput is wall-clock including evaluation and callbacks, and not synchronised with the device. | `experiment.py:246-248`. | OPEN |
| H3 | P2 | No device-memory release between runs in one process. | seen risk. | FIXED for `screen` (processes); OPEN elsewhere |
| H4 | P3 | `torch.compile`/graph capture not assessed on MPS. | -- | OPEN (measure, report) |

## Data processing and handling
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| D1 | P1 | **Dev set leaks training text**: 4.5% of dev lines with >= 6 words (167/3,674) are exact copies of training lines (33% of all dev lines, mostly short utterances). Dev loss and bits/char are optimistic. | measured on Strict-Small. | OPEN |
| D2 | P2 | Duplicate removal is per source only; cross-source duplicates remain; no near-duplicate removal. | `prepare.py:88`. | OPEN |
| D3 | P2 | No Unicode normalisation (NFC) before tokenising. | `clean.py`. | OPEN |
| D4 | P2 | Output lineage is partial: inputs are hashed but the settings, tokenizer hash and code revision are not recorded in `stats.json`. | `prepare.py`. | OPEN |
| D5 | P3 | Whitespace word counts used for dedupe/stats mis-count languages without spaces (needed for the MultiLingual track). | `clean.dedupe`. | OPEN |
| D6 | P3 | `splitlines()` could split on form-feed and other separators. | measured: differs from `split("\n")` only by the trailing newline in all six sources. | CHECKED (no defect) |

## Evaluation (raised by the user: do not lower the bar)
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| V1 | P1 | Reported scores have no uncertainty: no confidence intervals, no paired comparison between models. | `Bench.Report`. | OPEN |
| V2 | P1 | Screening uses the fast BLiMP subset; headline numbers must come from the full official suites (BLiMP, supplement, EWoK, entity tracking, COMPS, reading, GLUE), not the subset. | `configs/screen.toml`. | OPEN (rule + official-pipeline script) |
| V3 | P1 | The 24-run screen estimates main effects only; interactions are aliased and two-level designs miss mid-range optima. | design theory; reviewer note. | OPEN (foldover + full factorial on the top factors; then replicated confirmation) |
| V4 | P2 | Single seed per screen run; winners need replicates before any claim. | design. | OPEN |
| V5 | P2 | The in-house scorer was validated against the official pipeline once, on a model that no longer exists. | `docs/baselines.md` (lost). | OPEN |
| V6 | P2 | Thresholds (tau) are fixed in advance (good), but no calibration measure is reported next to precision. | `Bench`. | OPEN |

## Rules adopted
Warnings are errors; nothing is silenced without a written reason; failures stop
the run with the cause; results are reported as measured, negative ones too;
evaluation settings are never relaxed to make a result pass.
