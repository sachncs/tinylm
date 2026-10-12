# Principal-engineer review (2026-10-12)

Scope: whole repo after the rebuild (src, scripts, configs, tests, data flow).
Method: read the code, grep for patterns, measure on the real corpus where a
claim could be checked. Each finding has evidence and a status. Statuses:
FIXED (with commit), OPEN (planned), CHECKED (suspected, measured, no defect).

## Engineering
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| E1 | P1 | Resume never checked that config or data match the checkpoint; a changed run silently continued from foreign state. | `Checkpoint` had no version, config hash or data hash. | FIXED (fingerprint of config, model and data; versioned checkpoint; mismatch raises) |
| E2 | P1 | Checkpoint and metrics writes were not durable: no fsync of file or directory; metrics.json not atomic. | `grep fsync src` = 0. | FIXED (`atomic.durable`: temp file, fsync, rename, directory fsync) |
| E3 | P1 | Results carried no provenance (git revision, dirty flag, library versions, device, config and data fingerprints). | `Metrics` had no such fields. | FIXED (`Provenance` in every `metrics.json`) |
| E4 | P2 | Output was `print` only: no structured logs, no run log file. | 1 `print` in src, 0 `logging`. | FIXED (`logs.setup`: one format, console plus optional file; library code uses `logging`) |
| E5 | P2 | 17 type/lint suppressions hid checks. | `grep -rn "pyright: ignore\|noqa"` = 17. | FIXED (16 removed by fixing the typing; 1 left, `noqa: PLR0913` on `Train.run`, with its reason written next to it; `reportUnnecessaryTypeIgnoreComment` and `RUF100` now fail the gate on any stale suppression) |
| E6 | P2 | The quality gate was a script, not enforced: a red commit was possible (happened twice). | history. | FIXED (versioned `pre-commit` hook runs the whole gate; tested to block a failing gate) |
| E7 | P2 | No coverage threshold; no property-based tests (hypothesis). | pyproject. | FIXED (gate fails below 97% coverage; measured 99.9% statement coverage, which shows paths were run, not that they are right; seven property-based tests with `hypothesis`) |
| E8 | P3 | `Offline` patches process-global socket functions; not thread-safe. | `offline.py`. | FIXED (a lock makes nesting or use from a second thread raise `Error` instead of restoring the wrong state; the lock and the environment flags are released if entering fails; tested) |
| E10 | P1 | **Found while hardening, introduced by me:** with weight averaging on, `bits_per_char` was measured on the live weights while the benchmark score used the averaged weights, so one result mixed two models; and the final checkpoint held the live weights, not the returned model. Any run with `ema > 0` is affected, including the EMA arms of the screen started before the fix. | `Experiment.execute` / `Train.run` order of `finish()` and the last checkpoint. | FIXED (final loss is evaluated on the returned model; the final checkpoint stores it; `Experiment.restore` and `experiment.json` added; invariants tested). The screen running at the time is a preview only for the EMA factor on bits/char; the screen is re-run on the corrected code. |
| E9 | P3 | Long runs in one process leak memory and one crash kills the batch. | seen risk. | FIXED (`jobs.run_all`: one process per run, `caffeinate`, first failure stops the batch with its exit status; used by `screen.py` and `batch.py`) |

| R1 | P1 | A long run killed by Ctrl-C, SIGTERM or a crash lost all progress and the next attempt restarted from zero. | `run.py` had no signal handling and no resume path. | FIXED (`Train.run(halt=...)` stops at a step boundary and checkpoints; SIGINT/SIGTERM set the halt flag; `Experiment.run(resume=True)` continues exactly (dev-loss history persisted per epoch); `jobs.command` passes `--resume`, so re-running a batch continues the interrupted run; a changed configuration or dataset is refused; tested to resume to the identical result) |

## Optimisation
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| O1 | P3 | Training syncs with the device every step (`isfinite(loss)`, `float(loss)`). Hypothesis: this serialises the device. **Measured: 162.8 ms vs 162.0 ms per step with the sync removed (0.5%)**, so it is not a bottleneck; per-step checking is kept because it reports the exact failing step. | `train.py`, MPS benchmark. | CHECKED (no change) |
| O2 | P3 | Attention masks were rebuilt on CPU and copied to the device on every forward; pure-causal full attention did not use `is_causal`. Hypothesis: a speed-up. **Measured: 154.0 ms vs 155.1 ms per step (under 1%)**. Masks are now cached and global layers take the fused path (tested equal to the explicit-mask path), kept for cleanliness, not speed. | `model.py`, MPS benchmark. | FIXED (no speed gain) |
| O3 | P2 | `Prepare` held every cleaned line of the corpus in Python lists (multiple copies). **Measured: 7.36 GB peak resident for Strict.** | `prepare.py`, `tokenizer.train` copied lines. | FIXED (streams through temporary files; only 64-bit fingerprints of long training lines stay in memory). **Measured: 2.74 GB peak, 127 s vs 138 s, and the train and dev streams are byte-identical to the old output** (golden test on the fixture, fingerprint comparison on full Strict) |
| O4 | P3 | Scoring tokenised sentences one by one. | `bench.py`. | FIXED (`encode_batch`) |
| O5 | P3 | bf16/fp16 gave no speed-up on MPS (measured 7.4k/8.1k vs 7.9k tokens/s): documented, no change. | earlier measurement. | CHECKED |

## Hardware acceleration
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| H1 | P1 | (fixed in `device.py`: availability and fallback checks, loud errors) `device = "mps"` was trusted blindly: no availability check, and MPS CPU-fallback env var is not verified off (silent CPU fallback hides slow or wrong paths). | configs, `train.py`. | FIXED (`Train.run` resolves the device first and raises with the reason) |
| H2 | P2 | Throughput was wall-clock including evaluation and callbacks, not synchronised with the device. | `experiment.py`. | FIXED (`Result.seconds`: device-synchronised, callbacks excluded) |
| H3 | P2 | No device-memory release between runs in one process. | seen risk. | FIXED (one process per run; `Device.release` exists for in-process use) |
| H4 | P2 | `torch.compile` on MPS. **Measured under identical GPU contention: 187 ms vs 242 ms per step (about 23% faster)**, with graph breaks reported at the cached-mask lookup and a one-time compile cost that is large next to a 5-minute screen. Worth it for long final runs only if it holds in isolation and stays numerically equivalent. | benchmark (screen was using the GPU, so only the pair is comparable). | OPEN (re-measure with an idle GPU incl. compile time; fix graph breaks; check numerics) |

## Data processing and handling
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| D1 | P1 | **Dev set leaked training text**: 4.5% of dev lines with >= 6 words (167/3,674) were exact copies of training lines (33% of all dev lines, mostly short utterances). Dev loss and bits/char were optimistic. | measured on Strict-Small. | FIXED (`Prepare.unique` drops those lines; `stats.json` records the count, 167, which matches the hand measurement) |
| D2 | P2 | Duplicate removal was per source only; no near-duplicate removal. **Measured:** lines that repeat after normalising case and punctuation are 233 extra lines in Strict-Small (0.022%) and 1,831 in Strict (0.016%); lines sharing an 8-word opening are under 0.07%. | `prepare.py`, measurement script. | CHECKED (exact long-line dedupe is shared across sources; fuzzy deduplication would act on well under 0.1% of lines, so it was not built) |
| D3 | P2 | No Unicode normalisation (NFC) before tokenising. | `clean.py`. | FIXED |
| D4 | P2 | Output lineage was partial. | `prepare.py`. | FIXED (settings and tokenizer hash in `stats.json`, version 2; code revision is in each run's provenance) |
| D5 | P3 | Whitespace word counts mis-counted languages without spaces (needed for the MultiLingual track). | `clean.dedupe`. | FIXED (`clean.words`: each Han, Hiragana or Katakana character counts as a word; used by dedupe and dev decontamination; tested on Chinese, Japanese and mixed text) |
| D6 | P3 | `splitlines()` could split on form-feed and other separators. | measured: differs from `split("\n")` only by the trailing newline in all six sources. | CHECKED (no defect) |

## Evaluation (raised by the user: do not lower the bar)
| # | Sev | Finding | Evidence | Status |
|---|-----|---------|----------|--------|
| V1 | P1 | Reported scores had no uncertainty: no confidence intervals, no paired comparison between models. | `Bench.Report`. | FIXED (`Bootstrap`: 95% intervals on every benchmark mean; per-item outcomes saved; `scripts/compare.py` for paired differences) |
| V2 | P1 | (script and rule exist: `scripts/official.py`, `docs/standards.md`; running it on finalists is pending) Screening uses the fast BLiMP subset; headline numbers must come from the full official suites (BLiMP, supplement, EWoK, entity tracking, COMPS, reading, GLUE), not the subset. | `configs/screen.toml`. | OPEN (rule + official-pipeline script) |
| V3 | P1 | The 24-run screen estimates main effects only; interactions are aliased and two-level designs miss mid-range optima. | design theory; reviewer note. | PARTLY FIXED (`Design.foldover` and `Factorial`: full 2^k design with all two-factor interactions judged by Lenth's method, tested on planted interactions; the follow-up runs on the screen's top factors are OPEN until the screen finishes; mid-range optima need line searches, OPEN) |
| V4 | P2 | Single seed per screen run; winners need replicates before any claim. | design. | OPEN |
| V5 | P2 | The in-house scorer was validated against the official pipeline once, on a model that no longer exists. | `docs/baselines.md` (lost). | OPEN |
| V6 | P2 | Thresholds (tau) are fixed in advance (good), but no calibration measure was reported next to precision. | `Bench`. | FIXED (`Calibration`: reliability table and expected calibration error of sigmoid(|margin|) against correctness; validated on synthetic calibrated and over-confident data) |

## Rules adopted
Warnings are errors; nothing is silenced without a written reason; failures stop
the run with the cause; results are reported as measured, negative ones too;
evaluation settings are never relaxed to make a result pass.
