# Plan: rebuild, store the data offline and efficiently, then search every parameter by screening design

## Context
The previous project folder (`/Users/sachin/Documents/Codex/github-repos/new`, ~70 commits) was deleted on 2026-10-12 17:54 by something outside this session; nothing was recoverable. The user said "start from scratch". The folder has been recreated: 12 commits, all tests green, a post-commit hook writes `git bundle --all` to `~/Documents/Codex/backups/tinylm.bundle` after every commit (protects against a repeat).

Rebuilt so far (tested, committed): scaffold on standard CPython 3.14.8 (GIL build), seeding, `model.py` (local/global attention, ReLU/GELU, RMSNorm/Dynamic Tanh, dropout, untied head, codebook pyramid, `local_window` override, 50M budget guard), ledger (10-epoch cap), packing, manifest, cleaning, deterministic tokenizer, prepare (BOS separator), hybrid objective, optimizers (AdamW, Muon+AdamW), EMA. Uncommitted: `tests/test_store.py` (written, red).

What the lost runs taught us (kept in the conversation; to be re-recorded in `docs/findings/`): BLiMP seed noise at ~7M params is 0.680 +/- 0.008; data-cleaning/vocab arms were within that noise; warmup-stable-decay lowered dev bits/char by 0.013 (~10x noise); EMA no effect; official GPT-2 baseline BLiMP 0.644 (Strict-Small) / 0.734 (Strict); best published Strict-Small 0.80; official evaluator prepends BOS (train with BOS separators); throughput 39k tok/s at 7M, ~8k tok/s at 45M.

## Engineering standards (user, final wording)
**Naming**
- Names are judged in **local context**: inside module `store` a class is `Store`, not `TokenStore`; `prepare.run`, not `prepare.prepare`. No stutter (`Train.train.training` is not allowed), no redundant module/type prefix or suffix on class or function names. Compound names such as `from_tensor` are fine when they say something the context does not.
- Single words preferred for files, modules, scripts, configs (`plan.md`, `store.py`, `screen.py`); `test_<word>.py` follows pytest.
- **Nouns with behaviour**: work is done by a class named for the thing and its methods, imported by name: `Token(ids, vocab).store(directory)`, `Token.load(directory)`, `Prepare(settings).run(corpus, out)`. A module-level function that repeats its module name (`prepare.prepare`) or a class that repeats it (`store.TokenStore`) is not allowed. Module names for such classes are plural nouns (`tokens.py` holds `Token`, `prepare.py` holds `Prepare`).
- Exception classes: one per module named `Error` (`tokens.Error`), the standard-library pattern.
- No leading underscore on anything; Google style guide; Python 3.14 only.

**Data modelling and data engineering**
- Every record that crosses a function or file boundary is a **frozen dataclass** (`slots=True`), validated in `__post_init__`; no ad-hoc dicts except at the JSON/TOML/torch serialisation edge (`dataclasses.asdict` there, reconstructed and validated on read).
- Strict pipeline rules: schema-versioned metadata (`version` in every `meta.json`), SHA-256 on every stored artifact, atomic writes (temp file then rename), deterministic bytes for the same input and seed, idempotent steps (re-running changes nothing), lineage recorded (config, code revision, input checksums), and **fail loudly** on bad data (never skip, never silently repair; corrupt input raises a typed error and names the file and reason).
- Offline by construction: after `fetch`, nothing touches the network (enforced by test).

**Testing**
- Test-first. Each public function has good-path and negative-path tests (bad types, out-of-range, empty, truncated, corrupt, missing, duplicate, wrong version), plus determinism, round-trip and property tests where they apply, and end-to-end tests on tiny fixtures. Commit only when green (`pytest && ruff && pyright`).

**Reliability**
- Atomic checkpoints with exact resume; ledger-enforced epoch cap; parameter-budget guard in every entry point; external `git bundle` backup after each commit (already installed); long jobs resumable; every run's `metrics.json` reproducible from its config and seed.

**Rename pass (first task, before more code builds on the old names)**
`model`: ModelConfig -> Config, count_parameters -> count, check_parameter_budget -> check, build_attention_mask -> visibility, is_global_layer -> is_global, make_norm -> norm, init_weights -> init. `ledger`: DataBudget -> Budget, BudgetExceededError -> Error. `packing`: epoch_offset -> offset, epoch_sequences -> sequences, build_stream -> join. `manifest`: Manifest -> Corpus, file_digest -> digest, verify_file -> verify, verify_directory -> check, IntegrityError -> Error. `clean`: normalize_typography -> typography, repair_case -> recase, strip_speaker_tag -> untag, clean_lines -> apply, drop_duplicate_long_lines -> dedupe. `prepare`: prepare() + Settings -> `Prepare` dataclass with `run`, `read`, `split`, `encode` methods. `objectives`: HybridSpec -> Hybrid, mask_tokens -> corrupt, hybrid_loss -> loss. `ema`: Ema -> Average. `store.py` -> `tokens.py`: `Token` dataclass with `store`, `load`, `rows`, `from_tensor`; `Error`.

## Goal the user restated
The user expects BLiMP above 0.90 is attainable and wants the **best combination** (attention, activation, norm, and the rest) found. Part C below is exactly that combination search; the result is reported against the 0.90 target either way (best published Strict-Small found: 0.80, so the claim will be tested, not assumed).

## New user requirements folded in
1. **Dataset local, offline, efficient**: download once, pinned and checksummed; training/eval never touch the network; fastest practical access.
2. **Decide every choice from observation**: layer diagnostics; no defaults assumed ("full attention / GELU / RMSNorm / tied / dropout 0" is only the control).
3. **All parameters and functions**, not five axes; custom attention/activation/norm/dropout allowed if they beat controls.
4. **Fast**: 1-epoch screens on real data, minutes per arm; evaluate recall, superposition, accuracy, precision (confident-correct rate), size; Pareto elimination on (accuracy+precision, size).
5. Dropout and width searched jointly (needing dropout implies spare capacity -> try a slightly smaller model).

## Size targets and protocol (user, 2026-10-12)
- Final model sizes (latest wording from the user): **Strict-Small ~4M, Strict ~13M, MultiLingual ~22M** parameters (an earlier message said 9M for Strict-Small; the later 4M supersedes it) (all far below the 50M cap; smaller is the objective).
- **All architecture, activation, attention, norm, parameter and function decisions are made on Strict-Small** (faster dry runs). The Strict model is then built from those decisions at ~13M. Each screen is one full pass over the whole Strict-Small data; one full 1-epoch pass over the whole Strict data is run once to prove the finalist scales (~1.5 h at 13M).
- Reference shapes at vocab 8192, length 256: 192 wide x 6 layers = 4.3M (Strict-Small centre); 320 x 8 = 12.5M (Strict). MultiLingual needs a larger vocabulary and gated data (blocked until the user logs in to Hugging Face).
- The screening's width/depth factor levels straddle ~4M, not 45M.

## Part A: finish the rebuild (test-first, one commit each, commit only when green)
1. `src/tinylm/tokens.py` (`Token`; tests in `tests/test_tokens.py`): flat memory-mapped token file `tokens.bin` + `meta.json` (dtype, length, vocab, sha256). uint16 when vocab <= 65536 (2 bytes/token: 29 MB Strict-Small, ~290 MB Strict) else uint32. `open_store(dir, verify)` returns read-only `numpy.memmap`; `rows(starts, width)` gathers a batch with no whole-stream copy; `from_tensor` for tests. Negatives: out-of-vocab/negative ids, empty, bad dims, truncated/corrupt/missing/malformed meta, out-of-range starts.
2. `packing.epoch_starts(num_tokens, seq_len, epoch, seed, shift)`: permuted chunk start indices (replaces per-epoch full-stream copies); `epoch_sequences` re-expressed on top of it.
3. `prepare.py` writes `train/` and `dev/` stores + tokenizer + stats instead of `.pt`; byte-determinism test kept.
4. `scripts/fetch_data.py` + `src/tinylm/offline.py`: download Strict, Strict-Small and the public eval sets (`BabyLM-2026-Strict-Evals`) at pinned revisions into `data/`, write manifests under `manifests/`, then verify. `offline.enforce()` sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` and a test patches `socket` to prove prepare -> train -> eval run with zero network access. MultiLingual datasets are gated (401): out of scope until the user logs in.
5. `train.py`: schedule (cosine, WSD), hybrid or causal loss, ledger accounting, atomic checkpoints, exact resume, EMA, NaN guard, reads batches from the store (`rows`). `evaluate`.
6. `bench.py` (BLiMP scorer with BOS context, margins, precision/coverage at tau), `contamination.py`, `hf_export.py` (self-contained model file, covers all options), `experiment.py` (TOML configs, dotted overrides incl. tuples, queues, screen runs), `diagnostics.py` (attention mass beyond a window, FFN sparsity/dead units/negative mass, residual RMS, Dynamic Tanh alpha/saturation, per-block gradient norm).
7. Re-download data, re-run contamination check (previously 0.18% 6-gram overlap), re-validate the scorer against the official pipeline.

## Part B: measurement suite (what every candidate gets)
- Accuracy: BLiMP-filtered macro accuracy; dev bits/char (dev text identical across tokenizers) as the noise-resistant signal.
- Precision: pooled correct/confident and coverage at tau in {0.5, 1, 2} nats (fixed in advance).
- Size: parameters, non-embedding parameters, tokens/s.
- Memorisation recall (train vs held-out prefixes -> exact/token recall, gap); associative recall (text-form key-value probe at distances 8/32/128/200); superposition (embedding/hidden effective rank, off-diagonal cosine, small sparse autoencoder features-per-dimension). Implemented in `probes.py`, validated on planted-structure cases.
- Layer diagnostics from Part A.6 decide: full vs windowed attention (attention mass beyond 128/64 tokens + loss increase when a trained model is masked with `local_window`), GELU vs ReLU (dead units, sparsity, negative mass), RMSNorm vs Dynamic Tanh (residual growth, alpha/saturation, gradient norms), dropout type/rate (train-dev gap; joint with width).

## Part C: search every factor with a screening design (replaces one-at-a-time)
- **Design**: a 24-run Plackett-Burman two-level design screens up to 23 factors at once. Three unused columns are **dummy factors**: their apparent "effects" estimate the noise in the same experiment, so no separate noise round is needed (answers the user's earlier "no noise-first").
- **Factors (high/low)**: width (192 / 384), depth (4 / 8), FFN multiplier (2 / 4), activation (GELU / ReLU; SwiGLU added as option), norm (RMS / Dynamic Tanh), attention (full / local-global w64, every 4th global), positional (learned / RoPE, added), tied head, dropout (0 / 0.1), vocab (4k / 16k), QK-norm (off / on, added), peak LR (1e-3 / 4e-3), weight decay (0.01 / 0.3), warmup (2% / 10%), batch tokens (4k / 16k), optimizer (AdamW / Muon), schedule (cosine / WSD), objective (causal / hybrid 50%), grad clip, label smoothing (0 / 0.1, added), init std (0.02 / 0.04), + 3 dummies.
- **Cost**: 24 runs x ~6-13 min, 1 epoch Strict-Small, about 3 hours. Response: dev bits/char; secondary: BLiMP and confident-precision.
- **Analysis** (`src/tinylm/screen.py`, tested on synthetic designs with planted effects): main effects with the dummy-column noise estimate; effects below ~2x the dummy spread are declared "no evidence". Known limits, stated up front: main-effects only (two-factor interactions are aliased; a foldover doubles the runs if the top factors look interacting), two levels miss non-monotone optima (follow-up line searches on significant continuous factors), and 1 epoch cannot see late overfitting (4-epoch check on dropout/weight decay for the finalist).
- **Pareto elimination**: after screening, candidates built from significant effects are compared on (accuracy+precision, size); dominated ones drop; ties go to the smaller model.
- **Custom components**: where diagnostics expose a weakness (e.g. attention mass concentrated inside a short window -> learned-span or gated attention; ReLU dead units -> a gated or leaky variant; Dynamic Tanh saturation -> a learned-slope variant; dropout -> token/span dropout or stochastic depth), implement it behind a config option with tests, and screen it against its control in a follow-up PB or paired run.
- **Confirmation**: top 2-3 on a second seed at 4 epochs; one 10-epoch run of the finalist; official zero-shot pipeline. Report rank agreement between 1-epoch screens and a handful of 10-epoch runs.
- **Honest target**: BLiMP 0.90 stretch (best published Strict-Small found 0.80); report the result either way.

## Critical files
`src/tinylm/{model,store,packing,prepare,train,bench,diagnostics,probes,screen,experiment,hf_export,offline}.py`, `scripts/{fetch_data,run_screen,summarize_runs}.py`, `configs/screen_*.toml`, `docs/findings/*.md`, `docs/PLAN.md`.

## Verification
Per commit: `uv run pytest`, `ruff check`, `pyright` green before committing. Offline proof: socket-patched end-to-end test. Storage: size = 2 bytes/token test; throughput test that `rows` does not copy the stream. Screening: planted-effect synthetic designs recover known effects and flag dummies as null. Final: official pipeline on the exported finalist.

## Status (2026-10-12, evening)

Done and tested (389 tests, 99.9% statement coverage, strict types, warnings
are errors, versioned pre-commit gate): offline pinned data (`Fetch`,
`Offline`), compact memory-mapped token stores (`Token`), streaming `Prepare`
(2.7 GB peak on Strict, byte-identical to the old output) with decontaminated
dev set, `Train` (versioned, fingerprinted, durable checkpoints), `Experiment`
(provenance, restore, resolved config saved), `Bench` (BLiMP-style scorer with
bootstrap intervals, confident-correct precision, calibration), `Layers`,
`Probes`, `Screen` (24-run Plackett-Burman with dummy-column noise, foldover,
full factorial with Lenth), `Export` (generated wrapper, verified for every
model option), `Official` (official-pipeline wrapper), `Device`, `jobs`.
`docs/review.md` lists every finding with evidence and status.

Running now: the 24-run, 20-factor screen on Strict-Small from a frozen
snapshot of the corrected code (about 2.5 hours; ~6 minutes per run).

Remaining, in order (nothing below has been done):
1. Analyse the screen (`scripts/screen.py analyse`); flag factors above the
   dummy-column noise. Many arms sit at chance on BLiMP after one epoch, so
   bits/char carries the analysis.
2. Full 2^k factorial (`Factorial`) on the flagged factors to find
   interactions; mid-range line searches for continuous ones.
3. Replicated confirmation (more seeds, 4 epochs) of the Pareto survivors;
   dropout and weight decay need this, since 1 epoch cannot show them.
4. Final models at the stated sizes: Strict-Small ~4M, Strict ~13M; run the
   official pipeline (`scripts/official.py`) and compare with a paired
   bootstrap. MultiLingual (~22M) is blocked on gated Hugging Face data.
5. Open review items: re-measure `torch.compile` on an idle GPU (H4), near
   duplicates (D2), languages without spaces (D5), in-house vs official scorer
   cross-check on a trained model (V5).
6. Report: update `docs/paper.md` only with measured, interval-qualified
   numbers; the 0.90 BLiMP stretch target is reported as reached or not.
