# Finalised architecture (after the review refactor)

Every choice is labelled with its evidence. "Measured" means a run in this
repo (1 epoch of Strict-Small, fast BLiMP subset, bits per character); "not
distinguishable" means the effect was inside the noise estimated from dummy
columns and a known-null factor; "assumed" means no measurement yet.

## Model (`src/tinylm/model.py`)
| Choice | Decision | Evidence |
|---|---|---|
| Layout | pre-norm, decoder-only transformer, learned positions, sequence 256 | sequence 128 vs 256 not distinguishable (factorial); 256 kept for the evaluation context |
| Norm | **RMSNorm** | Dynamic Tanh worse: BLiMP -0.057, bits/char +0.66 (Lenth significant); with AdamW it collapses to chance (0.503) |
| Attention | **full causal** (fused path) | local/global (w128) no detectable effect in the screen; kept as an option |
| Activation | **GELU** | GELU vs ReLU not distinguishable |
| Head | **tied** embeddings | untied no detectable effect; tying saves V*d parameters, which matters at 4M |
| Vocabulary | **BPE 16k** | 16k vs 4k: bits/char -0.12 (significant); BLiMP +0.016 (below margin). The table costs 2.1M parameters at d=128, a deliberate trade |
| Dropout | **0** | 1 epoch cannot judge; revisit with 4-epoch runs |
| Width/depth | by budget, see below | no size effect detected between 1.3M and 10M at 1 epoch |

## Training
| Choice | Decision | Evidence |
|---|---|---|
| Optimizer | **Muon on hidden matrices, AdamW on embeddings and norms** | bits/char -0.67, BLiMP +0.053 (significant); interaction with norm: the best cell is RMSNorm+Muon (1.560 bpc, 0.612) |
| Objective | **causal only** | hybrid masked objective no effect |
| Schedule | warmup-stable-decay, floor 0.1 | WSD no detectable effect at 1 epoch; earlier runs showed a 0.013 bits/char gain; confirm |
| LR | 4e-3 (Small), 3e-3 (Strict) | 1e-3 vs 4e-3 not distinguishable; larger models get the lower rate (assumed) |
| EMA, weight decay, clip | EMA off; decay 0.1; clip 1.0 | no detectable effect; decay needs multi-epoch confirmation |
| Compile | opt-in `Train.compile` for long runs | 20% faster on MPS, losses equal to 1e-6 |

## Sizes (vocab 16384, tied; `model.count`)
| Track | Config | Parameters |
|---|---|---|
| Strict-Small (~4M) | d=128, L=8, 2 heads | **3.70M** (`configs/small.toml`) |
| Strict (~13M) | d=256, L=11, 4 heads | **12.92M** (`configs/strict.toml`) |
| MultiLingual (~22M) | not decided | blocked: gated data needs `hf auth login` |

Alternatives checked for the same budgets: 144x7 = 4.14M, 160x5 = 4.20M,
288x8 = 12.76M, 320x6 = 12.70M. Width versus depth is not resolved by the
evidence, so the shapes follow common small-model practice (deeper, 64-wide
heads) and are themselves a hypothesis for the confirmation stage.

## Engineering (what the refactor added)
Fingerprinted atomic checkpoints with exact resume and signal-safe halting;
durable writes; provenance in every result; one process per run; offline
enforcement; streaming data preparation with dev decontamination; official
evaluation pipeline plus in-house scorer with bootstrap intervals; strict
gate (format, lint, pyright strict, warnings-as-errors tests, 97% coverage).

## Not yet decided or measured
Dropout and weight decay at multiple epochs; replicates over seeds (V4);
official scores of the finalists (V2); MultiLingual. The 0.90 BLiMP target is
unsupported so far: the best measured fast-BLiMP is 0.635 at 1 epoch, and the
best published Strict-Small result is 0.80.
