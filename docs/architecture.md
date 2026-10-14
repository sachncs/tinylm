# Architecture

Every choice is labelled with its evidence. "Measured" means a run in this
repository (screens used 1 epoch of Strict-Small, confirmations 4 epochs with
three seeds per arm); "not distinguishable" means the effect was inside the
noise estimated from dummy columns, a known-null factor and seed replicates.
Configs: `configs/small.toml` and `configs/strict.toml`.

## Model (`src/tinylm/model.py`)
| Choice | Decision | Evidence |
|---|---|---|
| Layout | pre-norm, decoder-only transformer, learned positions, sequence 256 | sequence 128 vs 256 not distinguishable (factorial); 256 kept for the evaluation context |
| Norm | **RMSNorm** | Dynamic Tanh worse: BLiMP -0.057, bits/char +0.66 (Lenth-significant); with AdamW it collapses to chance (0.503) |
| Attention | full causal, plus **two additions** below | local/global windows had no detectable effect in the screen; kept as an option |
| Recency penalty | **on**: learned per-head `slope * distance` subtracted from scores, slopes start at 0 | bits/char 1.438 vs 1.443 on all 3 seeds; BLiMP mixed |
| Exclusive self attention | **on** in the middle layers (2 to 5 of 8; 3 to 7 of 11) | bits/char 1.437 vs 1.443 on all 3 seeds; BLiMP not distinguishable |
| Both together | used in the final models | bits/char 1.433 vs 1.443 on all 3 seeds (-0.0096); BLiMP 0.657 vs 0.657 |
| Cross attention to memory slots | **off** (option kept) | no gain for +4.4% parameters |
| Activation | **GELU** | GELU vs ReLU not distinguishable |
| Head | **tied** embeddings | untied no detectable effect; tying saves V*d parameters, which matters at 4M |
| Vocabulary | **BPE 16k** | 16k vs 4k: bits/char -0.12 (significant); BLiMP +0.016 (below margin). The table costs 2.1M parameters at d=128, a deliberate trade |
| Dropout | **0** | measured: 0.1 is worse on every seed (1.475 vs 1.443 bits/char) |
| Width/depth | by budget, see below | no size effect detected between 1.3M and 10M at 1 epoch |

Exclusive self attention follows Zhai (2026, arXiv:2603.09078): each head's
output has its component along the token's own value vector removed. Its
independent reproductions were not found; the gain here is small.

## Training
| Choice | Decision | Evidence |
|---|---|---|
| Optimizer | **Muon on hidden matrices, AdamW on embeddings and norms** | bits/char -0.67, BLiMP +0.053 (significant); interaction with norm: the best cell is RMSNorm+Muon |
| Objective | **causal only** | hybrid masked objective no effect at 1 epoch |
| Schedule | warmup-stable-decay, floor 0.1 | no detectable effect at 1 epoch; the decay phase drives the final-epoch loss drop |
| LR | 4e-3 (Small), 3e-3 (Strict) | 1e-3 vs 4e-3 not distinguishable; lr 1e-2 with batch 256 was worse (one run) |
| EMA, weight decay, clip | EMA off; decay 0.1; clip 1.0 | no detectable effect; weight decay never tested at multiple epochs |
| Epochs | 10 (the allowed maximum) | 4 to 10 epochs lifted fast BLiMP 0.657 to 0.679 (Small) |
| Compile | opt-in `train.compile` | 20% faster on MPS, losses equal to 1e-6 |
| Large batches | `train.accumulate` micro-batches | batch 256 swaps memory on 18 GB without it |

## Sizes (vocab 16384, tied)
| Track | Config | Parameters | Status |
|---|---|---|---|
| Strict-Small | d=128, L=8, 2 heads | **3,704,976** | trained, 10 epochs |
| Strict | d=256, L=11, 4 heads | **12,916,524** | trained, 10 epochs |
| MultiLingual (~22M) | not decided | n/a | blocked: gated data |

Width versus depth is not resolved by the evidence, so the shapes follow
common small-model practice (deeper, 64-wide heads) and are themselves a
hypothesis; no alternative shape was trained to completion.

## Engineering
Fingerprinted atomic checkpoints with exact resume and signal-safe halting;
durable writes; provenance in every result; one process per run; offline
enforcement; streaming data preparation with dev decontamination; official
evaluation pipeline plus in-house scorer with bootstrap intervals; strict
gate (format, lint, pyright strict, warnings-as-errors tests, 97% coverage).

## Not decided or measured
Weight decay and the schedule at multiple epochs; seed replicates of the
final models; ablations of recency and exclusive attention at 12.9M;
MultiLingual. The 0.90 BLiMP target is not reached: official BLiMP is 67.4
(3.7M) and 74.8 (12.9M).
