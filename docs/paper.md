# TinyLM: how small can an accurate and precise language model be? (working draft)

Status: results for the 3.7M Strict-Small model are final for this draft; the
12.9M Strict model is training (see "Pending"). Every number below was measured
in this repository; nothing is quoted from memory. Intervals are 95% bootstrap
intervals over benchmark items unless stated.

## Setup
BabyLM 2026 Strict-Small (14.4M training tokens after cleaning and
decontamination), local pinned data, no network at training time. Decoder-only
transformer, 8 layers, width 128, 2 heads, RMSNorm, GELU, tied 16k BPE
embeddings, Muon on hidden matrices and AdamW elsewhere, WSD schedule, dropout
0, sequence 256, batch 32. Added: a learned per-head recency penalty on
attention scores and exclusive self attention (Zhai 2026, arXiv:2603.09078) in
layers 2 to 5. 3,704,976 parameters. Ten epochs, the allowed maximum.

## Result: Strict-Small, 3.7M parameters, 10 epochs (`runs/final/small10`)
| Measure | Value |
|---|---|
| BLiMP-filtered (official pipeline) | **67.39** |
| BLiMP supplement (official) | 57.04 |
| Entity tracking (official) | 17.88 |
| Fast BLiMP subset (in-house) | 0.679 [0.672, 0.686] |
| Confident-correct rate at tau = 1 nat | 0.713, covering 81.5% of pairs |
| Dev bits per character | 1.405 |

The in-house scorer agrees with the official one to 0.04 (BLiMP) and 0.11
(supplement) points on this model; earlier on a different model it read 1.5
points low, so only official numbers are headline numbers. EWoK, COMPS, reading
and GLUE-style tasks have not been run.

## What was decided and on what evidence (Strict-Small, 4 epochs, 3 seeds per arm)
| Choice | Evidence |
|---|---|
| RMSNorm over Dynamic Tanh | bits/char +0.66 and BLiMP -0.057 for Dynamic Tanh (Lenth-significant in a 32-run factorial); with AdamW it collapses to chance |
| Muon over AdamW | bits/char -0.67, BLiMP +0.053 (same factorial); the norm x optimizer interaction is also significant |
| 16k vocabulary over 4k | bits/char -0.12 (significant); BLiMP +0.016 (below margin) |
| Dropout 0 | dropout 0.1: 1.475 vs 1.443 bits/char, BLiMP 0.646 vs 0.657, worse on every seed |
| Recency penalty | bits/char 1.438 vs 1.443 on all 3 seeds; BLiMP not distinguishable |
| Exclusive attention, layers 2-5 | bits/char 1.437 vs 1.443 on all 3 seeds; BLiMP not distinguishable |
| Both together | bits/char 1.433 vs 1.443 on all 3 seeds (-0.0096); BLiMP 0.657 vs 0.657 |
| Rejected | cross attention to 32 memory slots (no gain for +4.4% parameters), lr 1e-2 with batch 256 (worse), local/global attention, GELU vs ReLU, width and depth at 1 to 10M, EMA, hybrid masked objective: no detectable effect at 1 epoch |

Details and raw tables: `docs/findings/`. Screening was a 24-run
Plackett-Burman design with dummy columns and a known-null factor to calibrate
the noise.

## Limits, stated plainly
- The architecture choices bought about 0.01 bits/char and no measurable BLiMP;
  most of the benchmark gain came from more epochs (0.657 at 4 epochs, 0.679 at
  10 on the fast subset).
- Fast BLiMP differences under about 0.01 are inside seed noise; bits per
  character was the more reliable signal.
- Single final run per size; the final models have no seed replicates.
- The 0.90 BLiMP target is **not reached**: 67.4 official. The best published
  Strict-Small result found while planning was about 0.80.
- Contamination: 167 dev lines that also occurred in training were removed; no
  BLiMP text is used in training or in any training signal.

## Pending
- Strict model (12.9M, 10 epochs, about 15 hours): result and its official
  scores will be added here when the run finishes.
- MultiLingual (22M): blocked on gated Hugging Face data (needs `hf auth login`
  and acceptance of the dataset terms).
- Recency and exclusive attention were chosen on the 3.7M model and are applied
  to the 12.9M model untested at that size.
