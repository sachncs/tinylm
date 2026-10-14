# TinyLM: how small can an accurate and precise language model be? (working draft)

Status: results for the 3.7M Strict-Small and 12.9M Strict models are final for
this draft; MultiLingual is blocked (see "Pending"). Every number below was measured
in this repository; nothing is quoted from memory. Official scores are point
values from the official pipeline, which keeps no item-level output; beside them
are 95% bootstrap intervals from the in-house scorer on the same items, which
agreed with the official figure within 0.3 points on these models.

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
| COMPS (official) | 52.46 |
| EWoK, fast subset (official, 1,100 items) | 53.91 |
| Reading: eye tracking / self-paced (official) | 0.30 / 0.02 |
| BLiMP full set, in-house scorer | 67.43 [67.11, 67.75] |
| Supplement full set, in-house scorer | 57.15 [54.06, 60.29] |
| Fast BLiMP subset (in-house) | 0.679 [0.672, 0.686] |
| Confident-correct rate at tau = 1 nat | 0.713, covering 81.5% of pairs |
| Dev bits per character | 1.405 |

The in-house scorer agrees with the official one to 0.04 (BLiMP) and 0.11
(supplement) points on this model; earlier on a different model it read 1.5
points low, so only official numbers are headline numbers. COMPS, the EWoK fast subset and the
reading task were run later and are included (the extra files are a separate
pinned download, `data/extra`, with its own manifest; the EWoK archive uses the
password published in the evaluation pipeline's README). EWoK is the fast
subset only, not the full set, whose licence terms must be accepted on Hugging
Face. The reading scores are the pipeline's own eye-tracking and self-paced
reading figures, run with `python -m evaluation_pipeline.reading.run`; both
models are near zero, i.e. no measurable alignment with human reading times.
 A rerun scored BLiMP 67.40 against 67.39
earlier, so the official scorer is not bit-for-bit repeatable at the second
decimal.

## Result: Strict, 12.9M parameters, 10 epochs (`runs/final/strict10`)
Same recipe scaled up: width 256, 11 layers, 4 heads, exclusive attention in
layers 3 to 7, lr 3e-3, Strict data (137.7M training tokens), 16.4 hours.
| Measure | Value |
|---|---|
| BLiMP-filtered (official pipeline) | **74.80** |
| BLiMP supplement (official) | 60.23 |
| Entity tracking (official) | 15.75 |
| COMPS (official) | 55.82 |
| EWoK, fast subset (official, 1,100 items) | 54.73 |
| Reading: eye tracking / self-paced (official) | 0.43 / 0.00 |
| BLiMP full set, in-house scorer | 75.09 [74.80, 75.36] |
| Supplement full set, in-house scorer | 60.80 [57.70, 63.92] |
| Fast BLiMP subset (in-house) | 0.748 [0.742, 0.754] |
| Confident-correct rate at tau = 1 nat | 0.790, covering 83.4% of pairs |
| Dev bits per character | 1.271 |

Going from 3.7M to 12.9M parameters (and 10x the data) raised official BLiMP by
7.4 points and the supplement by 3.2, but entity tracking fell from 17.88 to
15.75; the latter is a single run and was not investigated. The in-house scorer
read 0.3 and 0.6 points above the official one here.

## GLUE-style fine-tuning (official pipeline, validation accuracy)
Each task fine-tunes the final model once (10 epochs, lr 3e-5, batch 32, sequence
256, seed 42, last-token pooling; `scripts/glue.py`). The majority baseline is
the accuracy of always predicting the commonest validation label.
| Task | Majority baseline | Strict-Small 3.7M | Strict 12.9M |
|---|---|---|---|
| BOOLQ | 0.640 | 0.643 | 0.675 |
| MNLI | 0.357 | 0.549 | 0.582 |
| MRPC | 0.681 | 0.740 (F1 0.826) | 0.725 (F1 0.818) |
| MULTIRC | 0.575 | 0.566 | 0.589 |
| QQP | 0.628 | 0.745 (F1 0.654) | 0.760 (F1 0.673) |
| RTE | 0.540 | 0.576 | 0.698 |
| WSC | 0.615 | 0.673 | 0.596 |
| Mean accuracy | 0.577 | 0.642 | 0.661 |

MNLI, QQP and MRPC are clearly above the baseline for both models. BoolQ and
MultiRC are barely above it, and WSC for the 12.9M model (0.596) is *below* its
baseline (0.615). The small tasks (RTE, WSC, MRPC) have a few hundred
validation items and one fine-tuning seed, so differences of several points
there are within noise; the larger model's lower WSC and MRPC scores should not
be read as a real regression.

Reference point: the organisers' GPT-2 baseline (124M parameters, 20 epochs, BPE 32k) is listed at BLiMP 64.4 on Strict-Small and 73.4 on Strict (`docs/baselines.md`). The models here score 67.4 and 74.8 with 33 and 10 times fewer parameters. The baseline was not rerun here and it is unconfirmed that it was scored on the same filtered BLiMP set and pipeline version, so read the comparison as indicative.

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

Official scores of the 4-epoch pair, seed 0 only (`official.json` in each run):
baseline BLiMP 64.93 / supplement 57.51 / entity tracking 20.29; recency +
exclusive attention 66.48 / 56.90 / 20.64. Seed 0 was the most favourable seed
for the combination in the in-house comparison (+0.010; seeds 1 and 2: -0.002,
-0.009), so the +1.55 BLiMP points are not evidence of a general gain.

## Limits, stated plainly
- The architecture choices bought about 0.01 bits/char and no measurable BLiMP;
  most of the benchmark gain came from more epochs (0.657 at 4 epochs, 0.679 at
  10 on the fast subset).
- Fast BLiMP differences under about 0.01 are inside seed noise; bits per
  character was the more reliable signal.
- Single final run per size; the final models have no seed replicates. Recency
  and exclusive attention were never ablated at the 12.9M scale.
- The 0.90 BLiMP target is **not reached** at either size: 67.4 official
  (3.7M) and 74.8 official (12.9M). The best published
  Strict-Small result found while planning was about 0.80.
- Contamination: 167 dev lines that also occurred in training were removed; no
  BLiMP text is used in training or in any training signal.

## Pending
- MultiLingual (22M): blocked on gated Hugging Face data (needs `hf auth login`
  and acceptance of the dataset terms); not trained.
- Not run: full EWoK (licence); GLUE used a single fine-tuning seed; seed replicates and
  ablations of the final recipe at 12.9M.
