# Attention variants and training settings, Strict-Small 3.7M (2026-10-13)

Frozen candidate (`configs/small.toml`: d=128, 8 layers, 2 heads, RMSNorm, GELU,
tied 16k vocabulary, Muon, WSD) trained for 4 epochs of Strict-Small on three
seeds per arm. Scores are fast-BLiMP accuracy (in-house scorer, reads about 1
point below the official one) and dev bits per character. Seeds of the baseline
agree within 0.0005 bits/char, so differences above ~0.003 are real.

| Arm | Parameters | Bits/char | BLiMP fast | Verdict |
|---|---|---|---|---|
| Baseline (dropout 0) | 3,704,960 | 1.4425 | 0.657 | reference |
| Dropout 0.1 | 3,704,960 | 1.4754 | 0.646 | worse on every seed: dropped |
| Recency penalty | 3,704,976 | 1.4380 | 0.661 | better bits/char on all 3 seeds; BLiMP mixed (+0.014, -0.008, +0.005) |
| Exclusive self attention, layers 2-5 | 3,704,960 | 1.4366 | 0.656 | better bits/char on all 3 seeds; BLiMP indistinguishable |
| Recency + exclusive (1 seed) | 3,704,976 | 1.4302 | 0.666 | promising; replicated below |
| Cross attention to 32 slots, layers 2-5 | 3,869,312 | 1.4417 | 0.661 | no gain for +4.4% parameters: dropped |
| Cross + exclusive | 3,869,312 | 1.4373 | 0.653 | no better than exclusive alone: dropped |
| lr 1e-2, batch 256 (one run, warmup 50) | 3,704,960 | 1.468 | 0.643 | worse; fewer steps; three settings changed together |

Notes: a first attempt at batch 256 without micro-batching swapped memory and
was stopped; `train.accumulate` was added so large batches run as micro-batches.
No arm moves fast BLiMP by more than about 0.01, which is the order of the
seed-to-seed noise; bits/char is the more reliable signal.
