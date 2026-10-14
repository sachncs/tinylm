# Finding: 24-run, 20-factor screen on Strict-Small (1 epoch)

2026-10-12. Code revision `c7919af` (frozen snapshot), Strict-Small, BPE vocab
4k/16k as screened, one seed per run, 1 epoch with the learning-rate schedule
compressed into it. Responses: dev bits/char, fast-BLiMP accuracy, precision at
margin >= 1 nat. Models 1.3M to 9.9M parameters. All 24 runs completed.
Raw: `runs/screen/x*/metrics.json`; analysis: `scripts/screen.py analyse`.

## What the data support

Noise was estimated from 3 dummy columns **plus the known-null factor
`shift`** (it cannot change a 1-epoch run, because epoch 0 always starts at
offset 0). Using only the dummies, `shift` itself came out "significant" on all
three responses (ratio 3.5, 4.7, 3.0): a false positive, so the dummies alone
understate the noise (two-factor interactions leak into main effects in this
design). The stricter estimate is used below.

| Factor (high vs low) | bits/char | BLiMP | precision | Reading |
|----------------------|-----------|-------|-----------|---------|
| norm: Dynamic Tanh vs RMSNorm | +0.421 (sig.) | -0.044 (sig.) | -0.041 (sig.) | **RMSNorm better**, all three responses |
| optimizer: Muon vs AdamW | -0.388 (sig.) | +0.038 (sig.) | +0.032 (sig.) | **Muon better**, all three responses |
| vocab: 16k vs 4k | -0.091 | +0.027 (sig.) | +0.041 (sig.) | larger vocabulary helps accuracy and precision |
| batch 64 vs 16 | +0.114 | -0.026 | -0.023 (sig.) | smaller batch better, but confounded with step count at 1 epoch |
| seq_len 256 vs 128 | +0.025 | -0.012 | -0.020 (sig.) | shorter sequences better on precision |
| peak lr 4e-3 vs 1e-3 | +0.117 | +0.004 | +0.020 (sig.) | precision only |
| width, depth, untied head, local/global attention, GELU vs ReLU, dropout, warmup, weight decay, clip, floor, EMA, hybrid loss, WSD | below the noise | below | below | **no detectable effect at 1 epoch** |

(sig. = beyond the stricter noise estimate; effects are high minus low level.)

## What they do not support
- **Size**: width, depth and the untied head show no effect on any response, and
  the smallest runs (1.3M parameters, BLiMP 0.557 and 0.582) sit close to much
  larger ones. This agrees with the dataset, not the model, being the limit.
- **Dropout, weight decay, EMA, WSD** cannot be judged at 1 epoch (their
  benefit appears with repeated data), so "no effect" there means "not
  visible", not "useless".
- **Batch, lr, warmup** change the number of optimizer steps in a 1-epoch run,
  so their screen effects mix quality with step count.
- Many arms sit at chance on BLiMP (about 0.50), which is why bits/char is
  the primary response.
- Main effects only: two-factor interactions are aliased. The factorial in
  `scripts/follow.py` (norm, optimizer, vocab, seq_len, lr) is the check; its
  result is in [Factorial](factorial.md).

## Pareto front (accuracy, precision, size), single seeds
x23 1.34M BLiMP 0.582 / precision 0.600; x06 3.70M 0.589 / 0.612;
x18 3.71M 0.604 / 0.626; x11 6.74M 0.617 / 0.638. These are shortlisting
evidence, not a ranking: each is one seed with an interval of about +/-0.007.

## Method note
The best 1-epoch BLiMP score is 0.617, far below the 0.90 stretch target. One
epoch is a screening instrument, not a measure of final quality.
