# Finding: 32-run factorial on the five shortlisted factors (Strict-Small, 1 epoch)

2026-10-13. A full 2^5 design (`scripts/follow.py`, base `configs/follow.toml`):
norm (RMSNorm / Dynamic Tanh), optimizer (AdamW / Muon), vocabulary (4k / 16k),
sequence length (128 / 256) and peak learning rate (1e-3 / 4e-3). The base is
d=192, 6 layers, GELU, full attention, tied head, no dropout, WSD, batch 32.
One seed per cell; effects are judged with Lenth's method, which estimates the
noise from the effects themselves (no replicates needed) and also examines all
two-factor interactions. This is a 1-epoch instrument, a shortlist for the
longer runs in [Attention](attention.md), not a final ranking.

## Significant effects (high minus low level; margin from Lenth)
| Response | Effects beyond the margin |
|---|---|
| Dev bits/char (margin 0.115, lower is better) | Muon -0.669, norm (Dynamic Tanh) +0.656, **norm x Muon -0.556**, vocab 16k -0.119 |
| Fast BLiMP (margin 0.017) | norm -0.057, Muon +0.053, **norm x Muon +0.035** (vocab +0.016 is just below the margin) |
| Confident-correct rate at 1 nat (margin 0.026) | norm -0.046, Muon +0.042 |

Sequence length and learning rate are not significant on any response.

## The interaction, as cell means (8 runs per cell)
| | AdamW | Muon |
|---|---|---|
| RMSNorm | 1.672 bits/char, BLiMP 0.595 | **1.560**, **0.612** |
| Dynamic Tanh | 2.884, 0.503 (chance) | 1.659, 0.591 |

Dynamic Tanh with AdamW fails outright; with Muon it comes close to RMSNorm with
AdamW but stays behind RMSNorm with Muon. RMSNorm with Muon is the best cell on
every response. The best single run (RMSNorm, Muon, 16k vocabulary, 5.85M
parameters) reached 1.514 bits/char and fast BLiMP 0.635.

## Limits
One seed per cell, so the effects are screening-grade; the 8-run cells give a
within-cell spread but the cells differ in other factors. Only five factors were
varied, and the width, depth and attention variants are not part of this design.
