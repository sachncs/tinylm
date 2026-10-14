# Finding: benchmark leakage (rebuilt pipeline)

2026-10-12. `scripts/leak.py data/strict data/evals/.../blimp_filtered` on the
cleaned Strict corpus (offline): **0.18% of eligible BLiMP sentences share a
6-gram with training** (limit 2%). Same value as before the rebuild.
Training token counts after cleaning and dev decontamination: Strict-Small
13,546,776 and Strict 137,654,021 at a 16k vocabulary (Strict-Small is
14,387,899 at the 8k vocabulary used by the screens).

## Dev-set leakage and its fix
The held-out development split also leaked: before the fix, 167 of the
Strict-Small dev lines with at least six words (4.5%) were exact copies of
training lines, which made dev loss and bits per character optimistic.
`Prepare.unique` now removes them from the dev split, and `stats.json` records
the count (167 for Strict-Small, 3,879 for Strict). The 0.18% overlap above was
measured on the cleaned training corpus against BLiMP, not on the dev split.
