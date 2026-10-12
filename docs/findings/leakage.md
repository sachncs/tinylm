# Finding: benchmark leakage (rebuilt pipeline)

2026-10-12. `scripts/leak.py data/strict data/evals/.../blimp_filtered` on the
cleaned Strict corpus (offline): **0.18% of eligible BLiMP sentences share a
6-gram with training** (limit 2%). Same value as before the rebuild.
Dataset token counts also reproduced exactly: Strict-Small 14,387,899 and
Strict 145,849,281 training tokens at vocab 8192.
