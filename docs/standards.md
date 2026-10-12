# Evaluation standards

These rules exist so that a number can be trusted. They are not relaxed to
make a result pass.

1. **Headline numbers come from the official pipeline** on the full suites
   (`scripts/official.py`), not from the in-house scorer or the fast subsets.
   The in-house scorer and the fast BLiMP subset are for ranking during
   screening only, and every screening table says so.
2. **Every reported score carries a 95% bootstrap interval** (`Bootstrap`), and
   two models are compared with a paired bootstrap on the same items
   (`scripts/compare.py`). A difference whose interval includes zero is
   reported as "not distinguishable", never as a win.
3. **Noise is measured, not assumed.** The screen includes dummy columns whose
   apparent effects estimate noise; any claimed gain must exceed it. Finalists
   are repeated with more seeds before any claim.
4. **Thresholds are fixed in advance.** The confident-correct margins
   (tau = 0.5, 1, 2 nats) and the significance rules are set before results
   are seen and are not tuned per model.
5. **No tuning on the test sets.** Model selection uses the held-out dev split,
   from which any line of six or more words that also occurs in training has
   been removed (`Prepare.unique`), and a contamination check reports
   benchmark overlap (`scripts/leak.py`).
6. **Negative and null results are reported** next to the positive ones, with
   the run that produced them (`metrics.json` carries code revision, dirty
   flag, library versions, device, config and data fingerprints).
7. **Stretch targets are reported as reached or not**, with the best published
   reference points beside them; they are never assumed.
8. **Interactions are looked for.** A main-effects screen is a shortlist only;
   the shortlisted factors go into a full factorial that estimates every
   two-factor interaction (`Factorial`) before an architecture is frozen.
