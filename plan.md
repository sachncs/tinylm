# Plan and status

## Goal
Find the smallest language model that is accurate and precise on the BabyLM
2026 benchmark, by measurement: Strict-Small about 4M parameters, Strict about
13M, MultiLingual about 22M. All decisions were made on Strict-Small, and the
Strict model was built from them. A BLiMP score of 0.90 was the stretch target.

## Done
- Offline, pinned, checksummed data; streaming preparation with a
  decontaminated dev set; memory-mapped token stores.
- Test-first training library: fingerprinted atomic checkpoints with exact
  resume, Muon and AdamW, WSD schedules, optional `torch.compile`, gradient
  accumulation, strict types and warnings as errors (`scripts/gate.sh`).
- A 24-run Plackett-Burman screen of 20 factors with dummy columns
  (`docs/findings/screen.md`), a 32-run factorial on the top five factors, and
  4-epoch, 3-seed comparisons of dropout, a recency penalty, exclusive self
  attention, cross attention and a large-batch recipe
  (`docs/findings/attention.md`).
- Final models: Strict-Small 3.7M and Strict 12.9M, both 10 epochs, scored with
  the official pipeline (BLiMP, supplement, entity tracking, COMPS, EWoK fast
  subset, reading) and GLUE fine-tuning; probes in `docs/findings/probes.md`.
- Weights published as release assets; documentation site built with Astro.

## Result
Official BLiMP 67.4 (3.7M) and 74.8 (12.9M). The 0.90 target is not reached;
the best published Strict-Small result found while planning was about 0.80.
Details and limits: `docs/paper.md`.

## Open
- MultiLingual (~22M): blocked, the Hugging Face datasets are gated and need an
  account that has accepted their terms.
- Full EWoK (gated the same way); only the fast subset was scored.
- Seed replicates of the final models, and ablations of recency and exclusive
  attention at 12.9M.
- Weight decay and the schedule were never tested over several epochs.
- A change that could move BLiMP by several points (a tuned hybrid
  causal/masked objective, or distillation from a teacher trained on the same
  data) has not been tried.
- No licence file has been chosen.
