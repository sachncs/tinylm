# Reference points

Official BabyLM 2026 GPT-2 baselines (HF Space `BabyLM-community/BabyLM-Leaderboard-2026`,
`baseline_results/`), 124M parameters, 20 epochs, BPE 32k:

| Track | BLiMP | MultiBLiMP-eng | HellaSwag | Winogrande | XStoryCloze |
|-------|-------|----------------|-----------|------------|-------------|
| Strict-Small (10M words) | 0.6439 | 0.8039 | 0.257 | 0.5103 | 0.4837 |
| Strict (100M words) | 0.7343 | 0.8857 | 0.2649 | 0.5144 | 0.4923 |

Published best (arXiv 2504.08165, 2023 round): BLiMP 0.80 Strict-Small
(ELC-BERT), 0.86 Strict (BootBERT). Current leaderboard entries are not
readable (results dataset is private).

This repository's own numbers are in each run's `metrics.json` and
`official.json`, never copied here by hand.
