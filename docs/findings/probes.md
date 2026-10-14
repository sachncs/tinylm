# Probes and layer diagnostics (final models)

Run with `scripts/probe.py runs/final/<run>`; the full report is written to
`probes.json` in the run folder. Both models are the final 10-epoch ones. All
probes run on the CPU in eval mode. Samples are small (hundreds of lines,
16 sequences of 256 tokens), so treat the numbers as descriptive, not as tests.

## Memorisation (teacher-forced recall of the second half of a line)
| Model | Train: token / exact | Held-out dev: token / exact | Lines (train / dev) |
|---|---|---|---|
| Small 3.7M | 0.293 / 0.008 | 0.283 / 0.005 | 258 / 221 |
| Strict 12.9M | 0.327 / 0.004 | 0.304 / 0.004 | 234 / 236 |

The gap between training lines and held-out lines is 0.01 to 0.02 in token
recall and under 0.01 in exact recall. Neither model has memorised its training
text; recall is next-token prediction, not retrieval.

## Associative (key-value) recall
A random key and value, then filler tokens, then the key again: the model must
output the value. Chance is about 0.00006 (1 in 16,384).

| Model | 8 tokens | 32 | 128 | 200 |
|---|---|---|---|---|
| Small 3.7M | 0.018 | 0.012 | 0.002 | 0.006 |
| Strict 12.9M | 0.066 | 0.040 | 0.004 | 0.010 |

Both are above chance but low, and the score falls with distance. At this size
and amount of training the models have, at most, weak copy-from-context ability.
This probe uses random tokens, which are unlike the training text.

## Attention reach
Mean attention probability that lands more than 16, 64 or 128 tokens back, per
layer (see `probes.json` for every layer). In the Small model the share beyond
16 tokens is 0.02 to 0.77 depending on the layer, beyond 64 tokens at most 0.41,
and beyond 128 tokens at most 0.16. Most attention is local, but several layers
use the longer range, which is consistent with local/global windows making no
measurable difference at these lengths.

## Feed-forward units
No GELU unit is dead in any layer of either model (dead share 0.0).

## Superposition
Effective rank (participation ratio) and mean absolute cosine between rows of
the representations, computed without centring.

| Representation | Small rank (width 128) | Strict rank (width 256) |
|---|---|---|
| Embedding table | 45.0 | 55.0 |
| After the first block | 30.9 | 27.1 |
| Middle blocks | 5.7 to 7.1 | 13.1 to 17.2 |
| After the last block | 2.0 | 2.1 |

Effective rank falls with depth, and the last block's states are close to
low-dimensional, with mean absolute cosine 0.69 in the Small model. A shared
mean direction likely dominates because the states were not centred; this was
not separated out. A sparse autoencoder with 4x the width on the middle layer
finds 1.9 (Small) and 1.3 (Strict) alive features per dimension, reconstructing
99% and 87% of the variance. This is evidence that the middle layer carries more
features than dimensions, not a measurement of how many the model uses.
