# Models

Two decoder-only language models trained from scratch on the BabyLM 2026
Strict-Small and Strict corpora. Weights are published as release assets of
the repository (`SHA256SUMS` lists the checksums); each archive contains a
Hugging Face model folder that loads with `trust_remote_code=True`.

| Model | Parameters | Training data | Epochs | Official BLiMP |
|---|---|---|---|---|
| `tinylm-small-3.7M` | 3,704,976 | BabyLM Strict-Small (14.4M tokens) | 10 | 67.4 |
| `tinylm-strict-12.9M` | 12,916,524 | BabyLM Strict (137.7M tokens) | 10 | 74.8 |

These scores are above the organisers' GPT-2 baseline as listed in the leaderboard
repository (64.4 and 73.4 BLiMP with 124M parameters; not rerun here, so
indicative), far below the best published BabyLM results (about 80 on
Strict-Small), and well short of the 90 this project hoped for. See
[Results](paper.md) for every benchmark with its limits.

## Use

```sh
curl -LO https://github.com/sachncs/tinylm/releases/download/v0.1.0/tinylm-small-3.7M.tar.gz
tar xzf tinylm-small-3.7M.tar.gz
```

```python
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

path = "tinylm-small-3.7M"
tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True
).eval()

ids = tokenizer("The children are playing in the park.", return_tensors="pt")
with torch.no_grad():
    logits = model(input_ids=ids["input_ids"]).logits  # (1, length, 16384)
```

The models are tiny and trained on child-directed and simple text, so they are
research artefacts for studying data-efficient language modelling, not
general-purpose assistants. Sequences are limited to 256 tokens.

`AutoModel` is also supported: it returns the hidden state at the last real
token of each row (padding on either side is handled exactly), which is what
the official GLUE fine-tuning pipeline pools.

## Architecture

Pre-norm decoder-only transformer with RMSNorm, GELU feed-forward layers, tied
embeddings and a 16k BPE vocabulary, plus two attention changes chosen by
measurement: a learned per-head recency penalty on attention scores, and
exclusive self attention (Zhai, 2026) in the middle layers. Full details and the
evidence for each choice are in [Architecture](architecture.md).

| | Small | Strict |
|---|---|---|
| Width / layers / heads | 128 / 8 / 2 | 256 / 11 / 4 |
| Exclusive attention layers | 2 to 5 | 3 to 7 |
| Optimizer | Muon (matrices) + AdamW | same |
| Peak learning rate | 4e-3 | 3e-3 |
| Training time (Apple M3 Pro) | about 45 minutes | about 16 hours |

## Scores

Official evaluation pipeline unless noted (accuracy in percent).

| Task | Small 3.7M | Strict 12.9M |
|---|---|---|
| BLiMP (filtered) | 67.4 | 74.8 |
| BLiMP supplement | 57.0 | 60.2 |
| Entity tracking | 17.9 | 15.8 |
| COMPS | 52.5 | 55.8 |
| EWoK (fast subset) | 53.9 | 54.7 |
| GLUE mean accuracy (single run per task) | 64.2 | 66.1 |

Each archive also carries `scores.json` with the numbers for that model.

## Data and licence notes

The models were trained only on the BabyLM 2026 corpora, downloaded at pinned
revisions and checksummed (`manifests/`). Exact duplicate lines between the
training and development text were removed, and no benchmark text is used for
training. The corpora are subject to their own terms; check them before
reusing the weights. The repository does not yet carry a licence file.
