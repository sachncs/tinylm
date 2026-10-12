"""Masked-token corruption and the hybrid causal/masked training loss.

The masked objective follows GPT-BERT: the network sees a corrupted sequence
with bidirectional attention, and the output at position i - 1 predicts the
original token at position i. Causal and masked training therefore share the
same output convention and the same weights.
"""

import dataclasses

import torch
from torch.nn import functional

from tinylm import model

MASK_REPLACE = 0.8
RANDOM_REPLACE = 0.1
TOKEN_NDIM = 2
MIN_SPLIT_ROWS = 2


@dataclasses.dataclass(frozen=True)
class HybridSpec:
    """Hybrid objective settings.

    Attributes:
        masked_fraction: Share of each batch trained with the masked
            objective; the rest is trained causally.
        mask_rate: Probability that a position is selected for prediction.
        mask_id: Token id used for the mask symbol.
    """

    masked_fraction: float
    mask_rate: float
    mask_id: int

    def __post_init__(self) -> None:
        """Validates ranges."""
        if not 0.0 <= self.masked_fraction <= 1.0:
            raise ValueError("masked_fraction must be in [0, 1]")
        if not 0.0 < self.mask_rate < 1.0:
            raise ValueError("mask_rate must be in (0, 1)")


def mask_tokens(
    tokens: torch.Tensor,
    mask_rate: float,
    mask_id: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corrupts tokens BERT-style.

    Selected positions are replaced by mask_id (80%), a random token (10%)
    or left unchanged (10%). Position 0 is never selected because no output
    precedes it.

    Returns:
        (corrupted tokens, boolean selection mask), both of tokens' shape.

    Raises:
        ValueError: If mask_rate is outside (0, 1) or tokens is not 2-D.
    """
    if not 0.0 < mask_rate < 1.0:
        raise ValueError("mask_rate must be in (0, 1)")
    if tokens.ndim != TOKEN_NDIM:
        raise ValueError(
            f"expected 2-D tokens, got shape {tuple(tokens.shape)}"
        )
    selected = torch.rand(tokens.shape, generator=generator) < mask_rate
    selected[:, 0] = False
    roll = torch.rand(tokens.shape, generator=generator)
    random_tokens = torch.randint(
        0, vocab_size, tokens.shape, generator=generator
    )
    corrupted = tokens.clone()
    use_mask = selected & (roll < MASK_REPLACE)
    use_random = (
        selected
        & (roll >= MASK_REPLACE)
        & (roll < MASK_REPLACE + RANDOM_REPLACE)
    )
    corrupted[use_mask] = mask_id
    corrupted[use_random] = random_tokens[use_random]
    return corrupted, selected


def hybrid_loss(
    net: model.TinyLM,
    batch: torch.Tensor,
    spec: HybridSpec,
    generator: torch.Generator,
) -> torch.Tensor:
    """Token-weighted mean of causal and masked next-token losses.

    The first round(masked_fraction * batch) rows use the masked objective;
    the remaining rows use the causal objective.

    Args:
        net: Model.
        batch: Integer tensor (batch, length + 1).
        spec: Objective settings.
        generator: CPU generator driving the corruption.

    Raises:
        ValueError: If masked_fraction is strictly between 0 and 1 but the
            batch has fewer than two rows.
    """
    rows = batch.shape[0]
    if 0.0 < spec.masked_fraction < 1.0 and rows < MIN_SPLIT_ROWS:
        raise ValueError("batch must have at least 2 rows to split")
    count = min(max(round(spec.masked_fraction * rows), 0), rows)
    total = torch.zeros((), device=batch.device)
    weight = 0
    if count < rows:
        causal = batch[count:]
        logits = net(causal[:, :-1], causal=True)
        total = total + functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            causal[:, 1:].reshape(-1),
            reduction="sum",
        )
        weight += causal[:, 1:].numel()
    if count > 0:
        masked = batch[:count]
        corrupted, selected = mask_tokens(
            masked.cpu(),
            spec.mask_rate,
            spec.mask_id,
            net.config.vocab_size,
            generator,
        )
        corrupted = corrupted.to(batch.device)
        selected = selected.to(batch.device)
        logits = net(corrupted[:, :-1], causal=False)
        targets = masked[:, 1:]
        picked = selected[:, 1:]
        if bool(picked.any()):
            total = total + functional.cross_entropy(
                logits[picked], targets[picked], reduction="sum"
            )
            weight += int(picked.sum())
    return total / max(weight, 1)
