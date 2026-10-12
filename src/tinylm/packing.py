"""Packs a token stream into fixed-length sequences, one view per epoch."""

import collections.abc

import torch

NUM_VIEWS = 10


def offset(epoch: int, seq_len: int) -> int:
    """Returns the stream offset at which an epoch starts chunking.

    Offsets are spread evenly over one sequence length so that ten epochs cut
    the stream at ten different places and no two epochs share boundaries.

    Raises:
        ValueError: If epoch is negative or seq_len is not positive.
    """
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    return (epoch % NUM_VIEWS) * seq_len // NUM_VIEWS


def sequences(
    stream: torch.Tensor,
    seq_len: int,
    epoch: int,
    seed: int,
    *,
    shift: bool = True,
) -> torch.Tensor:
    """Cuts the stream into shuffled sequences for one epoch.

    Args:
        stream: 1-D integer tensor of token ids.
        seq_len: Tokens per sequence.
        epoch: Zero-based epoch; selects the chunk boundaries.
        seed: Seed for the sequence order.
        shift: If False, every epoch uses the boundaries of epoch 0.

    Returns:
        Tensor of shape (num_sequences, seq_len). Each row is a contiguous
        run of the stream; rows never overlap within an epoch.

    Raises:
        ValueError: If seq_len or epoch is out of range or stream is not 1-D.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if stream.ndim != 1:
        raise ValueError("stream must be 1-D")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    start = offset(epoch, seq_len) if shift else 0
    count = max(0, (stream.numel() - start) // seq_len)
    rows = stream[start : start + count * seq_len].reshape(count, seq_len)
    generator = torch.Generator()
    generator.manual_seed(seed * 1_000_003 + epoch)
    return rows[torch.randperm(count, generator=generator)]


def join(
    lines: collections.abc.Iterable[list[int]], separator: int
) -> torch.Tensor:
    """Concatenates tokenised lines, ending each with the separator id."""
    ids: list[int] = []
    for line in lines:
        if line:
            ids.extend(line)
            ids.append(separator)
    return torch.tensor(ids, dtype=torch.long)
