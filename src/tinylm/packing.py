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


def starts(
    length: int, width: int, epoch: int, seed: int, *, shift: bool = True
) -> torch.Tensor:
    """Plans the shuffled chunk starts of one epoch.

    Chunks of `width` tokens never overlap within an epoch; their boundaries
    move with the epoch so repeated passes see different sequences. Only the
    start indices are materialised; pair with Token.rows to read the data.

    Args:
        length: Number of tokens in the stream.
        width: Tokens per chunk.
        epoch: Zero-based epoch; selects the chunk boundaries.
        seed: Seed for the chunk order.
        shift: If False, every epoch uses the boundaries of epoch 0.

    Returns:
        Long tensor of chunk start indices in training order.

    Raises:
        ValueError: If length is negative, width is not positive or epoch is
            negative.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if width <= 0:
        raise ValueError("width must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    first = offset(epoch, width) if shift else 0
    count = max(0, (length - first) // width)
    generator = torch.Generator()
    generator.manual_seed(seed * 1_000_003 + epoch)
    order = torch.randperm(count, generator=generator)
    return first + width * order


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
