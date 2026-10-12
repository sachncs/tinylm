"""Deterministic seeding for reproducible runs."""

import random

import torch

MAX_SEED = 2**32 - 1


def set_seed(seed: int) -> None:
    """Seeds Python and torch random number generators.

    Args:
        seed: Value in [0, 2**32 - 1].

    Raises:
        ValueError: If seed is outside the valid range.
    """
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")
    random.seed(seed)
    torch.random.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
