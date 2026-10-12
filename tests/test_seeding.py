"""Tests for deterministic seeding."""

import random

import pytest
import torch

from tinylm import seeding


def test_same_seed_reproduces_python_and_torch_streams():
    seeding.set_seed(123)
    first = (random.random(), torch.rand(3))
    seeding.set_seed(123)
    second = (random.random(), torch.rand(3))
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])


def test_different_seeds_differ():
    seeding.set_seed(1)
    a = torch.rand(8)
    seeding.set_seed(2)
    assert not torch.equal(a, torch.rand(8))


@pytest.mark.parametrize("bad_seed", [-1, 2**32])
def test_rejects_out_of_range_seed(bad_seed: int):
    with pytest.raises(ValueError, match="seed"):
        seeding.set_seed(bad_seed)
