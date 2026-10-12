"""Shared fixtures."""

import collections.abc

import pytest
import torch


@pytest.fixture(autouse=True)
def fresh_compiler() -> collections.abc.Iterator[None]:
    """Clears torch.compile caches around each test.

    Stale Dynamo/Inductor state from an earlier compiled model made a later
    MPS compile fail in code generation when many tests share one process;
    real runs use one process per run and are unaffected.
    """
    torch.compiler.reset()
    yield
    torch.compiler.reset()
