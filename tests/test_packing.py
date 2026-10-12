"""Tests for per-epoch sequence packing."""

import pytest
import torch

from tinylm import packing


def stream(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long)


def starts(t: torch.Tensor) -> set[int]:
    return {int(x) for x in t[:, 0]}


def test_ten_epochs_have_ten_distinct_offsets_and_cycle():
    offsets = [packing.offset(e, 100) for e in range(10)]
    assert len(set(offsets)) == 10 and all(0 <= o < 100 for o in offsets)
    assert packing.offset(0, 100) == packing.offset(10, 100)


@pytest.mark.parametrize("bad", [(-1, 100), (0, 0)])
def test_offset_rejects_bad_arguments(bad: tuple[int, int]):
    with pytest.raises(ValueError):
        packing.offset(*bad)


def test_rows_are_contiguous_non_overlapping_runs_covering_the_stream():
    rows = packing.sequences(stream(1000), 64, epoch=3, seed=1)
    assert rows.shape[1] == 64
    assert bool((rows[:, 1:] - rows[:, :-1] == 1).all())
    flat = rows.flatten()
    assert flat.unique().numel() == flat.numel()
    assert 1000 - rows.numel() < 128


def test_deterministic_and_seed_changes_order_only():
    a = packing.sequences(stream(2000), 32, 1, 9)
    assert torch.equal(a, packing.sequences(stream(2000), 32, 1, 9))
    b = packing.sequences(stream(2000), 32, 1, 10)
    assert not torch.equal(a, b)
    assert starts(a) == starts(b)


def test_epochs_use_different_boundaries_unless_shift_is_off():
    a = packing.sequences(stream(500), 32, 0, 9)
    b = packing.sequences(stream(500), 32, 1, 9)
    assert starts(a).isdisjoint(starts(b))
    c = packing.sequences(stream(500), 32, 0, 9, shift=False)
    d = packing.sequences(stream(500), 32, 7, 9, shift=False)
    assert starts(c) == starts(d)


def test_short_stream_gives_empty_result():
    assert packing.sequences(stream(10), 64, 0, 0).shape == (0, 64)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="seq_len"):
        packing.sequences(stream(100), 0, 0, 0)
    with pytest.raises(ValueError, match="1-D"):
        packing.sequences(torch.zeros(4, 4, dtype=torch.long), 2, 0, 0)
    with pytest.raises(ValueError, match="epoch"):
        packing.sequences(stream(100), 8, -1, 0)


def test_build_stream_separates_lines_and_skips_empty():
    assert [int(x) for x in packing.join([[5, 6], [], [7]], 1)] == [
        5,
        6,
        1,
        7,
        1,
    ]
    assert packing.join([], 1).numel() == 0
