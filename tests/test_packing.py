"""Tests for per-epoch chunk-start planning."""

import pytest
import torch

from tinylm import packing


def test_ten_epochs_have_ten_distinct_offsets_and_cycle():
    offsets = [packing.offset(e, 100) for e in range(10)]
    assert len(set(offsets)) == 10 and all(0 <= o < 100 for o in offsets)
    assert packing.offset(0, 100) == packing.offset(10, 100)


@pytest.mark.parametrize("bad", [(-1, 100), (0, 0)])
def test_offset_rejects_bad_arguments(bad: tuple[int, int]):
    with pytest.raises(ValueError):
        packing.offset(*bad)


def test_starts_are_non_overlapping_chunk_starts_covering_the_stream():
    starts = packing.starts(1000, 64, epoch=3, seed=1)
    expected = packing.offset(3, 64) + 64 * torch.arange(len(starts))
    assert torch.equal(starts.sort().values, expected)
    assert starts.unique().numel() == starts.numel()
    assert int(starts.max()) + 64 <= 1000
    assert 1000 - 64 * len(starts) < 128


def test_deterministic_seed_changes_order_only():
    a = packing.starts(2000, 32, 1, 9)
    assert torch.equal(a, packing.starts(2000, 32, 1, 9))
    b = packing.starts(2000, 32, 1, 10)
    assert not torch.equal(a, b)
    assert torch.equal(a.sort().values, b.sort().values)


def test_epochs_use_different_boundaries_unless_shift_is_off():
    a = {int(x) for x in packing.starts(500, 32, 0, 9)}
    b = {int(x) for x in packing.starts(500, 32, 1, 9)}
    assert a.isdisjoint(b)
    c = {int(x) for x in packing.starts(500, 32, 0, 9, shift=False)}
    d = {int(x) for x in packing.starts(500, 32, 7, 9, shift=False)}
    assert c == d


def test_short_stream_gives_no_starts():
    assert packing.starts(10, 64, 0, 0).numel() == 0


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="width"):
        packing.starts(100, 0, 0, 0)
    with pytest.raises(ValueError, match="epoch"):
        packing.starts(100, 8, -1, 0)
    with pytest.raises(ValueError, match="length"):
        packing.starts(-5, 8, 0, 0)


def test_join_separates_lines_and_skips_empty():
    assert [int(x) for x in packing.join([[5, 6], [], [7]], 1)] == [
        5,
        6,
        1,
        7,
        1,
    ]
    assert packing.join([], 1).numel() == 0
