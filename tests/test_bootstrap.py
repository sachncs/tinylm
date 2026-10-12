"""Tests for bootstrap intervals on benchmark outcomes."""

import pytest
import torch

from tinylm.bootstrap import Bootstrap, Interval


def outcomes(n: int, rate: float, seed: int = 0) -> list[bool]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return [bool(x) for x in (torch.rand(n, generator=generator) < rate)]


def test_all_correct_has_a_degenerate_interval_at_one():
    found = Bootstrap().mean({"a": [True] * 50, "b": [True] * 80})
    assert (found.low, found.high) == (1.0, 1.0)
    assert found.estimate == 1.0 and found.level == 0.95


def test_interval_brackets_the_estimate_and_matches_the_normal_width():
    group = {"only": outcomes(1000, 0.5)}
    found = Bootstrap(resamples=2000).mean(group)
    assert found.low < found.estimate < found.high
    expected = 2 * 1.96 * (0.25 / 1000) ** 0.5
    assert (found.high - found.low) == pytest.approx(expected, rel=0.2)


def test_more_items_give_a_narrower_interval_and_runs_are_deterministic():
    small = Bootstrap().mean({"g": outcomes(100, 0.6)})
    large = Bootstrap().mean({"g": outcomes(5000, 0.6)})
    assert (large.high - large.low) < (small.high - small.low)
    assert Bootstrap(seed=3).mean({"g": outcomes(200, 0.5)}) == Bootstrap(
        seed=3
    ).mean({"g": outcomes(200, 0.5)})


def test_macro_average_weights_groups_equally():
    found = Bootstrap().mean({"easy": [True] * 10, "hard": [False] * 990})
    assert found.estimate == pytest.approx(0.5)


def test_paired_difference_detects_a_real_gap_and_a_non_gap():
    better = {"g": [True] * 1000}
    worse = {"g": outcomes(1000, 0.7, seed=1)}
    gap = Bootstrap().paired(better, worse)
    assert gap.low > 0.2 and gap.estimate == pytest.approx(
        1 - sum(worse["g"]) / 1000
    )
    same = Bootstrap().paired(worse, worse)
    assert (same.low, same.high, same.estimate) == (0.0, 0.0, 0.0)
    noise_a, noise_b = outcomes(400, 0.6, 1), outcomes(400, 0.6, 2)
    tie = Bootstrap().paired({"g": noise_a}, {"g": noise_b})
    assert tie.low < 0.0 < tie.high


def test_validation():
    with pytest.raises(ValueError, match="level"):
        Bootstrap(level=1.0)
    with pytest.raises(ValueError, match="resamples"):
        Bootstrap(resamples=10)
    with pytest.raises(ValueError, match="groups"):
        Bootstrap().mean({})
    with pytest.raises(ValueError, match="empty"):
        Bootstrap().mean({"a": []})
    with pytest.raises(ValueError, match="same groups"):
        Bootstrap().paired({"a": [True]}, {"b": [True]})
    with pytest.raises(ValueError, match="same length"):
        Bootstrap().paired({"a": [True, False]}, {"a": [True]})


def test_interval_validates_order():
    with pytest.raises(ValueError, match="order"):
        Interval(estimate=0.5, low=0.8, high=0.7, level=0.95)
