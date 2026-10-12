"""Tests for train/eval n-gram contamination detection."""

import pytest

from tinylm.overlap import Error, Overlap


def test_ngrams_are_lowercase_and_punctuation_free():
    assert Overlap(2).ngrams("The cat, sat!") == {
        ("the", "cat"),
        ("cat", "sat"),
    }
    assert Overlap(3).ngrams("hello there") == set()


@pytest.mark.parametrize("n", [0, -1])
def test_rejects_non_positive_n(n: int):
    with pytest.raises(ValueError, match="n"):
        Overlap(n)


def test_rate_detects_verbatim_partial_and_none():
    train = ["well the quick brown fox jumps over the lazy dog today"]
    assert (
        Overlap(5).rate(train, ["The quick brown fox jumps over the lazy dog."])
        == 1.0
    )
    assert (
        Overlap(4).rate(
            ["she sells sea shells by the sea shore"],
            ["the quick brown fox jumps over the lazy dog"],
        )
        == 0.0
    )
    half = Overlap(4).rate(
        ["the quick brown fox jumps over"],
        ["the quick brown fox jumps", "completely different words here"],
    )
    assert half == 0.5


def test_short_sentences_are_skipped_not_counted():
    assert Overlap(4).rate(["a b c d e f"], ["a b", "a b c d"]) == 1.0
    with pytest.raises(ValueError, match="eligible"):
        Overlap(4).rate(["a b c d"], ["x y"])


def test_empty_inputs_are_errors():
    with pytest.raises(ValueError, match="empty"):
        Overlap(2).rate([], ["a b c d"])
    with pytest.raises(ValueError, match="empty"):
        Overlap(2).rate(["a b"], [])


def test_check_raises_above_threshold_and_returns_rate_below():
    train = ["the quick brown fox jumps over the lazy dog"]
    with pytest.raises(Error, match="overlap"):
        Overlap(4).check(train, ["the quick brown fox jumps over"], 0.1)
    assert (
        Overlap(3).check(["she sells sea shells"], ["the quick brown fox"], 0.1)
        == 0.0
    )
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="threshold"):
            Overlap(2).check(["a b c"], ["a b c"], bad)
