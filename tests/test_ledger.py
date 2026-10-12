"""Tests for the epoch budget ledger."""

import pytest

from tinylm import ledger


def test_limit_consume_and_epochs():
    budget = ledger.DataBudget(unique_tokens=1000, max_epochs=10)
    assert budget.limit == 10_000
    budget.consume(2500)
    budget.consume(500)
    assert budget.consumed == 3000
    assert budget.epochs_used == pytest.approx(3.0)
    assert budget.remaining == 7000


def test_exactly_the_limit_is_allowed_and_more_raises_without_counting():
    budget = ledger.DataBudget(unique_tokens=10, max_epochs=2)
    budget.consume(15)
    with pytest.raises(ledger.BudgetExceededError, match="epochs"):
        budget.consume(6)
    assert budget.consumed == 15
    budget.consume(5)
    assert budget.remaining == 0


def test_state_roundtrip():
    budget = ledger.DataBudget(unique_tokens=100)
    budget.consume(123)
    restored = ledger.DataBudget.from_state(budget.state())
    assert (restored.consumed, restored.unique_tokens, restored.max_epochs) == (
        123,
        100,
        budget.max_epochs,
    )


@pytest.mark.parametrize("unique", [0, -5])
def test_rejects_non_positive_unique_tokens(unique: int):
    with pytest.raises(ValueError, match="unique_tokens"):
        ledger.DataBudget(unique_tokens=unique)


@pytest.mark.parametrize("epochs", [0, -1, 11])
def test_rejects_epoch_cap_outside_one_to_ten(epochs: int):
    with pytest.raises(ValueError, match="max_epochs"):
        ledger.DataBudget(unique_tokens=10, max_epochs=epochs)


@pytest.mark.parametrize("count", [0, -1])
def test_rejects_non_positive_consumption(count: int):
    with pytest.raises(ValueError, match="count"):
        ledger.DataBudget(unique_tokens=10).consume(count)


def test_from_state_rejects_overdrawn_or_incomplete_state():
    with pytest.raises(ValueError, match="consumed"):
        ledger.DataBudget.from_state(
            {"unique_tokens": 10, "max_epochs": 1, "consumed": 11}
        )
    with pytest.raises(ValueError, match="state"):
        ledger.DataBudget.from_state({"unique_tokens": 10})
