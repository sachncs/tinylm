"""Property-based tests: invariants that must hold for any valid input."""

import pathlib
import tempfile

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from tinylm import clean, ledger, packing
from tinylm.bench import Confidence
from tinylm.bootstrap import Bootstrap
from tinylm.ledger import Budget
from tinylm.screen import Point, front
from tinylm.tokens import Token

SETTINGS = settings(derandomize=True, deadline=None, max_examples=60)


@SETTINGS
@given(
    vocab=st.integers(2, 70_000),
    data=st.data(),
    length=st.integers(1, 300),
)
def test_token_store_round_trips_any_valid_ids(
    vocab: int, data: st.DataObject, length: int
):
    ids = data.draw(
        st.lists(st.integers(0, vocab - 1), min_size=length, max_size=length)
    )
    tensor = torch.tensor(ids, dtype=torch.long)
    token = Token.from_tensor(tensor, vocab)
    with tempfile.TemporaryDirectory() as tmp:
        token.store(pathlib.Path(tmp) / "s")
        loaded = Token.load(pathlib.Path(tmp) / "s", verify=True)
        assert loaded.fingerprint() == token.fingerprint()
        starts = torch.arange(len(ids))
        assert torch.equal(loaded.rows(starts, 1).reshape(-1), tensor)


@SETTINGS
@given(
    length=st.integers(0, 5000),
    width=st.integers(1, 300),
    epoch=st.integers(0, 40),
    seed=st.integers(0, 10_000),
    shift=st.booleans(),
)
def test_chunk_starts_are_disjoint_in_range_and_cover_all_but_a_tail(
    length: int, width: int, epoch: int, seed: int, shift: bool
):
    starts = packing.starts(length, width, epoch, seed, shift=shift)
    ordered = starts.sort().values
    if starts.numel():
        assert int(ordered.min()) >= 0
        assert int(ordered.max()) + width <= length
        assert bool((ordered[1:] - ordered[:-1] >= width).all())
    first = packing.offset(epoch, width) if shift else 0
    assert length - first - width * starts.numel() < width or length < first
    assert torch.equal(
        starts, packing.starts(length, width, epoch, seed, shift=shift)
    )


@SETTINGS
@given(st.text(max_size=120))
def test_cleaning_any_text_is_idempotent_and_never_leaves_edge_spaces(
    text: str,
):
    once = list(clean.apply([text], "gutenberg", clean.Options()))
    twice = list(clean.apply(once, "gutenberg", clean.Options()))
    assert once == twice
    assert all(line == line.strip() and line for line in once)


@SETTINGS
@given(
    unique=st.integers(1, 10_000),
    epochs=st.integers(1, 10),
    spends=st.lists(st.integers(1, 5000), max_size=30),
)
def test_ledger_never_lets_consumption_pass_the_cap(
    unique: int, epochs: int, spends: list[int]
):
    budget = Budget(unique, epochs)
    for spend in spends:
        before = budget.consumed
        try:
            budget.consume(spend)
        except ledger.Error:
            assert budget.consumed == before
        assert budget.consumed <= unique * epochs


@SETTINGS
@given(
    margins=st.lists(
        st.floats(-10, 10, allow_nan=False), min_size=1, max_size=200
    ),
    tau=st.floats(0, 12),
)
def test_confidence_counts_are_consistent(margins: list[float], tau: float):
    found = Confidence.of(margins, tau)
    assert 0 <= found.correct <= found.confident <= found.total == len(margins)
    assert found.coverage == found.confident / found.total
    if found.confident:
        assert found.precision == found.correct / found.confident
    else:
        assert found.precision is None


@SETTINGS
@given(
    groups=st.dictionaries(
        st.text("abc", min_size=1, max_size=2),
        st.lists(st.booleans(), min_size=1, max_size=40),
        min_size=1,
        max_size=4,
    )
)
def test_bootstrap_interval_is_ordered_and_inside_zero_one(
    groups: dict[str, list[bool]],
):
    found = Bootstrap(resamples=120).mean(groups)
    assert 0.0 <= found.low <= found.high <= 1.0
    assert 0.0 <= found.estimate <= 1.0


@SETTINGS
@given(
    st.lists(
        st.tuples(
            st.floats(0, 1),
            st.one_of(st.none(), st.floats(0, 1)),
            st.floats(0.1, 50),
        ),
        min_size=1,
        max_size=12,
    )
)
def test_pareto_front_is_never_empty_and_nothing_in_it_is_dominated(
    raw: list[tuple[float, float | None, float]],
):
    points = tuple(Point(f"p{i}", a, p, s) for i, (a, p, s) in enumerate(raw))
    result = front(points)
    assert result.front
    assert len(result.front) + len(result.dominated) == len(points)
    for kept in result.front:
        for other in points:
            if other is kept:
                continue
            better = (
                other.accuracy >= kept.accuracy
                and other.worst() >= kept.worst()
                and other.size <= kept.size
            )
            same = (
                other.accuracy == kept.accuracy
                and other.worst() == kept.worst()
                and other.size == kept.size
            )
            assert not better or (same and other.name > kept.name)
