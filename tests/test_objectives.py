"""Tests for masked-token corruption and the hybrid objective."""

import pytest
import torch

from tinylm import model, objectives, seeding

MASK_ID = 4
VOCAB = 64


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def tokens(batch: int = 64, length: int = 128) -> torch.Tensor:
    return torch.randint(10, VOCAB, (batch, length), generator=gen(99))


def test_mask_tokens_rate_first_position_and_unselected_unchanged():
    original = tokens()
    corrupted, selected = objectives.corrupt(
        original, 0.15, MASK_ID, VOCAB, gen()
    )
    assert float(selected.float().mean()) == pytest.approx(0.15, abs=0.01)
    assert not bool(selected[:, 0].any())
    assert torch.equal(corrupted[~selected], original[~selected])
    assert int(corrupted.min()) >= 0 and int(corrupted.max()) < VOCAB


def test_replacement_split_is_80_10_10():
    original = tokens(256, 128)
    corrupted, selected = objectives.corrupt(
        original, 0.3, MASK_ID, VOCAB, gen()
    )
    picked, orig = corrupted[selected], original[selected]
    assert float((picked == MASK_ID).float().mean()) == pytest.approx(
        0.8, abs=0.02
    )
    kept = (picked == orig) & (picked != MASK_ID)
    assert float(kept.float().mean()) == pytest.approx(0.1, abs=0.02)


def test_mask_tokens_deterministic_and_validates():
    a = objectives.corrupt(tokens(), 0.15, MASK_ID, VOCAB, gen(3))
    b = objectives.corrupt(tokens(), 0.15, MASK_ID, VOCAB, gen(3))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for rate in (0.0, -0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="mask_rate"):
            objectives.corrupt(tokens(), rate, MASK_ID, VOCAB, gen())
    with pytest.raises(ValueError, match="shape"):
        objectives.corrupt(
            torch.zeros(5, dtype=torch.long), 0.15, MASK_ID, VOCAB, gen()
        )


def net() -> model.TinyLM:
    seeding.set_seed(0)
    return model.TinyLM(
        model.Config(
            vocab_size=VOCAB,
            max_seq_len=32,
            d_model=16,
            n_layers=2,
            n_heads=2,
            window=4,
            global_every=2,
        )
    )


def spec(**kw: float) -> objectives.Hybrid:
    values: dict[str, float | int] = {
        "masked_fraction": 0.5,
        "mask_rate": 0.25,
        "mask_id": MASK_ID,
    }
    values.update(kw)
    return objectives.Hybrid(**values)  # pyright: ignore[reportArgumentType]


def test_pure_causal_matches_plain_next_token_loss():
    model_ = net()
    batch = torch.randint(10, VOCAB, (8, 17), generator=gen(5))
    loss = objectives.loss(model_, batch, spec(masked_fraction=0.0), gen())
    expected = torch.nn.functional.cross_entropy(
        model_(batch[:, :-1]).reshape(-1, VOCAB), batch[:, 1:].reshape(-1)
    )
    torch.testing.assert_close(loss, expected)


def test_hybrid_is_finite_deterministic_and_backpropagates():
    model_ = net()
    batch = torch.randint(10, VOCAB, (8, 17), generator=gen(5))
    a = objectives.loss(model_, batch, spec(), gen(2))
    b = objectives.loss(model_, batch, spec(), gen(2))
    assert float(a.detach()) == float(b.detach())
    a.backward()
    grads = [p.grad for p in model_.parameters() if p.grad is not None]
    assert (
        torch.isfinite(a)
        and grads
        and all(torch.isfinite(g).all() for g in grads)
    )


def test_validation():
    for fraction in (-0.1, 1.1):
        with pytest.raises(ValueError, match="masked_fraction"):
            spec(masked_fraction=fraction)
    with pytest.raises(ValueError, match="batch"):
        objectives.loss(
            net(),
            torch.randint(10, VOCAB, (1, 17), generator=gen(5)),
            spec(),
            gen(),
        )
