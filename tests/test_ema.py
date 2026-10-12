"""Tests for exponential moving average of weights."""

import pytest
import torch

from tinylm import ema, model, seeding


def net() -> model.TinyLM:
    seeding.set_seed(0)
    return model.TinyLM(
        model.ModelConfig(
            vocab_size=32,
            max_seq_len=16,
            d_model=16,
            n_layers=2,
            n_heads=2,
            window=4,
            global_every=2,
        )
    )


def perturb(model_: model.TinyLM, amount: float) -> None:
    with torch.no_grad():
        for p in model_.parameters():
            p.add_(amount)


def test_initial_update_decay_one_and_restore():
    model_ = net()
    average = ema.Ema(model_, decay=0.9)
    start = average.shadow["final_norm.weight"].clone()
    torch.testing.assert_close(start, model_.state_dict()["final_norm.weight"])
    perturb(model_, 1.0)
    average.update(model_)
    expected = 0.9 * start + 0.1 * model_.state_dict()["final_norm.weight"]
    torch.testing.assert_close(average.shadow["final_norm.weight"], expected)
    live = {k: v.clone() for k, v in model_.state_dict().items()}
    with average.applied(model_):
        torch.testing.assert_close(
            model_.state_dict()["final_norm.weight"],
            average.shadow["final_norm.weight"],
        )
    for key, value in model_.state_dict().items():
        torch.testing.assert_close(value, live[key])
    frozen = ema.Ema(net(), decay=1.0)
    before = {k: v.clone() for k, v in frozen.shadow.items()}
    perturb(model_, 1.0)
    frozen.update(model_)
    for key, value in frozen.shadow.items():
        torch.testing.assert_close(value, before[key])


def test_restore_happens_even_if_body_raises():
    model_ = net()
    average = ema.Ema(model_, decay=0.5)
    perturb(model_, 1.0)
    average.update(model_)
    live = model_.state_dict()["final_norm.weight"].clone()
    with pytest.raises(RuntimeError), average.applied(model_):
        raise RuntimeError("boom")
    torch.testing.assert_close(model_.state_dict()["final_norm.weight"], live)


def test_state_round_trip_and_validation():
    model_ = net()
    average = ema.Ema(model_, decay=0.9)
    perturb(model_, 1.0)
    average.update(model_)
    restored = ema.Ema(net(), decay=0.9)
    restored.load_state_dict(average.state_dict())
    for key, value in average.shadow.items():
        torch.testing.assert_close(restored.shadow[key], value)
    with pytest.raises(ValueError, match="keys"):
        restored.load_state_dict({"bogus": torch.zeros(1)})
    for decay in (-0.1, 1.1):
        with pytest.raises(ValueError, match="decay"):
            ema.Ema(net(), decay=decay)
