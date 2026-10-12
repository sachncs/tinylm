"""Tests for optimizer construction and parameter partitioning."""

import copy

import pytest
import torch

from tinylm import model, optim, seeding


def net() -> model.TinyLM:
    seeding.set_seed(0)
    return model.TinyLM(
        model.ModelConfig(
            vocab_size=64,
            max_seq_len=16,
            d_model=16,
            n_layers=2,
            n_heads=2,
            window=4,
            global_every=2,
        )
    )


def step(
    model_: model.TinyLM, opt: optim.Optimizer, tokens: torch.Tensor
) -> None:
    opt.zero_grad()
    torch.nn.functional.cross_entropy(
        model_(tokens[:, :-1]).reshape(-1, 64), tokens[:, 1:].reshape(-1)
    ).backward()
    opt.step()


def test_partition_covers_every_parameter_once_with_hidden_matrices_only():
    model_ = net()
    matrices, others = optim.partition(model_)
    ids = [id(p) for p in matrices + others]
    assert len(ids) == len(set(ids)) and set(ids) == {
        id(p) for p in model_.parameters()
    }
    assert matrices and all(p.ndim == 2 for p in matrices)
    other_ids = {id(p) for p in others}
    assert {
        id(model_.tok_emb.weight),
        id(model_.pos_emb.weight),
        id(model_.final_norm.weight),
    } <= other_ids


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_step_changes_parameters_and_lr_is_settable(name: str):
    model_ = net()
    opt = optim.build(model_, name, lr=0.01, weight_decay=0.0)
    before = [p.detach().clone() for p in model_.parameters()]
    for group in opt.param_groups:
        group["lr"] = 0.02
    step(model_, opt, torch.randint(0, 64, (4, 9)))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(before, model_.parameters(), strict=True)
    )
    assert all(g["lr"] == 0.02 for g in opt.param_groups)


def test_state_dict_round_trip_continues_identically():
    a, b = net(), net()
    opt_a = optim.build(a, "muon", lr=0.01, weight_decay=0.1)
    tokens = torch.randint(0, 64, (4, 9))
    step(a, opt_a, tokens)
    b.load_state_dict(a.state_dict())
    opt_b = optim.build(b, "muon", lr=0.01, weight_decay=0.1)
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))
    step(a, opt_a, tokens)
    step(b, opt_b, tokens)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="optimizer"):
        optim.build(net(), "sgd", lr=0.01, weight_decay=0.0)
    with pytest.raises(ValueError, match="lr"):
        optim.build(net(), "adamw", lr=0.0, weight_decay=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        optim.build(net(), "adamw", lr=0.1, weight_decay=-0.1)
