"""Tests for the decoder-only language model and its options."""

from typing import Any

import pytest
import torch

from tinylm import model, seeding

PARAM_BUDGET = 50_000_000


def make_config(**overrides: Any) -> model.Config:
    values: dict[str, Any] = {
        "vocab_size": 64,
        "max_seq_len": 16,
        "d_model": 32,
        "n_layers": 4,
        "n_heads": 4,
        "window": 4,
        "global_every": 2,
    }
    values.update(overrides)
    return model.Config(**values)


def make_net(**overrides: Any) -> model.TinyLM:
    seeding.set_seed(0)
    return model.TinyLM(make_config(**overrides)).eval()


def batch(length: int = 12) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(5)
    return torch.randint(0, 64, (2, length), generator=generator)


class TestConfig:
    @pytest.mark.parametrize(
        "field",
        [
            "vocab_size",
            "max_seq_len",
            "d_model",
            "n_layers",
            "n_heads",
            "window",
            "global_every",
        ],
    )
    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_non_positive_fields(self, field: str, bad: int):
        with pytest.raises(ValueError, match=field):
            make_config(**{field: bad})

    def test_rejects_d_model_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            make_config(d_model=30, n_heads=4)

    @pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
    def test_rejects_bad_dropout(self, rate: float):
        with pytest.raises(ValueError, match="dropout"):
            make_config(dropout=rate)

    @pytest.mark.parametrize(
        "field,value", [("activation", "swish"), ("norm", "batch")]
    )
    def test_rejects_unknown_choices(self, field: str, value: str):
        with pytest.raises(ValueError, match=field):
            make_config(**{field: value})

    @pytest.mark.parametrize("sizes", [(0,), (-3, 4)])
    def test_rejects_bad_codebook_sizes(self, sizes: tuple[int, ...]):
        with pytest.raises(ValueError, match="codebook_sizes"):
            make_config(codebook_sizes=sizes)

    def test_codebooks_need_more_layers_than_books(self):
        with pytest.raises(ValueError, match="n_layers"):
            make_config(n_layers=3, codebook_sizes=(8, 4, 2))

    def test_list_codebooks_become_a_tuple(self):
        assert make_config(codebook_sizes=[8, 4]).codebook_sizes == (8, 4)

    def test_defaults(self):
        config = model.Config()
        assert (config.activation, config.norm, config.dropout) == (
            "gelu",
            "rms",
            0.0,
        )
        assert config.tie_embeddings and config.codebook_sizes == ()


class TestMaskAndSchedule:
    def mask(self, length: int, window: int, causal: bool, glob: bool):
        return model.visibility(length, window, causal, glob)

    def test_causal_local_window(self):
        mask = self.mask(10, 3, True, False)
        assert [int(i) for i in torch.nonzero(mask[5]).flatten()] == [3, 4, 5]

    def test_bidirectional_local_window_is_symmetric(self):
        mask = self.mask(10, 3, False, False)
        assert [int(i) for i in torch.nonzero(mask[5]).flatten()] == [
            3,
            4,
            5,
            6,
            7,
        ]
        assert torch.equal(mask, mask.T)

    def test_global_masks(self):
        assert torch.equal(
            self.mask(6, 2, True, True),
            torch.tril(torch.ones(6, 6, dtype=torch.bool)),
        )
        assert bool(self.mask(6, 2, False, True).all())

    @pytest.mark.parametrize("causal", [True, False])
    def test_window_at_least_length_equals_global(self, causal: bool):
        assert torch.equal(
            self.mask(8, 8, causal, False), self.mask(8, 8, causal, True)
        )

    @pytest.mark.parametrize("length,window", [(0, 4), (5, 0), (-1, 4)])
    def test_rejects_non_positive(self, length: int, window: int):
        with pytest.raises(ValueError, match="positive"):
            self.mask(length, window, True, False)

    def test_every_fourth_layer_is_global(self):
        flags = [model.is_global(i, 4) for i in range(8)]
        assert flags == [False, False, False, True] * 2
        with pytest.raises(ValueError, match="positive"):
            model.is_global(0, 0)


class TestForward:
    def test_logits_shape(self):
        assert make_net()(batch()).shape == (2, 12, 64)

    def test_is_causal(self):
        net = make_net()
        a = batch()
        b = a.clone()
        b[:, 8:] = (b[:, 8:] + 1) % 64
        with torch.no_grad():
            out_a, out_b = net(a), net(b)
        torch.testing.assert_close(out_a[:, :8], out_b[:, :8])
        assert not torch.allclose(out_a[:, 8:], out_b[:, 8:])

    def test_bidirectional_differs_from_causal(self):
        net = make_net(global_every=1)
        with torch.no_grad():
            assert not torch.allclose(net(batch()), net(batch(), causal=False))

    def test_same_seed_same_weights(self):
        for p, q in zip(
            make_net().parameters(), make_net().parameters(), strict=True
        ):
            assert torch.equal(p, q)

    def test_rejects_bad_inputs(self):
        net = make_net()
        with pytest.raises(ValueError, match="max_seq_len"):
            net(torch.zeros((1, 17), dtype=torch.long))
        with pytest.raises(ValueError, match="vocab"):
            net(torch.full((1, 4), 64, dtype=torch.long))
        with pytest.raises(ValueError, match="shape"):
            net(torch.zeros((4,), dtype=torch.long))

    def test_local_layer_ignores_tokens_beyond_window(self):
        net = make_net(n_layers=1, global_every=4)
        a = torch.randint(0, 64, (1, 16))
        b = a.clone()
        b[0, 0] = (b[0, 0] + 1) % 64
        with torch.no_grad():
            torch.testing.assert_close(net(a)[:, 4:], net(b)[:, 4:])

    def test_global_layer_sees_whole_prefix(self):
        net = make_net(n_layers=1, global_every=1)
        a = torch.randint(0, 64, (1, 16))
        b = a.clone()
        b[0, 0] = (b[0, 0] + 1) % 64
        with torch.no_grad():
            assert not torch.allclose(net(a)[:, -1], net(b)[:, -1])

    def test_local_window_override(self):
        net = make_net(global_every=1)
        with torch.no_grad():
            torch.testing.assert_close(
                net(batch(), local_window=12), net(batch())
            )
            assert not torch.allclose(
                net(batch(), local_window=2), net(batch())
            )
        with pytest.raises(ValueError, match="local_window"):
            net(batch(), local_window=0)


class TestParametersAndInit:
    def test_default_config_fits_budget(self):
        assert model.count(model.TinyLM(model.Config())) < PARAM_BUDGET

    def test_tied_weights_counted_once_and_untied_adds_a_matrix(self):
        tied = make_net()
        untied = make_net(tie_embeddings=False)
        assert tied.head.weight is tied.tok_emb.weight
        assert untied.head.weight is not untied.tok_emb.weight
        assert (model.count(untied) - model.count(tied)) == 64 * 32

    def test_budget_guard(self):
        net = make_net()
        total = model.count(net)
        model.check(net, total)
        with pytest.raises(ValueError, match="budget"):
            model.check(net, total - 1)

    def test_init_is_small_and_residual_scaled(self):
        seeding.set_seed(0)
        net = model.TinyLM(model.Config())
        assert float(net.tok_emb.weight.detach().std()) == pytest.approx(
            0.02, rel=0.1
        )
        block = net.blocks[0]
        assert isinstance(block, model.Block)
        expected = 0.02 / (2 * model.Config().n_layers) ** 0.5
        assert float(block.attn.proj.weight.detach().std()) == pytest.approx(
            expected, rel=0.1
        )

    def test_untrained_loss_is_near_uniform(self):
        net = make_net()
        tokens = batch(16)
        targets = torch.randint(0, 64, tokens.shape)
        with torch.no_grad():
            loss = torch.nn.functional.cross_entropy(
                net(tokens).reshape(-1, 64), targets.reshape(-1)
            )
        assert float(loss) == pytest.approx(
            float(torch.tensor(64.0).log()), abs=0.3
        )


class TestDropout:
    def test_adds_no_parameters(self):
        assert model.count(make_net()) == model.count(make_net(dropout=0.3))

    def test_eval_exact_and_train_stochastic_but_seeded(self):
        net = make_net(dropout=0.5)
        with torch.no_grad():
            assert torch.equal(net(batch()), net(batch()))
        net.train()
        seeding.set_seed(1)
        a = net(batch())
        b = net(batch())
        assert not torch.equal(a, b)
        seeding.set_seed(1)
        assert torch.equal(a, net(batch()))


class TestActivationAndNorm:
    def test_ffn_activation_module(self):
        for name, kind in (("gelu", torch.nn.GELU), ("relu", torch.nn.ReLU)):
            block = make_net(activation=name).blocks[0]
            assert isinstance(block, model.Block)
            assert isinstance(block.act, kind)

    def test_dyt_replaces_every_norm(self):
        net = make_net(norm="dyt")
        assert not any(isinstance(m, torch.nn.RMSNorm) for m in net.modules())
        assert (
            sum(isinstance(m, model.DynamicTanh) for m in net.modules())
            == 2 * 4 + 1
        )

    def test_dyt_is_bounded_odd_and_has_learnable_alpha(self):
        norm = model.DynamicTanh(8)
        x = torch.randn(4, 8) * 100
        assert float(norm(x).detach().abs().max()) <= 1.0 + 1e-6
        torch.testing.assert_close(norm(-x), -norm(x))
        assert norm.alpha.requires_grad
        assert sum(p.numel() for p in norm.parameters()) == 8 + 8 + 1

    def test_gradients_finite_with_dyt_and_relu(self):
        net = make_net(norm="dyt", activation="relu")
        net(batch()).logsumexp(-1).mean().backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)


class TestCodebooks:
    def cfg(self) -> dict[str, Any]:
        return {"codebook_sizes": (48, 24, 12)}

    def test_one_codebook_between_each_pair_of_stages(self):
        net = make_net(**self.cfg())
        books = [m for m in net.modules() if isinstance(m, model.Codebook)]
        assert [b.codes.shape for b in books] == [(48, 32), (24, 32), (12, 32)]

    def test_parameters_are_counted(self):
        base = model.count(make_net())
        full = model.count(make_net(**self.cfg()))
        assert full - base == (48 + 24 + 12) * 32 + 3 * 32

    def test_still_causal_and_full_vocab(self):
        net = make_net(**self.cfg())
        a = batch()
        b = a.clone()
        b[:, 8:] = (b[:, 8:] + 1) % 64
        with torch.no_grad():
            out_a = net(a)
            torch.testing.assert_close(out_a[:, :8], net(b)[:, :8])
        assert out_a.shape == (2, 12, 64)

    def test_output_is_a_mixture_of_codes_and_grads_flow(self):
        book = model.Codebook(8, 16, model.Config(d_model=16))
        x = torch.randn(3, 5, 16)
        delta = (book(x) - x).detach()
        assert float(delta.min()) >= book.codes.min().item() - 1e-5
        assert float(delta.max()) <= book.codes.max().item() + 1e-5
        net = make_net(**self.cfg())
        net(batch()).logsumexp(-1).mean().backward()
        for module in net.modules():
            if isinstance(module, model.Codebook):
                assert module.codes.grad is not None


class TestMaskCacheAndFastPath:
    def test_local_mask_is_cached_per_arguments_and_matches_visibility(self):
        model.local_mask.cache_clear()
        a = model.local_mask(10, 3, True, "cpu")
        assert model.local_mask(10, 3, True, "cpu") is a
        assert model.local_mask(10, 4, True, "cpu") is not a
        assert torch.equal(a, model.visibility(10, 3, True, is_global=False))
        info = model.local_mask.cache_info()
        assert info.hits == 1 and info.misses == 2

    def test_repeated_forwards_build_no_new_masks(self):
        net = make_net()
        model.local_mask.cache_clear()
        net(batch())
        misses = model.local_mask.cache_info().misses
        net(batch())
        net(batch())
        assert model.local_mask.cache_info().misses == misses

    def test_global_layers_skip_the_explicit_mask(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        net = make_net(global_every=1)
        calls: list[object] = []
        original = model.Attention.forward

        def spy(
            self: model.Attention,
            x: torch.Tensor,
            mask: torch.Tensor | None,
            causal: bool,
        ) -> torch.Tensor:
            calls.append(mask)
            return original(self, x, mask, causal)

        monkeypatch.setattr(model.Attention, "forward", spy)
        net(batch())
        assert calls and all(m is None for m in calls)

    @pytest.mark.parametrize("causal", [True, False])
    def test_fused_path_equals_the_explicit_mask_path(self, causal: bool):
        net = make_net(global_every=1)
        with torch.no_grad():
            fast = net(batch(), causal=causal)
            slow = net(batch(), causal=causal, local_window=12)
        torch.testing.assert_close(fast, slow, rtol=1e-5, atol=1e-5)


def attentions(net: model.TinyLM) -> list[Any]:
    """Attention modules of net, untyped so tests can reach the slope."""
    return [block.attn for block in net.blocks]


class TestRecency:
    """Per-head distance penalty on attention scores (config.recency)."""

    def test_zero_penalty_equals_plain_full_attention_exactly(self):
        plain = make_net(window=16, global_every=1)
        recent = make_net(window=16, global_every=1, recency=True)
        recent.load_state_dict(plain.state_dict(), strict=False)
        torch.testing.assert_close(
            recent(batch()), plain(batch()), rtol=1e-5, atol=1e-6
        )

    def test_adds_one_slope_per_head_per_layer_starting_at_zero(self):
        plain, recent = make_net(), make_net(recency=True)
        assert model.count(recent) - model.count(plain) == 4 * 4
        for attn in attentions(recent):
            assert torch.equal(attn.slope, torch.zeros(4))

    def test_a_positive_slope_changes_the_output_and_stays_causal(self):
        net = make_net(window=16, global_every=1, recency=True)
        before = net(batch())
        for attn in attentions(net):
            attn.slope.data.fill_(0.5)
        after = net(batch())
        assert not torch.allclose(before, after)
        changed = batch().clone()
        changed[:, -1] = (changed[:, -1] + 1) % 64
        torch.testing.assert_close(net(changed)[:, :-1], after[:, :-1])

    def test_a_negative_slope_is_treated_as_no_penalty(self):
        net = make_net(window=16, global_every=1, recency=True)
        zero = net(batch())
        for attn in attentions(net):
            attn.slope.data.fill_(-3.0)
        torch.testing.assert_close(net(batch()), zero)

    def test_the_slope_receives_a_gradient_at_zero_so_it_can_be_learned(self):
        net = make_net(recency=True).train()
        logits = net(batch())
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 64), batch()[:, 1:].reshape(-1)
        )
        loss.backward()
        grads = [a.slope.grad for a in attentions(net)]
        assert all(g is not None and bool(g.abs().sum() > 0) for g in grads)

    def test_penalty_is_none_when_off_and_grows_with_distance_when_on(self):
        off = attentions(make_net())[0]
        assert off.penalty(5, torch.device("cpu")) is None
        on = attentions(make_net(recency=True))[0]
        on.slope.data.fill_(0.25)
        penalty = on.penalty(5, torch.device("cpu"))
        assert penalty is not None
        assert penalty.shape == (4, 5, 5)
        assert float(penalty[0, 3, 3].detach()) == 0
        assert float(penalty[0, 3, 0].detach()) == pytest.approx(0.75)

    def test_works_bidirectionally_and_with_a_local_window(self):
        net = make_net(recency=True)
        for attn in attentions(net):
            attn.slope.data.fill_(0.3)
        assert net(batch(), causal=False).shape == (2, 12, 64)
        assert net(batch(), local_window=3).shape == (2, 12, 64)

    def test_invalid_value_is_rejected(self):
        with pytest.raises(ValueError, match="recency"):
            make_config(recency="yes")

    def test_compiled_forward_matches_eager(self):
        net = make_net(recency=True)
        for attn in attentions(net):
            attn.slope.data.fill_(0.2)
        torch.testing.assert_close(
            torch.compile(net)(batch()), net(batch()), rtol=1e-4, atol=1e-5
        )


class TestExclusive:
    """Exclusive self attention: output orthogonal to the token's own value."""

    def test_exclude_removes_exactly_the_component_along_v(self):
        generator = torch.Generator().manual_seed(1)
        out = torch.randn(2, 3, 5, 8, generator=generator)
        v = torch.randn(2, 3, 5, 8, generator=generator)
        result = model.exclude(out, v)
        assert float((result * v).sum(-1).abs().max()) < 1e-5
        torch.testing.assert_close(model.exclude(result, v), result)

    def test_exclude_leaves_a_zero_value_vector_alone(self):
        out = torch.ones(1, 1, 2, 4)
        assert torch.equal(model.exclude(out, torch.zeros(1, 1, 2, 4)), out)

    def test_a_single_token_attends_only_to_itself_so_nothing_remains(self):
        net = make_net(exclusive=(0, 1, 2, 3))
        attn = attentions(net)[0]
        hidden = torch.randn(
            2, 1, 32, generator=torch.Generator().manual_seed(2)
        )
        assert float(attn(hidden, None, True).detach().abs().max()) < 1e-6
        plain = attentions(make_net())[0]
        assert float(plain(hidden, None, True).detach().abs().max()) > 1e-4

    def test_changes_the_output_adds_no_parameters_and_stays_causal(self):
        plain, xsa = make_net(), make_net(exclusive=(1, 2))
        assert model.count(plain) == model.count(xsa)
        xsa.load_state_dict(plain.state_dict())
        assert not torch.allclose(plain(batch()), xsa(batch()))
        changed = batch().clone()
        changed[:, -1] = (changed[:, -1] + 1) % 64
        torch.testing.assert_close(xsa(changed)[:, :-1], xsa(batch())[:, :-1])

    def test_only_the_listed_layers_are_exclusive(self):
        net = make_net(exclusive=(1, 2))
        assert [a.exclusive for a in attentions(net)] == [
            False,
            True,
            True,
            False,
        ]

    def test_combines_with_local_windows_recency_and_bidirectional_use(self):
        net = make_net(exclusive=(1, 2), recency=True)
        assert net(batch(), causal=False).shape == (2, 12, 64)
        assert net(batch(), local_window=3).shape == (2, 12, 64)

    @pytest.mark.parametrize("bad", [(4,), (-1,), (1, 1), (0.5,), ("a",)])
    def test_invalid_layer_lists_are_rejected(self, bad: tuple[Any, ...]):
        with pytest.raises(ValueError, match="exclusive"):
            make_config(exclusive=bad)

    def test_list_input_is_normalised_to_a_tuple(self):
        assert make_config(exclusive=[2, 1]).exclusive == (1, 2)

    def test_compiled_forward_matches_eager(self):
        net = make_net(exclusive=(1, 2))
        torch.testing.assert_close(
            torch.compile(net)(batch()), net(batch()), rtol=1e-4, atol=1e-5
        )


def crosses(net: model.TinyLM) -> list[Any]:
    """Cross-attention modules of net (None where a layer has none)."""
    return [block.cross for block in net.blocks]


class TestCross:
    """Cross attention from token states to learned memory slots."""

    def test_adds_exactly_the_documented_parameters_to_listed_layers(self):
        plain, cross = make_net(), make_net(cross=(1, 2), slots=8)
        width, slots = 32, 8
        per_layer = 2 * width * width + 2 * slots * width + width
        assert model.count(cross) - model.count(plain) == 2 * per_layer

    def test_only_the_listed_layers_have_it(self):
        net = make_net(cross=(0, 3))
        assert [c is not None for c in crosses(net)] == [
            True,
            False,
            False,
            True,
        ]

    def test_with_the_output_projection_zeroed_it_is_the_plain_model(self):
        plain = make_net()
        net = make_net(cross=(1, 2))
        net.load_state_dict(plain.state_dict(), strict=False)
        for cross in crosses(net):
            if cross is not None:
                cross.proj.weight.data.zero_()
        torch.testing.assert_close(net(batch()), plain(batch()))

    def test_it_changes_the_output_and_stays_causal(self):
        plain, net = make_net(), make_net(cross=(1, 2))
        net.load_state_dict(plain.state_dict(), strict=False)
        assert not torch.allclose(net(batch()), plain(batch()))
        changed = batch().clone()
        changed[:, -1] = (changed[:, -1] + 1) % 64
        torch.testing.assert_close(net(changed)[:, :-1], net(batch())[:, :-1])

    def test_the_memory_slots_receive_gradients_and_are_used(self):
        net = make_net(cross=(1,), slots=4).train()
        cross = crosses(net)[1]
        cross.proj.weight.data.normal_(std=0.1)
        logits = net(batch())
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 64), batch()[:, 1:].reshape(-1)
        )
        loss.backward()
        assert float(cross.keys.grad.abs().sum()) > 0
        assert float(cross.values.grad.abs().sum()) > 0

    def test_editing_the_slots_changes_the_output(self):
        net = make_net(cross=(1,), slots=4)
        cross = crosses(net)[1]
        cross.proj.weight.data.normal_(std=0.1)
        before = net(batch()).detach()
        cross.values.data.add_(1.0)
        assert not torch.allclose(net(batch()).detach(), before)

    def test_combines_with_exclusive_recency_windows_and_bidirectional_use(
        self,
    ):
        net = make_net(cross=(1, 2), exclusive=(1, 2), recency=True)
        assert net(batch(), causal=False).shape == (2, 12, 64)
        assert net(batch(), local_window=3).shape == (2, 12, 64)

    @pytest.mark.parametrize("bad", [(4,), (-1,), (1, 1), (0.5,)])
    def test_invalid_layer_lists_are_rejected(self, bad: tuple[Any, ...]):
        with pytest.raises(ValueError, match="cross"):
            make_config(cross=bad)

    @pytest.mark.parametrize("slots", [0, -2])
    def test_slots_must_be_positive(self, slots: int):
        with pytest.raises(ValueError, match="slots"):
            make_config(slots=slots)

    def test_compiled_forward_matches_eager(self):
        net = make_net(cross=(1, 2))
        torch.testing.assert_close(
            torch.compile(net)(batch()), net(batch()), rtol=1e-4, atol=1e-5
        )
