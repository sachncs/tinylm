"""Tests for the decoder-only language model and its options."""

import pytest
import torch

from tinylm import model, seeding

PARAM_BUDGET = 50_000_000


def make_config(**overrides: object) -> model.ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 64,
        "max_seq_len": 16,
        "d_model": 32,
        "n_layers": 4,
        "n_heads": 4,
        "window": 4,
        "global_every": 2,
    }
    values.update(overrides)
    return model.ModelConfig(**values)  # pyright: ignore[reportArgumentType]


def make_net(**overrides: object) -> model.TinyLM:
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
        config = model.ModelConfig()
        assert (config.activation, config.norm, config.dropout) == (
            "gelu",
            "rms",
            0.0,
        )
        assert config.tie_embeddings and config.codebook_sizes == ()


class TestMaskAndSchedule:
    def mask(self, length: int, window: int, causal: bool, glob: bool):
        return model.build_attention_mask(length, window, causal, glob)

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
        flags = [model.is_global_layer(i, 4) for i in range(8)]
        assert flags == [False, False, False, True] * 2
        with pytest.raises(ValueError, match="positive"):
            model.is_global_layer(0, 0)


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
        assert (
            model.count_parameters(model.TinyLM(model.ModelConfig()))
            < PARAM_BUDGET
        )

    def test_tied_weights_counted_once_and_untied_adds_a_matrix(self):
        tied = make_net()
        untied = make_net(tie_embeddings=False)
        assert tied.head.weight is tied.tok_emb.weight
        assert untied.head.weight is not untied.tok_emb.weight
        assert (
            model.count_parameters(untied) - model.count_parameters(tied)
        ) == 64 * 32

    def test_budget_guard(self):
        net = make_net()
        total = model.count_parameters(net)
        model.check_parameter_budget(net, total)
        with pytest.raises(ValueError, match="budget"):
            model.check_parameter_budget(net, total - 1)

    def test_init_is_small_and_residual_scaled(self):
        seeding.set_seed(0)
        net = model.TinyLM(model.ModelConfig())
        assert float(net.tok_emb.weight.detach().std()) == pytest.approx(
            0.02, rel=0.1
        )
        block = net.blocks[0]
        assert isinstance(block, model.Block)
        expected = 0.02 / (2 * model.ModelConfig().n_layers) ** 0.5
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
        assert model.count_parameters(make_net()) == model.count_parameters(
            make_net(dropout=0.3)
        )

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
    def cfg(self) -> dict[str, object]:
        return {"codebook_sizes": (48, 24, 12)}

    def test_one_codebook_between_each_pair_of_stages(self):
        net = make_net(**self.cfg())
        books = [m for m in net.modules() if isinstance(m, model.Codebook)]
        assert [b.codes.shape for b in books] == [(48, 32), (24, 32), (12, 32)]

    def test_parameters_are_counted(self):
        base = model.count_parameters(make_net())
        full = model.count_parameters(make_net(**self.cfg()))
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
        book = model.Codebook(8, 16, model.ModelConfig(d_model=16))
        x = torch.randn(3, 5, 16)
        delta = book(x) - x
        assert float(delta.min()) >= book.codes.min().item() - 1e-5
        assert float(delta.max()) <= book.codes.max().item() + 1e-5
        net = make_net(**self.cfg())
        net(batch()).logsumexp(-1).mean().backward()
        for module in net.modules():
            if isinstance(module, model.Codebook):
                assert module.codes.grad is not None
