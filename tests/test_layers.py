"""Tests for layer-level diagnostics of attention, activations and norms."""

import math
from typing import Any

import pytest
import torch

from tinylm import model, seeding
from tinylm.layers import Layers


def net(**overrides: Any) -> model.TinyLM:
    values: dict[str, Any] = {
        "vocab_size": 64,
        "max_seq_len": 32,
        "d_model": 32,
        "n_layers": 4,
        "n_heads": 4,
        "window": 8,
        "global_every": 1,
    }
    values.update(overrides)
    seeding.set_seed(0)
    return model.TinyLM(model.Config(**values)).eval()


def tokens(length: int = 24) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(1)
    return torch.randint(0, 64, (4, length), generator=generator)


class TestAttention:
    def test_shape_unit_interval_and_monotone_in_window(self):
        masses = Layers(net()).attention(tokens(), (2, 8, 16))
        assert [m.window for m in masses] == [2, 8, 16]
        for m in masses:
            assert len(m.layers) == 4 and all(0.0 <= v <= 1.0 for v in m.layers)
        for layer in range(4):
            assert (
                masses[0].layers[layer]
                >= masses[1].layers[layer]
                >= masses[2].layers[layer]
            )

    def test_window_covering_the_sequence_has_zero_mass_beyond(self):
        (mass,) = Layers(net()).attention(tokens(24), (24,))
        assert mass.layers == (0.0,) * 4

    def test_windowed_layer_has_zero_mass_beyond_its_window(self):
        (mass,) = Layers(net(global_every=99, n_layers=1)).attention(
            tokens(), (8,)
        )
        assert mass.layers[0] == pytest.approx(0.0, abs=1e-7)

    def test_uniform_attention_gives_the_geometric_expectation(self):
        model_ = net(n_layers=1)
        with torch.no_grad():
            Layers(model_).blocks()[0].attn.qkv.weight.zero_()
        length = 16
        (mass,) = Layers(model_).attention(tokens(length), (4,))
        expected = sum(max(0, i - 3) / (i + 1) for i in range(length)) / length
        assert mass.layers[0] == pytest.approx(expected, abs=1e-5)

    def test_rejects_bad_windows_and_tokens(self):
        with pytest.raises(ValueError, match="window"):
            Layers(net()).attention(tokens(), (0,))
        with pytest.raises(ValueError, match="shape"):
            Layers(net()).attention(torch.zeros(5, dtype=torch.long), (4,))


class TestFeedForward:
    def test_relu_is_sparse_and_gelu_is_not(self):
        relu = Layers(net(activation="relu")).ffn(tokens())
        gelu = Layers(net(activation="gelu")).ffn(tokens())
        assert len(relu) == len(gelu) == 4
        assert all(0.3 < s.inactive < 0.7 for s in relu)
        assert all(s.inactive < 0.1 for s in gelu)

    def test_zero_weights_make_every_unit_dead(self):
        model_ = net(activation="relu")
        with torch.no_grad():
            for block in Layers(model_).blocks():
                block.ffn_in.weight.zero_()
        assert all(
            s.inactive == 1.0 and s.dead == 1.0
            for s in Layers(model_).ffn(tokens())
        )

    def test_negative_mass_zero_for_relu_positive_for_gelu(self):
        assert all(
            s.negative == 0.0
            for s in Layers(net(activation="relu")).ffn(tokens())
        )
        assert all(
            s.negative > 0.0
            for s in Layers(net(activation="gelu")).ffn(tokens())
        )


class TestNorms:
    def test_one_residual_rms_per_block(self):
        stats = Layers(net()).norms(tokens())
        assert len(stats.residual) == 4
        assert all(math.isfinite(v) and v > 0 for v in stats.residual)

    def test_dyt_alpha_saturation_and_absence_for_rms(self):
        stats = Layers(net(norm="dyt")).norms(tokens())
        assert len(stats.alpha) == 9 and all(
            a == pytest.approx(0.5) for a in stats.alpha
        )
        assert all(0.0 <= s <= 1.0 for s in stats.saturation)
        plain = Layers(net(norm="rms")).norms(tokens())
        assert plain.alpha == () and plain.saturation == ()

    def test_gradient_norms_only_when_requested(self):
        assert Layers(net()).norms(tokens()).gradient == ()
        stats = Layers(net()).norms(tokens(), gradients=True)
        assert len(stats.gradient) == 4
        assert all(math.isfinite(v) and v > 0 for v in stats.gradient)

    def test_saturation_when_inputs_are_huge(self):
        model_ = net(norm="dyt")
        with torch.no_grad():
            model_.tok_emb.weight.mul_(10000.0)
        assert Layers(model_).norms(tokens()).saturation[0] > 0.9


def test_parts_count_shared_weights_once():
    parts = Layers(net()).parts()
    assert sum(parts.values()) == model.count(net())
    assert parts["head"] == 0
