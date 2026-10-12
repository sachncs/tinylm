"""Tests for recall and superposition probes on planted cases."""

from typing import Any

import pytest
import torch

from tinylm import model, seeding, tokenizer
from tinylm.probes import Associative, Memory, Sae, Superposition


def corpus() -> list[str]:
    return [f"item {i} says the cat sat on the mat today ." for i in range(60)]


def tiny(tok_vocab: int, **kw: object) -> model.TinyLM:
    seeding.set_seed(0)
    values: dict[str, Any] = {
        "vocab_size": tok_vocab,
        "max_seq_len": 64,
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 2,
        "window": 8,
        "global_every": 1,
    }
    values.update(kw)
    return model.TinyLM(model.Config(**values)).eval()


class TestMemory:
    def probe(self) -> Memory:
        tok = tokenizer.train(corpus(), vocab_size=300, kind="bpe")
        return Memory(tiny(tok.get_vocab_size()), tok, "cpu")

    def test_recall_fractions_are_probabilities_and_counts_match(self):
        found = self.probe().recall(corpus()[:20])
        assert 0.0 <= found.exact <= found.token <= 1.0
        assert found.lines == 20

    def test_a_trained_model_recalls_what_it_memorised(self):
        probe = self.probe()
        tok, net = probe.tok, probe.net
        ids = [
            torch.tensor([2, *tok.encode(line).ids]) for line in corpus()[:12]
        ]
        opt = torch.optim.AdamW(net.parameters(), lr=5e-3)
        net.train()
        for _ in range(600):
            for row in ids:
                loss = torch.nn.functional.cross_entropy(
                    net(row[None, :-1])[0], row[1:]
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        seen = probe.recall(corpus()[:12])
        unseen = probe.recall(
            [f"zebra quartz {i} jumped over vexing fog ." for i in range(12)]
        )
        assert seen.token > 0.9 and seen.exact > 0.5
        assert seen.token - unseen.token > 0.3

    def test_rejects_empty_and_too_short_lines(self):
        with pytest.raises(ValueError, match="lines"):
            self.probe().recall([])
        with pytest.raises(ValueError, match="short"):
            self.probe().recall(["a"])


class TestAssociative:
    def test_accuracy_by_distance_is_reported_and_bounded(self):
        net = tiny(300)
        found = Associative(net, "cpu").accuracy((4, 16), trials=64, seed=0)
        assert [a.distance for a in found] == [4, 16]
        assert all(0.0 <= a.accuracy <= 1.0 and a.trials == 64 for a in found)

    def test_probe_is_deterministic(self):
        a = Associative(tiny(300), "cpu").accuracy((8,), 32, 1)
        assert a == Associative(tiny(300), "cpu").accuracy((8,), 32, 1)

    def test_a_planted_lookup_model_scores_one(self):
        # Oracle: logits peak at the token that followed the last key earlier.
        class Oracle(model.TinyLM):
            """Logits peak at the token that followed the last key earlier."""

            def forward(
                self,
                tokens: torch.Tensor,
                causal: bool = True,
                local_window: int | None = None,
            ) -> torch.Tensor:
                out = torch.zeros(*tokens.shape, self.config.vocab_size)
                key = tokens[:, -1]
                for row in range(tokens.shape[0]):
                    first = int((tokens[row] == key[row]).nonzero()[0])
                    out[row, -1, tokens[row, first + 1]] = 10.0
                return out

        config = model.Config(
            vocab_size=300, max_seq_len=64, d_model=32, n_layers=2, n_heads=2
        )
        found = Associative(Oracle(config), "cpu").accuracy((8, 30), 20, 0)
        assert all(a.accuracy == 1.0 for a in found)

    def test_rejects_bad_arguments(self):
        probe = Associative(tiny(300), "cpu")
        with pytest.raises(ValueError, match="distance"):
            probe.accuracy((0,), 8, 0)
        with pytest.raises(ValueError, match="max_seq_len"):
            probe.accuracy((100,), 8, 0)
        with pytest.raises(ValueError, match="trials"):
            probe.accuracy((8,), 0, 0)


class TestSuperposition:
    def test_effective_rank_of_planted_matrices(self):
        assert Superposition.rank(torch.eye(8)) == pytest.approx(8.0)
        one = torch.ones(10, 1) @ torch.arange(1.0, 6.0)[None, :]
        assert Superposition.rank(one) == pytest.approx(1.0, abs=1e-4)
        seeding.set_seed(0)
        assert Superposition.rank(torch.randn(500, 16)) > 12

    def test_interference_of_planted_matrices(self):
        assert Superposition.interference(torch.eye(8)) == pytest.approx(0.0)
        assert Superposition.interference(torch.ones(5, 4)) == pytest.approx(
            1.0
        )

    def test_interference_rejects_too_few_rows_and_bad_shapes(self):
        with pytest.raises(ValueError, match="rows"):
            Superposition.interference(torch.ones(1, 4))
        with pytest.raises(ValueError, match="2-D"):
            Superposition.rank(torch.ones(4))

    def test_hidden_collects_one_state_per_token(self):
        net = tiny(300)
        states = Superposition(net).hidden(
            torch.randint(0, 300, (3, 10)), layer=0
        )
        assert states.shape == (30, 32)
        with pytest.raises(ValueError, match="layer"):
            Superposition(net).hidden(torch.randint(0, 300, (3, 10)), layer=5)


class TestSae:
    def planted(self) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(0)
        features = torch.nn.functional.normalize(
            torch.randn(48, 16, generator=generator), dim=1
        )
        picks = torch.randint(0, 48, (4000, 2), generator=generator)
        weights = torch.rand(4000, 2, generator=generator) + 0.5
        return (features[picks] * weights[..., None]).sum(1)

    def test_recovers_more_features_than_dimensions(self):
        found = Sae(width=16, dictionary=96, steps=1500, l1=1e-3, seed=0).fit(
            self.planted()
        )
        assert found.r2 > 0.8
        assert found.alive > 16
        assert found.per_dim == pytest.approx(found.alive / 16)

    def test_fit_is_deterministic(self):
        a = Sae(16, 96, 200, 1e-3, 0).fit(self.planted())
        assert a == Sae(16, 96, 200, 1e-3, 0).fit(self.planted())

    def test_validates(self):
        with pytest.raises(ValueError, match="dictionary"):
            Sae(width=16, dictionary=8, steps=10, l1=1e-3, seed=0)
        with pytest.raises(ValueError, match="width"):
            Sae(16, 96, 10, 1e-3, 0).fit(torch.randn(100, 7))
