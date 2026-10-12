"""Tests for the training loop, schedules, resume and evaluation."""

import pathlib

import pytest
import torch

from tinylm import ledger, model, seeding
from tinylm.tokens import Token
from tinylm.train import Error, Train

VOCAB = 32


def net() -> model.TinyLM:
    seeding.set_seed(0)
    return model.TinyLM(
        model.Config(
            vocab_size=VOCAB,
            max_seq_len=16,
            d_model=16,
            n_layers=2,
            n_heads=2,
            window=4,
            global_every=2,
        )
    )


def noise(n: int = 2000) -> Token:
    generator = torch.Generator()
    generator.manual_seed(0)
    return Token.from_tensor(
        torch.randint(0, VOCAB, (n,), generator=generator), VOCAB
    )


def periodic(n: int = 2000) -> Token:
    return Token.from_tensor((torch.arange(n) % 8) + 3, VOCAB)


def train(**overrides: object) -> Train:
    values: dict[str, object] = {
        "seq_len": 16,
        "batch": 8,
        "lr": 3e-3,
        "floor": 0.1,
        "warmup": 5,
        "decay": 0.0,
        "clip": 1.0,
        "epochs": 2,
        "seed": 0,
        "device": "cpu",
        "every": 10,
    }
    values.update(overrides)
    return Train(**values)  # pyright: ignore[reportArgumentType]


def same(a: model.TinyLM, b: model.TinyLM) -> bool:
    return all(
        torch.equal(p, q)
        for p, q in zip(a.parameters(), b.parameters(), strict=True)
    )


class TestSchedule:
    def test_warmup_peak_floor_and_bounds_for_cosine(self):
        t = train()
        rates = [t.rate(s, 100, 10) for s in range(11)]
        assert rates == sorted(rates) and rates[-1] == pytest.approx(3e-3)
        assert t.rate(100, 100, 10) == pytest.approx(3e-4)
        assert all(
            3e-4 - 1e-9 <= t.rate(s, 100, 10) <= 3e-3 + 1e-9
            for s in range(11, 101)
        )

    def test_wsd_has_flat_stable_phase_then_monotone_decay(self):
        t = train(schedule="wsd", tail=0.2)
        assert all(
            t.rate(s, 100, 10) == pytest.approx(3e-3) for s in range(10, 80)
        )
        tail = [t.rate(s, 100, 10) for s in range(80, 101)]
        assert tail == sorted(tail, reverse=True) and tail[-1] == pytest.approx(
            3e-4
        )

    @pytest.mark.parametrize(
        "args", [(-1, 100, 10), (0, 0, 10), (0, 100, -1), (0, 100, 100)]
    )
    def test_rate_rejects_bad_arguments(self, args: tuple[int, int, int]):
        with pytest.raises(ValueError):
            train().rate(*args)


class TestValidation:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("batch", 0),
            ("seq_len", 0),
            ("lr", 0.0),
            ("epochs", 0),
            ("epochs", 11),
            ("clip", 0.0),
            ("every", 0),
            ("masked", -0.1),
            ("masked", 1.5),
            ("corrupt", 0.0),
            ("corrupt", 1.0),
            ("optimizer", "sgd"),
            ("schedule", "step"),
            ("tail", 0.0),
            ("tail", 1.0),
            ("ema", -0.1),
            ("ema", 1.0),
            ("warmup", -1),
            ("decay", -0.1),
            ("floor", 1.5),
        ],
    )
    def test_rejects_bad_fields(self, field: str, value: object):
        with pytest.raises(ValueError, match=field):
            train(**{field: value})

    def test_rejects_seq_len_over_model_and_short_stream(
        self, tmp_path: pathlib.Path
    ):
        with pytest.raises(ValueError, match="max_seq_len"):
            train(seq_len=64).run(net(), noise(), tmp_path)
        with pytest.raises(ValueError, match="short"):
            train().run(net(), noise(50), tmp_path)

    def test_resume_without_checkpoint_is_rejected(
        self, tmp_path: pathlib.Path
    ):
        with pytest.raises(FileNotFoundError):
            train().run(net(), noise(), tmp_path, resume=True)


class TestRun:
    def test_learns_a_periodic_stream(self, tmp_path: pathlib.Path):
        result = train(epochs=10).run(net(), periodic(), tmp_path)
        assert result.losses[-1] < 0.5 * result.losses[0]

    def test_ledger_counts_epochs_and_never_exceeds_the_cap(
        self, tmp_path: pathlib.Path
    ):
        data = noise()
        result = train().run(net(), data, tmp_path)
        assert 1.0 < result.budget.epochs_used <= 2.0
        assert result.budget.consumed <= 2 * len(data)

    def test_exhausted_budget_stops_training(self, tmp_path: pathlib.Path):
        data = noise()
        small = ledger.Budget(unique_tokens=len(data), max_epochs=1)
        with pytest.raises(ledger.Error):
            train().run(net(), data, tmp_path, budget=small)

    def test_identical_runs_give_identical_weights(
        self, tmp_path: pathlib.Path
    ):
        a, b = net(), net()
        train().run(a, noise(), tmp_path / "a")
        train().run(b, noise(), tmp_path / "b")
        assert same(a, b)

    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"masked": 0.5},
            {"optimizer": "muon", "lr": 2e-3},
            {"ema": 0.9},
            {"schedule": "wsd"},
            {"shift": False},
        ],
    )
    def test_resume_equals_uninterrupted_run(
        self, tmp_path: pathlib.Path, extra: dict[str, object]
    ):
        config = train(**extra)
        full, resumed = net(), net()
        config.run(full, noise(), tmp_path / "full")
        config.run(net(), noise(), tmp_path / "part", stop=7)
        config.run(resumed, noise(), tmp_path / "part", resume=True)
        for p, q in zip(full.parameters(), resumed.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_non_finite_loss_raises(self, tmp_path: pathlib.Path):
        broken = net()
        with torch.no_grad():
            broken.final_norm.weight.fill_(float("nan"))
        with pytest.raises(Error, match="non-finite"):
            train().run(broken, noise(), tmp_path)

    def test_hybrid_and_ema_change_the_result_but_stay_valid(
        self, tmp_path: pathlib.Path
    ):
        base, hybrid, averaged = net(), net(), net()
        train().run(base, noise(), tmp_path / "a")
        train(masked=0.5).run(hybrid, noise(), tmp_path / "b")
        train(ema=0.9).run(averaged, noise(), tmp_path / "c")
        assert not same(base, hybrid) and not same(base, averaged)
        live = torch.load(tmp_path / "c" / "checkpoint.pt")["model"]
        assert any(
            not torch.equal(live[k], v)
            for k, v in averaged.state_dict().items()
        )

    def test_epoch_callback_runs_once_per_epoch_in_order(
        self, tmp_path: pathlib.Path
    ):
        seen: list[int] = []
        train(epochs=3).run(net(), noise(), tmp_path, after=seen.append)
        assert seen == [0, 1, 2]
        with pytest.raises(RuntimeError, match="stop at 0"):

            def boom(epoch: int) -> None:
                raise RuntimeError(f"stop at {epoch}")

            train().run(net(), noise(), tmp_path / "x", after=boom)

    def test_checkpoint_is_written_atomically(self, tmp_path: pathlib.Path):
        train().run(net(), noise(), tmp_path)
        assert (tmp_path / "checkpoint.pt").is_file()
        assert not list(tmp_path.glob("*.tmp"))


class TestEvaluate:
    def test_untrained_loss_is_near_log_vocab_and_training_improves_it(
        self, tmp_path: pathlib.Path
    ):
        data = periodic()
        model_ = net()
        before = train().evaluate(model_, data)
        assert before == pytest.approx(
            float(torch.tensor(float(VOCAB)).log()), abs=0.5
        )
        train(epochs=10).run(model_, data, tmp_path)
        assert train().evaluate(model_, data) < before

    def test_rejects_stream_shorter_than_one_sequence(self):
        with pytest.raises(ValueError, match="short"):
            train().evaluate(net(), noise(10))
