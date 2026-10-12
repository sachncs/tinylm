"""Tests for the training loop, schedules, resume and evaluation."""

import pathlib
from typing import Any

import pytest
import torch

from tinylm import ledger, model, seeding
from tinylm.device import Error as DeviceError
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


def train(**overrides: Any) -> Train:
    values: dict[str, Any] = {
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
    return Train(**values)


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
        self, tmp_path: pathlib.Path, extra: dict[str, Any]
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
        # The checkpoint of a finished run holds the model that was returned.
        saved = torch.load(tmp_path / "c" / "checkpoint.pt")["model"]
        assert all(
            torch.equal(saved[k], v) for k, v in averaged.state_dict().items()
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


class TestIntegrity:
    def test_resume_refuses_a_changed_config_or_changed_data(
        self, tmp_path: pathlib.Path
    ):
        train(lr=3e-3).run(net(), noise(), tmp_path, stop=3)
        with pytest.raises(Error, match="changed"):
            train(lr=1e-3).run(net(), noise(), tmp_path, resume=True)
        with pytest.raises(Error, match="changed"):
            train(lr=3e-3).run(net(), noise(1999), tmp_path, resume=True)
        train(lr=3e-3).run(net(), noise(), tmp_path, resume=True)

    def test_checkpoint_cadence_does_not_count_as_a_change(
        self, tmp_path: pathlib.Path
    ):
        train(every=10).run(net(), noise(), tmp_path, stop=3)
        train(every=7).run(net(), noise(), tmp_path, resume=True)

    def test_unknown_checkpoint_version_and_foreign_files_fail_loudly(
        self, tmp_path: pathlib.Path
    ):
        train().run(net(), noise(), tmp_path, stop=3)
        saved = torch.load(tmp_path / "checkpoint.pt")
        saved["version"] = 99
        torch.save(saved, tmp_path / "checkpoint.pt")
        with pytest.raises(Error, match="version"):
            train().run(net(), noise(), tmp_path, resume=True)
        torch.save({"hello": 1}, tmp_path / "checkpoint.pt")
        with pytest.raises(Error, match="not a valid checkpoint"):
            train().run(net(), noise(), tmp_path, resume=True)

    def test_run_reports_training_seconds_and_uses_loud_device_resolution(
        self, tmp_path: pathlib.Path
    ):
        assert train().run(net(), noise(), tmp_path).seconds > 0.0
        with pytest.raises(DeviceError, match="device"):
            train(device="tpu").run(net(), noise(), tmp_path / "x")


class TestHalt:
    def test_halt_stops_at_a_step_boundary_with_a_resumable_checkpoint(
        self, tmp_path: pathlib.Path
    ):
        data = noise()
        calls = {"n": 0}

        def halt() -> bool:
            calls["n"] += 1
            return calls["n"] > 7

        first = train().run(net(), data, tmp_path / "h", halt=halt)
        assert first.interrupted and first.steps == 7
        resumed = net()
        done = train().run(resumed, data, tmp_path / "h", resume=True)
        reference = net()
        train().run(reference, data, tmp_path / "ref")
        assert not done.interrupted and done.steps == first.steps + len(
            done.losses
        )
        for p, q in zip(
            resumed.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_a_finished_run_is_not_marked_interrupted(
        self, tmp_path: pathlib.Path
    ):
        assert not train().run(net(), noise(), tmp_path).interrupted

    def test_halt_that_never_fires_changes_nothing(
        self, tmp_path: pathlib.Path
    ):
        a, b = net(), net()
        train().run(a, noise(), tmp_path / "a")
        train().run(b, noise(), tmp_path / "b", halt=lambda: False)
        assert same(a, b)


class TestCompile:
    def available(self) -> str:
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def test_compiled_training_matches_eager_training(
        self, tmp_path: pathlib.Path
    ):
        device = self.available()
        eager, fast = net(), net()
        a = train(device=device, epochs=1).run(eager, noise(), tmp_path / "a")
        b = train(device=device, epochs=1, compile=True).run(
            fast, noise(), tmp_path / "b"
        )
        assert len(a.losses) == len(b.losses) > 10
        assert (
            max(abs(x - y) for x, y in zip(a.losses, b.losses, strict=True))
            < 1e-3
        )
        # Adam normalises tiny gradients, so float rounding differences between
        # fused and eager kernels show up as ~1e-3 in weights while the losses
        # still agree; the tolerance is one learning-rate step, not exactness.
        for p, q in zip(eager.parameters(), fast.parameters(), strict=True):
            torch.testing.assert_close(p.cpu(), q.cpu(), rtol=0, atol=5e-3)

    def test_compiled_run_checkpoints_and_resumes_like_a_plain_one(
        self, tmp_path: pathlib.Path
    ):
        device = self.available()
        config = train(device=device, compile=True)
        config.run(net(), noise(), tmp_path, stop=5)
        saved = torch.load(tmp_path / "checkpoint.pt")["model"]
        assert not any(key.startswith("_orig_mod") for key in saved)
        resumed = net()
        config.run(resumed, noise(), tmp_path, resume=True)
        reference = net()
        config.run(reference, noise(), tmp_path / "ref")
        for p, q in zip(
            resumed.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(p.cpu(), q.cpu(), rtol=0, atol=5e-3)

    def test_compile_changes_the_fingerprint_so_runs_are_not_mixed(
        self, tmp_path: pathlib.Path
    ):
        train(compile=False).run(net(), noise(), tmp_path, stop=3)
        with pytest.raises(Error, match="changed"):
            train(compile=True).run(net(), noise(), tmp_path, resume=True)
