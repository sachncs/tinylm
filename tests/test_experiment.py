"""Tests for TOML experiments: parse, override, queue and run."""

import dataclasses
import json
import pathlib

import pytest

from tinylm.clean import Options
from tinylm.experiment import Error, Experiment, Queue
from tinylm.prepare import Prepare
from tinylm.tokens import Token
from tinylm.train import Error as TrainError

TOML = """
name = "tiny"
data = "{data}"
out = "{out}"
{extra}
[model]
max_seq_len = 16
d_model = 16
n_layers = 2
n_heads = 2
window = 4
global_every = 2

[train]
seq_len = 16
batch = 4
lr = 0.003
floor = 0.1
warmup = 2
decay = 0.0
clip = 1.0
epochs = 2
seed = 0
device = "cpu"
every = 5
"""


def config(tmp: pathlib.Path, data: str = "d", extra: str = "") -> pathlib.Path:
    path = tmp / "c.toml"
    path.write_text(TOML.format(data=tmp / data, out=tmp / "o", extra=extra))
    return path


def prepared(tmp: pathlib.Path) -> pathlib.Path:
    corpus = tmp / "corpus"
    corpus.mkdir()
    (corpus / "childes.train.txt").write_text(
        "".join(f"*MOT:\tmore {i % 7} cookies please?\n" for i in range(600))
    )
    out = tmp / "d"
    Prepare(vocab=300, kind="bpe", block=20, every=10, options=Options()).run(
        corpus, out
    )
    return out


class TestLoad:
    def test_parses_valid_file(self, tmp_path: pathlib.Path):
        loaded = Experiment.load(config(tmp_path))
        assert (loaded.name, loaded.train.epochs, loaded.model.d_model) == (
            "tiny",
            2,
            16,
        )

    def test_rejects_missing_invalid_unknown_and_incomplete(
        self, tmp_path: pathlib.Path
    ):
        with pytest.raises(FileNotFoundError):
            Experiment.load(tmp_path / "none.toml")
        bad = tmp_path / "b.toml"
        bad.write_text("name = ")
        with pytest.raises(Error, match="toml"):
            Experiment.load(bad)
        text = config(tmp_path).read_text()
        bad.write_text(text.replace("[model]", "[model]\nbogus = 1"))
        with pytest.raises(Error, match="bogus"):
            Experiment.load(bad)
        bad.write_text('name = "x"\ndata = "a"\nout = "b"\n')
        with pytest.raises(Error, match="model"):
            Experiment.load(bad)

    def test_semantic_and_budget_errors(self, tmp_path: pathlib.Path):
        path = config(tmp_path)
        text = path.read_text()
        path.write_text(text.replace("epochs = 2", "epochs = 11"))
        with pytest.raises(Error, match="epochs"):
            Experiment.load(path)
        path.write_text(
            text.replace("d_model = 16", "d_model = 2048").replace(
                "n_layers = 2", "n_layers = 24"
            )
        )
        with pytest.raises(Error, match="budget"):
            Experiment.load(path, vocab=16_384)


class TestOverride:
    def base(self, tmp: pathlib.Path) -> Experiment:
        return Experiment.load(config(tmp))

    def test_numeric_string_and_tuple_overrides(self, tmp_path: pathlib.Path):
        variant = self.base(tmp_path).override(
            [
                "train.lr=0.01",
                "train.epochs=3",
                "model.d_model=32",
                "train.device=cpu",
                "model.n_layers=4",
                "model.codebook_sizes=48,24",
            ],
            name="v",
        )
        assert variant.train.lr == 0.01 and variant.train.epochs == 3
        assert variant.model.d_model == 32 and variant.model.codebook_sizes == (
            48,
            24,
        )
        assert variant.name == "v" and variant.out == tmp_path / "v"

    def test_empty_tuple_data_override_and_immutability(
        self, tmp_path: pathlib.Path
    ):
        base = self.base(tmp_path)
        assert (
            base.override(["model.codebook_sizes="], "v").model.codebook_sizes
            == ()
        )
        assert base.override(["data=elsewhere/prep"], "v").data == pathlib.Path(
            "elsewhere/prep"
        )
        base.override(["train.epochs=5"], "v")
        assert base.train.epochs == 2

    @pytest.mark.parametrize(
        "assignment",
        [
            "train.bogus=1",
            "nosection.x=1",
            "train.epochs",
            "train.epochs=abc",
            "train.epochs=1.5",
            "model.d_model=0",
            "train.epochs=11",
            "name=x",
            "model.codebook_sizes=a,b",
        ],
    )
    def test_bad_assignments_are_errors(
        self, tmp_path: pathlib.Path, assignment: str
    ):
        with pytest.raises(Error):
            self.base(tmp_path).override([assignment], "v")

    def test_over_budget_override_is_rejected(self, tmp_path: pathlib.Path):
        with pytest.raises(Error, match="budget"):
            self.base(tmp_path).override(
                ["model.d_model=2048", "model.n_layers=24"], "v", vocab=16_384
            )


class TestQueue:
    def write(self, tmp: pathlib.Path, body: str) -> pathlib.Path:
        path = tmp / "q.toml"
        path.write_text(body)
        return path

    def test_parse_and_pending(self, tmp_path: pathlib.Path):
        queue = Queue.load(
            self.write(
                tmp_path,
                'base = "x.toml"\n[[run]]\nname = "a"\n'
                'set = ["train.epochs=3"]\n'
                '[[run]]\nname = "b"\n',
            )
        )
        assert [(r.name, r.sets) for r in queue.runs] == [
            ("a", ("train.epochs=3",)),
            ("b", ()),
        ]
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "metrics.json").write_text("{}")
        assert [r.name for r in queue.pending(tmp_path)] == ["b"]

    @pytest.mark.parametrize(
        "body,match",
        [
            (
                'base = "x"\n[[run]]\nname = "a"\n[[run]]\nname = "a"\n',
                "duplicate",
            ),
            ('[[run]]\nname = "a"\n', "base"),
            ('base = "x"\n', "run"),
            ('base = "x"\n[[run]]\nset = []\n', "name"),
        ],
    )
    def test_rejects_malformed(
        self, tmp_path: pathlib.Path, body: str, match: str
    ):
        with pytest.raises(Error, match=match):
            Queue.load(self.write(tmp_path, body))


@pytest.mark.e2e
class TestRun:
    def test_end_to_end_writes_metrics_and_checkpoint(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        metrics = Experiment.load(config(tmp_path)).run()
        saved = json.loads((tmp_path / "o" / "metrics.json").read_text())
        assert saved["parameters"] == metrics.parameters > 0
        assert (
            len(saved["dev_loss"]) == 2
            and saved["dev_loss"][-1] < saved["dev_loss"][0]
        )
        assert saved["epochs_used"] <= 2.0 and saved["bits_per_char"] > 0
        assert (tmp_path / "o" / "checkpoint.pt").is_file()
        # With no weight averaging the final model is the last epoch's model.
        assert saved["final_loss"] == pytest.approx(saved["dev_loss"][-1])
        trace = saved["provenance"]
        assert len(trace["revision"]) == 40 and isinstance(trace["dirty"], bool)
        assert trace["device"] == "cpu" and len(trace["config"]) == 64
        assert len(trace["data"]) == 64 and trace["torch"]
        assert saved["seconds"] > 0

    def test_benchmarks_scored_and_missing_dir_fails_before_training(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        pairs = tmp_path / "pairs"
        pairs.mkdir()
        (pairs / "p1.jsonl").write_text(
            json.dumps(
                {
                    "sentence_good": "more 3 cookies please ?",
                    "sentence_bad": "please ? cookies 3 more",
                }
            )
            + "\n"
        )
        metrics = Experiment.load(
            config(tmp_path, extra=f'evals = ["{pairs}"]\ntaus = [0.5]')
        ).run()
        assert (
            set(metrics.benchmarks) == {"pairs"}
            and 0 <= metrics.benchmarks["pairs"] <= 1
        )
        out = tmp_path / "o2"
        text = config(
            tmp_path, extra=f'evals = ["{tmp_path / "absent"}"]'
        ).read_text()
        (tmp_path / "c2.toml").write_text(
            text.replace(str(tmp_path / "o"), str(out))
        )
        with pytest.raises(FileNotFoundError):
            Experiment.load(tmp_path / "c2.toml").run()
        assert not (out / "checkpoint.pt").exists()

    def test_missing_prepared_data_is_reported(self, tmp_path: pathlib.Path):
        with pytest.raises(FileNotFoundError):
            Experiment.load(config(tmp_path, data="absent")).run()


@pytest.mark.e2e
class TestFinalModel:
    def test_with_ema_the_reported_loss_is_of_the_averaged_weights(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        experiment = Experiment.load(config(tmp_path)).override(
            ["train.ema=0.9"], "avg"
        )
        metrics = experiment.run()
        net, tok = experiment.restore()
        dev = Token.load(experiment.data / "dev")
        assert experiment.train.evaluate(net, dev) == pytest.approx(
            metrics.final_loss, rel=1e-6
        )
        assert metrics.final_loss != pytest.approx(
            metrics.dev_loss[-1], rel=1e-9
        )
        assert tok.get_vocab_size() == net.config.vocab_size

    def test_the_resolved_experiment_is_saved_next_to_the_results(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        experiment = Experiment.load(config(tmp_path))
        experiment.run()
        saved = json.loads((tmp_path / "o" / "experiment.json").read_text())
        assert saved["name"] == "tiny" and saved["train"]["epochs"] == 2
        assert saved["model"]["d_model"] == 16

    def test_from_run_rebuilds_the_identical_experiment(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        experiment = Experiment.load(config(tmp_path)).override(
            ["train.epochs=1", "model.codebook_sizes="], "again"
        )
        experiment.run()
        assert Experiment.from_run(experiment.out) == experiment

    def test_from_run_rejects_a_missing_or_damaged_description(
        self, tmp_path: pathlib.Path
    ):
        with pytest.raises(FileNotFoundError):
            Experiment.from_run(tmp_path / "nothing")
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "experiment.json").write_text("{not json")
        with pytest.raises(Error, match="experiment.json"):
            Experiment.from_run(tmp_path / "run")
        (tmp_path / "run" / "experiment.json").write_text('{"name": "x"}')
        with pytest.raises(Error, match="missing"):
            Experiment.from_run(tmp_path / "run")

    def test_restore_without_a_finished_run_fails_loudly(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        with pytest.raises(FileNotFoundError):
            Experiment.load(config(tmp_path)).restore()


@pytest.mark.e2e
class TestInterruptAndResume:
    def test_an_interrupted_run_raises_then_resumes_to_the_same_result(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        experiment = Experiment.load(config(tmp_path))
        calls = {"n": 0}

        def halt() -> bool:
            calls["n"] += 1
            return calls["n"] > 9

        with pytest.raises(Error, match="step 9") as caught:
            experiment.run(halt=halt)
        assert caught.value.resumable
        assert not (tmp_path / "o" / "metrics.json").exists()
        resumed = experiment.run(resume=True)
        clean = Experiment.load(config(tmp_path)).override([], "clean").run()
        assert resumed.dev_loss == pytest.approx(clean.dev_loss, rel=1e-6)
        assert resumed.final_loss == pytest.approx(clean.final_loss, rel=1e-6)

    def test_resume_without_a_checkpoint_starts_fresh(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        metrics = Experiment.load(config(tmp_path)).run(resume=True)
        assert len(metrics.dev_loss) == 2

    def test_resume_refuses_a_changed_configuration(
        self, tmp_path: pathlib.Path
    ):
        prepared(tmp_path)
        experiment = Experiment.load(config(tmp_path))
        calls = {"n": 0}

        def halt() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        with pytest.raises(Error):
            experiment.run(halt=halt)
        changed = dataclasses.replace(
            experiment, train=dataclasses.replace(experiment.train, lr=0.01)
        )
        assert changed.out == experiment.out
        with pytest.raises(TrainError, match="changed"):
            changed.run(resume=True)
