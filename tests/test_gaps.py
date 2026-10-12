"""Tests for error paths and edge cases found by the coverage review."""

import dataclasses
import json
import pathlib

import huggingface_hub
import numpy
import pytest
import torch

from tinylm import ema, fetch, model, objectives, seeding
from tinylm.bench import Error as BenchError
from tinylm.bench import Pair
from tinylm.bootstrap import Bootstrap, Interval
from tinylm.clean import Options
from tinylm.device import Device
from tinylm.experiment import Error, Experiment, Queue, build, coerce
from tinylm.prepare import Prepare
from tinylm.probes import Sae, Superposition
from tinylm.screen import Point
from tinylm.tokens import Token
from tinylm.train import Train


def test_token_constructor_validates_directly():
    good = numpy.arange(10, dtype=numpy.uint16)
    with pytest.raises(ValueError, match="vocab"):
        Token(good, vocab=0)
    with pytest.raises(ValueError, match="1-D"):
        Token(good.reshape(2, 5), vocab=100)
    with pytest.raises(ValueError, match="empty"):
        Token(good[:0], vocab=100)
    with pytest.raises(ValueError, match="negative"):
        Token(numpy.array([1, -1], dtype=numpy.int64), vocab=100)
    with pytest.raises(ValueError, match="outside vocab"):
        Token(good, vocab=5)


def test_load_reports_missing_files_inside_an_existing_directory(
    tmp_path: pathlib.Path,
):
    (tmp_path / "s").mkdir()
    with pytest.raises(FileNotFoundError, match="lacks"):
        Token.load(tmp_path / "s")


def test_interval_and_paired_groups_validation():
    with pytest.raises(ValueError, match="level"):
        Interval(0.5, 0.4, 0.6, 1.5)
    with pytest.raises(ValueError, match="non-empty"):
        Bootstrap().paired({"a": []}, {"a": []})


def test_cuda_branches_call_the_cuda_api(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.append("release")
    )
    Device("cuda").sync()
    Device("cuda").release()
    assert calls == ["sync", "release"]


def test_fetch_download_asks_huggingface_for_the_pinned_dataset_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
):
    seen: dict[str, object] = {}

    def fake(**kwargs: object) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    fetch.download("org/name", "a" * 40, tmp_path, ("*.txt",))
    assert seen == {
        "repo_id": "org/name",
        "repo_type": "dataset",
        "revision": "a" * 40,
        "local_dir": tmp_path,
        "allow_patterns": ["*.txt"],
    }


def test_blank_lines_in_pair_files_are_skipped(tmp_path: pathlib.Path):
    path = tmp_path / "p.jsonl"
    row = json.dumps({"sentence_good": "a b", "sentence_bad": "b a"})
    path.write_text(f"\n{row}\n\n")
    assert Pair.read(path) == [Pair("a b", "b a")]
    (tmp_path / "e.jsonl").write_text("\n\n")
    with pytest.raises(BenchError, match="empty"):
        Pair.read(tmp_path / "e.jsonl")


def test_misc_validation_paths():
    with pytest.raises(ValueError, match="mask_rate"):
        objectives.Hybrid(0.5, 0.0, 4)
    with pytest.raises(ValueError, match="unique"):
        Prepare(
            vocab=300, kind="bpe", block=1, every=1, options=Options(), unique=0
        )
    with pytest.raises(ValueError, match="width"):
        Sae(width=0, dictionary=4, steps=1, l1=0.0, seed=0)
    with pytest.raises(ValueError, match="2-D"):
        Superposition.interference(torch.ones(4))
    with pytest.raises(ValueError, match="precision"):
        Point("x", 0.5, 1.5, 1.0)


def test_dev_set_that_is_entirely_leaked_is_an_error(tmp_path: pathlib.Path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    lines = [f"alpha beta gamma delta epsilon zeta eta {i}" for i in range(5)]
    (corpus / "gutenberg.train.txt").write_text("\n".join(lines + lines) + "\n")
    with pytest.raises(ValueError, match="held-out"):
        Prepare(vocab=300, kind="bpe", block=5, every=2, options=Options()).run(
            corpus, tmp_path / "out"
        )


def test_experiment_file_errors(tmp_path: pathlib.Path):
    path = tmp_path / "e.toml"
    path.write_text(
        'name = "x"\ndata = "d"\nout = "o"\nbogus = 1\n[model]\n[train]\n'
    )
    with pytest.raises(Error, match="top-level"):
        Experiment.load(path)
    path.write_text('data = "d"\nout = "o"\n[model]\n[train]\n')
    with pytest.raises(Error, match="missing key"):
        Experiment.load(path)
    with pytest.raises(FileNotFoundError):
        Queue.load(tmp_path / "none.toml")
    path.write_text("name = ")
    with pytest.raises(Error, match="toml"):
        Queue.load(path)
    with pytest.raises(Error, match="missing"):
        build(Train, "not a table", "train")


def test_coerce_booleans():
    field = next(f for f in dataclasses.fields(Train) if f.name == "shift")
    assert coerce(field, "true") is True and coerce(field, "false") is False
    with pytest.raises(ValueError, match="true or false"):
        coerce(field, "yes")


def test_resume_after_a_whole_epoch_skips_it_without_repeating_work(
    tmp_path: pathlib.Path,
):
    seeding.set_seed(0)
    net = model.TinyLM(
        model.Config(
            vocab_size=32,
            max_seq_len=16,
            d_model=16,
            n_layers=2,
            n_heads=2,
            window=4,
            global_every=2,
        )
    )
    generator = torch.Generator()
    generator.manual_seed(0)
    data = Token.from_tensor(
        torch.randint(0, 32, (2000,), generator=generator), 32
    )
    config = Train(
        seq_len=16,
        batch=8,
        lr=3e-3,
        floor=0.1,
        warmup=5,
        decay=0.0,
        clip=1.0,
        epochs=3,
        seed=0,
        device="cpu",
        every=10,
    )
    first = config.batches(len(data), 0)
    config.run(net, data, tmp_path, stop=first)
    seen: list[int] = []
    result = config.run(net, data, tmp_path, resume=True, after=seen.append)
    assert seen == [1, 2] and result.steps > first


def test_ema_keeps_buffers_consistent_with_model_dtype():
    seeding.set_seed(0)
    net = model.TinyLM(
        model.Config(
            vocab_size=32, max_seq_len=16, d_model=16, n_layers=2, n_heads=2
        )
    )
    average = ema.Average(net, 0.5)
    average.update(net)
    assert all(
        v.dtype == net.state_dict()[k].dtype for k, v in average.shadow.items()
    )
