"""Tests for the in-process minimal-pair benchmark scorer."""

import json
import math
import pathlib

import pytest

from tinylm import model, seeding, tokenizer
from tinylm.bench import Bench, Confidence, Error, Pair

LINES = ["the cat sat on the mat .", "the dogs run fast ."] * 40


def tok():
    return tokenizer.train(LINES, vocab_size=300, kind="bpe")


def bench(**kw: object) -> Bench:
    seeding.set_seed(0)
    tokenizer_ = tok()
    net = model.TinyLM(
        model.Config(
            vocab_size=tokenizer_.get_vocab_size(),
            max_seq_len=32,
            d_model=16,
            n_layers=2,
            n_heads=2,
            window=4,
            global_every=2,
        )
    )
    return Bench(net, tokenizer_, "cpu", **kw)  # pyright: ignore[reportArgumentType]


def write(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class TestPairs:
    def test_read_good_and_bad(self, tmp_path: pathlib.Path):
        write(
            tmp_path / "a.jsonl",
            [{"sentence_good": "a b", "sentence_bad": "b a"}],
        )
        assert Pair.read(tmp_path / "a.jsonl") == [Pair("a b", "b a")]

    def test_rejects_missing_field_bad_json_and_empty(
        self, tmp_path: pathlib.Path
    ):
        write(tmp_path / "a.jsonl", [{"sentence_good": "x"}])
        with pytest.raises(Error, match="sentence_bad"):
            Pair.read(tmp_path / "a.jsonl")
        (tmp_path / "b.jsonl").write_text("{oops\n")
        with pytest.raises(Error, match="line 1"):
            Pair.read(tmp_path / "b.jsonl")
        (tmp_path / "c.jsonl").write_text("")
        with pytest.raises(Error, match="empty"):
            Pair.read(tmp_path / "c.jsonl")


class TestConfidence:
    def test_precision_and_coverage_at_threshold(self):
        report = Confidence.of([3.0, -2.5, 0.2, -0.1, 1.5, 4.0], tau=1.0)
        assert (report.confident, report.correct) == (4, 3)
        assert report.precision == pytest.approx(3 / 4)
        assert report.coverage == pytest.approx(4 / 6)

    def test_larger_tau_never_increases_coverage(self):
        margins = [0.3, 0.8, 1.2, 2.5, -0.6, -3.0]
        coverage = [
            Confidence.of(margins, t).coverage for t in (0.25, 0.5, 1, 2)
        ]
        assert coverage == sorted(coverage, reverse=True)

    def test_boundary_ties_and_nothing_confident(self):
        assert Confidence.of([1.0, -1.0], 1.0).confident == 2
        assert Confidence.of([0.0, 0.0, 2.0], 0.0).correct == 1
        none = Confidence.of([0.1, -0.2], 1.0)
        assert none.precision is None and none.coverage == 0.0

    @pytest.mark.parametrize("tau", [-1.0, float("nan")])
    def test_rejects_bad_tau(self, tau: float):
        with pytest.raises(ValueError, match="tau"):
            Confidence.of([1.0], tau)

    def test_rejects_empty_margins(self):
        with pytest.raises(ValueError, match="margins"):
            Confidence.of([], 1.0)


class TestScoring:
    def test_scores_are_negative_finite_and_batch_invariant(self):
        b = bench()
        a, c = "the cat sat on the mat .", "the dogs run fast ."
        together = b.score([a, c])
        assert all(s < 0 and math.isfinite(s) for s in together)
        assert together == pytest.approx(
            [b.score([a])[0], b.score([c])[0]], abs=1e-4
        )

    def test_longer_sentence_has_lower_logprob(self):
        short, long = bench().score(
            ["the cat", "the cat sat on the mat . the cat sat"]
        )
        assert long < short

    def test_rejects_empty_and_overlong_sentences(self):
        with pytest.raises(ValueError, match="empty"):
            bench().score([""])
        with pytest.raises(ValueError, match="max_seq_len"):
            bench().score(["the cat sat on the mat . " * 40])

    def test_margins_and_accuracy(self, monkeypatch: pytest.MonkeyPatch):
        b = bench()
        table = {
            "a": -1.0,
            "b": -3.0,
            "c": -4.0,
            "d": -2.5,
            "e": -1.0,
            "f": -1.0,
        }

        def fake(_self: Bench, xs: list[str]) -> list[float]:
            return [table[x] for x in xs]

        monkeypatch.setattr(Bench, "score", fake)
        pairs = [Pair("a", "b"), Pair("c", "d"), Pair("e", "f")]
        assert b.margins(pairs) == [2.0, -1.5, 0.0]
        assert b.accuracy(pairs) == pytest.approx(1 / 3)
        with pytest.raises(ValueError, match="pairs"):
            b.margins([])


@pytest.mark.e2e
class TestDirectory:
    def pairs(self, directory: pathlib.Path) -> None:
        for name in ("p1", "p2"):
            write(
                directory / f"{name}.jsonl",
                [
                    {
                        "sentence_good": "the cat sat on the mat .",
                        "sentence_bad": "the mat on sat cat the .",
                    },
                    {
                        "sentence_good": "the dogs run fast .",
                        "sentence_bad": "fast run dogs the .",
                    },
                ],
            )

    def test_report_has_paradigms_mean_and_pooled_confidence(
        self, tmp_path: pathlib.Path
    ):
        self.pairs(tmp_path)
        report = bench().evaluate(tmp_path, taus=(0.5, 2.0))
        assert set(report.by_paradigm) == {"p1", "p2"}
        assert report.mean == pytest.approx(
            sum(report.by_paradigm.values()) / 2
        )
        assert report.confidence[0.5].total == 4
        assert (
            report.confidence[2.0].coverage <= report.confidence[0.5].coverage
        )

    def test_negative_directories(self, tmp_path: pathlib.Path):
        with pytest.raises(FileNotFoundError):
            bench().evaluate(tmp_path / "x")
        with pytest.raises(Error, match="no .jsonl"):
            bench().evaluate(tmp_path)
