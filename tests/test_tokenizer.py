"""Tests for deterministic tokenizer training."""

import pathlib

import pytest

from tinylm import tokenizer

CORPUS = [
    "the cat sat on the mat.",
    "the dog sat on the log.",
    "a cat and a dog are friends.",
    'She said, "hello there!"',
] * 50


@pytest.mark.parametrize("kind", ["bpe", "unigram"])
class TestTrain:
    def test_special_ids_vocab_size_and_roundtrip(self, kind: str):
        tok = tokenizer.train(CORPUS, vocab_size=300, kind=kind)
        for index, name in enumerate(tokenizer.SPECIAL_TOKENS):
            assert tok.token_to_id(name) == index
        assert tok.get_vocab_size() <= 300
        text = "the cat sat on the mat."
        assert tok.decode(tok.encode(text).ids) == text

    def test_training_is_deterministic_over_many_runs(self, kind: str):
        seen = {
            tokenizer.train(CORPUS, vocab_size=300, kind=kind).to_str()
            for _ in range(15)
        }
        assert len(seen) == 1

    def test_unseen_characters_do_not_crash(self, kind: str):
        tok = tokenizer.train(CORPUS, vocab_size=300, kind=kind)
        assert tok.encode("zebra \u4e2d\u6587 \U0001f600").ids

    def test_rejects_tiny_vocab(self, kind: str):
        with pytest.raises(ValueError, match="vocab_size"):
            tokenizer.train(CORPUS, vocab_size=3, kind=kind)


def test_rejects_unknown_kind_and_empty_corpora():
    with pytest.raises(ValueError, match="kind"):
        tokenizer.train(CORPUS, vocab_size=300, kind="wordpiece")
    for bad in ([], ["", "  "]):
        with pytest.raises(ValueError, match="corpus"):
            tokenizer.train(bad, vocab_size=300, kind="bpe")


def test_save_load_and_missing_file(tmp_path: pathlib.Path):
    tok = tokenizer.train(CORPUS, vocab_size=300, kind="bpe")
    path = tmp_path / "tok.json"
    tokenizer.save(tok, path)
    text = "the dog sat."
    assert tokenizer.load(path).encode(text).ids == tok.encode(text).ids
    with pytest.raises(FileNotFoundError):
        tokenizer.load(tmp_path / "missing.json")
