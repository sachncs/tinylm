"""End-to-end tests for corpus preparation."""

import json
import pathlib

import pytest
import torch

from tinylm import clean, prepare, tokenizer

pytestmark = pytest.mark.e2e


def write_corpus(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "childes.train.txt").write_text(
        "".join(
            f"*MOT:\tmore {i % 7} cookies please?\n*CHI:\tyes {i % 5}.\n"
            for i in range(400)
        )
    )
    (root / "open_subtitles.train.txt").write_text(
        "".join(f"WHO IS AT THE DOOR {i % 9}?\n" for i in range(400))
    )
    (root / "gutenberg.train.txt").write_text(
        "".join(
            f"“Well,” said Tom {i % 11}, “it’s late.”\n" for i in range(400)
        )
    )
    return root


def settings(**overrides: object) -> prepare.Settings:
    values: dict[str, object] = {
        "vocab_size": 300,
        "tokenizer_kind": "bpe",
        "dev_block_lines": 20,
        "dev_every_blocks": 10,
        "options": clean.Options(),
    }
    values.update(overrides)
    return prepare.Settings(**values)  # pyright: ignore[reportArgumentType]


def test_artifacts_counts_bos_cleaning_and_vocab_range(tmp_path: pathlib.Path):
    out = tmp_path / "out"
    stats = prepare.prepare(write_corpus(tmp_path / "c"), out, settings())
    for name in ("tokenizer.json", "train.pt", "dev.pt", "stats.json"):
        assert (out / name).is_file()
    train = torch.load(out / "train.pt")
    dev = torch.load(out / "dev.pt")
    tok = tokenizer.load(out / "tokenizer.json")
    assert train.numel() == stats.train_tokens > 5 * stats.dev_tokens > 0
    assert dev.numel() == stats.dev_tokens
    assert stats.dev_chars > stats.dev_tokens
    assert json.loads((out / "stats.json").read_text())["train_tokens"] == (
        stats.train_tokens
    )
    assert int(train.min()) >= 0 and int(train.max()) < tok.get_vocab_size()
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    assert int((train == bos).sum()) > 100 and int((train == eos).sum()) == 0
    text = tok.decode([int(i) for i in train[:2000]])
    assert "*MOT" not in text and "WHO IS" not in text and "“" not in text


def test_preparation_is_byte_deterministic(tmp_path: pathlib.Path):
    corpus = write_corpus(tmp_path / "c")
    prepare.prepare(corpus, tmp_path / "a", settings())
    prepare.prepare(corpus, tmp_path / "b", settings())
    for name in ("tokenizer.json", "train.pt", "dev.pt", "stats.json"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()


def test_negative_inputs(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        prepare.prepare(tmp_path / "nope", tmp_path / "o", settings())
    other = tmp_path / "other"
    other.mkdir()
    (other / "tweets.train.txt").write_text("hi\n")
    with pytest.raises(ValueError, match="source"):
        prepare.prepare(other, tmp_path / "o", settings())
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "childes.train.txt").write_text("")
    with pytest.raises(ValueError, match="corpus"):
        prepare.prepare(empty, tmp_path / "o", settings())
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "childes.train.txt").write_bytes(b"*MOT:\t\xff\xfe bad\n" * 50)
    with pytest.raises(UnicodeDecodeError):
        prepare.prepare(bad, tmp_path / "o", settings())


@pytest.mark.parametrize("field", ["dev_block_lines", "dev_every_blocks"])
def test_non_positive_dev_settings_rejected(tmp_path: pathlib.Path, field: str):
    with pytest.raises(ValueError, match=field):
        prepare.prepare(
            write_corpus(tmp_path / "c"), tmp_path / "o", settings(**{field: 0})
        )
