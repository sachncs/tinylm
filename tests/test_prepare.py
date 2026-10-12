"""End-to-end tests for corpus preparation."""

import json
import pathlib

import pytest

from tinylm import tokenizer
from tinylm.clean import Options
from tinylm.prepare import Prepare
from tinylm.tokens import Token

pytestmark = pytest.mark.e2e


def corpus(root: pathlib.Path) -> pathlib.Path:
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


def prepare(**overrides: object) -> Prepare:
    values: dict[str, object] = {
        "vocab": 300,
        "kind": "bpe",
        "block": 20,
        "every": 10,
        "options": Options(),
    }
    values.update(overrides)
    return Prepare(**values)  # pyright: ignore[reportArgumentType]


def test_outputs_counts_bos_cleaning_and_lineage(tmp_path: pathlib.Path):
    out = tmp_path / "out"
    stats = prepare().run(corpus(tmp_path / "c"), out)
    for name in (
        "tokenizer.json",
        "stats.json",
        "train/tokens.bin",
        "dev/tokens.bin",
    ):
        assert (out / name).is_file()
    train, dev = Token.load(out / "train"), Token.load(out / "dev")
    tok = tokenizer.load(out / "tokenizer.json")
    assert len(train) == stats.train_tokens > 5 * stats.dev_tokens > 0
    assert len(dev) == stats.dev_tokens and stats.dev_chars > stats.dev_tokens
    assert train.vocab == tok.get_vocab_size()
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    ids = train.ids
    assert int((ids == bos).sum()) > 100 and int((ids == eos).sum()) == 0
    text = tok.decode([int(i) for i in ids[:2000]])
    assert "*MOT" not in text and "WHO IS" not in text and "“" not in text
    saved = json.loads((out / "stats.json").read_text())
    assert saved["version"] == 1 and saved["train_tokens"] == stats.train_tokens
    assert {item["name"] for item in saved["inputs"]} == {
        "childes.train.txt",
        "open_subtitles.train.txt",
        "gutenberg.train.txt",
    }
    assert all(len(item["sha256"]) == 64 for item in saved["inputs"])


def test_run_is_byte_deterministic_and_refuses_existing_output(
    tmp_path: pathlib.Path,
):
    source = corpus(tmp_path / "c")
    prepare().run(source, tmp_path / "a")
    prepare().run(source, tmp_path / "b")
    for name in (
        "tokenizer.json",
        "stats.json",
        "train/tokens.bin",
        "train/meta.json",
        "dev/tokens.bin",
    ):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()
    with pytest.raises(FileExistsError):
        prepare().run(source, tmp_path / "a")
    assert not list(tmp_path.glob("*.partial"))


def test_failed_run_leaves_no_output(tmp_path: pathlib.Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "childes.train.txt").write_text("")
    with pytest.raises(ValueError, match="corpus"):
        prepare().run(empty, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_negative_inputs(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        prepare().run(tmp_path / "nope", tmp_path / "o")
    other = tmp_path / "other"
    other.mkdir()
    (other / "tweets.train.txt").write_text("hi\n")
    with pytest.raises(ValueError, match="source"):
        prepare().run(other, tmp_path / "o")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "childes.train.txt").write_bytes(b"*MOT:\t\xff\xfe bad\n" * 50)
    with pytest.raises(UnicodeDecodeError):
        prepare().run(bad, tmp_path / "o")


@pytest.mark.parametrize("field", ["block", "every"])
def test_non_positive_split_settings_rejected(field: str):
    with pytest.raises(ValueError, match=field):
        prepare(**{field: 0})
