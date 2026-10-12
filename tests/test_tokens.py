"""Tests for the memory-mapped token store."""

import json
import pathlib

import numpy
import pytest
import torch

from tinylm.tokens import Error, Token


def ids(n: int = 1000, vocab: int = 8192) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(0)
    return torch.randint(0, vocab, (n,), generator=generator)


def test_round_trip_preserves_every_token(tmp_path: pathlib.Path):
    data = ids()
    Token.from_tensor(data, vocab=8192).store(tmp_path / "s")
    loaded = Token.load(tmp_path / "s")
    assert len(loaded) == 1000 and loaded.vocab == 8192
    assert torch.equal(
        loaded.rows(torch.arange(0, 1000, 10), 10).reshape(-1), data
    )


def test_two_bytes_per_token_when_vocab_fits_four_otherwise(
    tmp_path: pathlib.Path,
):
    Token.from_tensor(ids(1000, 8192), vocab=8192).store(tmp_path / "small")
    assert (tmp_path / "small" / "tokens.bin").stat().st_size == 2000
    Token.from_tensor(ids(1000, 70000), vocab=70000).store(tmp_path / "big")
    assert (tmp_path / "big" / "tokens.bin").stat().st_size == 4000
    assert len(Token.load(tmp_path / "big")) == 1000


def test_rows_gathers_windows_at_arbitrary_starts(tmp_path: pathlib.Path):
    data = ids()
    Token.from_tensor(data, vocab=8192).store(tmp_path / "s")
    loaded = Token.load(tmp_path / "s")
    starts = torch.tensor([990, 0, 500])
    rows = loaded.rows(starts, 10)
    assert rows.dtype == torch.long and rows.shape == (3, 10)
    for row, start in zip(rows, [int(s) for s in starts], strict=True):
        assert torch.equal(row, data[start : start + 10])


def test_loaded_store_is_read_only(tmp_path: pathlib.Path):
    Token.from_tensor(ids(), vocab=8192).store(tmp_path / "s")
    with pytest.raises(ValueError, match="read-only"):
        Token.load(tmp_path / "s").ids[0] = 1


def test_in_memory_and_disk_stores_agree(tmp_path: pathlib.Path):
    data = ids()
    memory = Token.from_tensor(data, vocab=8192)
    memory.store(tmp_path / "s")
    starts = torch.tensor([3, 40, 700])
    assert torch.equal(
        memory.rows(starts, 7), Token.load(tmp_path / "s").rows(starts, 7)
    )


def test_store_is_byte_deterministic_and_refuses_to_overwrite(
    tmp_path: pathlib.Path,
):
    token = Token.from_tensor(ids(), vocab=8192)
    token.store(tmp_path / "a")
    token.store(tmp_path / "b")
    for name in ("tokens.bin", "meta.json"):
        assert (tmp_path / "a" / name).read_bytes() == (
            tmp_path / "b" / name
        ).read_bytes()
    with pytest.raises(FileExistsError):
        token.store(tmp_path / "a")
    assert not list(tmp_path.glob("*.partial"))


def test_meta_records_version_dtype_length_vocab_and_hash(
    tmp_path: pathlib.Path,
):
    Token.from_tensor(ids(), vocab=8192).store(tmp_path / "s")
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert meta["version"] == 1 and meta["dtype"] == "uint16"
    assert meta["length"] == 1000 and meta["vocab"] == 8192
    assert len(meta["sha256"]) == 64


def test_verify_detects_corruption_and_truncation(tmp_path: pathlib.Path):
    Token.from_tensor(ids(), vocab=8192).store(tmp_path / "s")
    Token.load(tmp_path / "s", verify=True)
    binary = tmp_path / "s" / "tokens.bin"
    raw = bytearray(binary.read_bytes())
    raw[10] ^= 0xFF
    binary.write_bytes(bytes(raw))
    Token.load(tmp_path / "s")
    with pytest.raises(Error, match="sha256"):
        Token.load(tmp_path / "s", verify=True)
    binary.write_bytes(bytes(raw[:-4]))
    with pytest.raises(Error, match="size"):
        Token.load(tmp_path / "s")


def test_construction_rejects_bad_ids():
    with pytest.raises(ValueError, match="vocab"):
        Token.from_tensor(torch.tensor([1, 9000]), vocab=8192)
    with pytest.raises(ValueError, match="negative"):
        Token.from_tensor(torch.tensor([1, -2]), vocab=8192)
    with pytest.raises(ValueError, match="1-D"):
        Token.from_tensor(torch.zeros(2, 2, dtype=torch.long), vocab=8192)
    with pytest.raises(ValueError, match="empty"):
        Token.from_tensor(torch.zeros(0, dtype=torch.long), vocab=8192)
    with pytest.raises(ValueError, match="vocab"):
        Token.from_tensor(ids(), vocab=0)
    with pytest.raises(ValueError, match="integer"):
        Token(numpy.zeros(4, dtype=numpy.float32), vocab=8192)


def test_load_rejects_missing_or_malformed(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        Token.load(tmp_path / "missing")
    Token.from_tensor(ids(), vocab=8192).store(tmp_path / "s")
    meta = tmp_path / "s" / "meta.json"
    meta.write_text("{not json")
    with pytest.raises(Error, match="meta"):
        Token.load(tmp_path / "s")
    meta.write_text(json.dumps({"version": 1, "dtype": "uint16"}))
    with pytest.raises(Error, match="meta"):
        Token.load(tmp_path / "s")
    meta.write_text(
        json.dumps(
            {
                "version": 9,
                "dtype": "uint16",
                "length": 1000,
                "vocab": 8192,
                "sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(Error, match="version"):
        Token.load(tmp_path / "s")


def test_rows_validate_arguments():
    token = Token.from_tensor(ids(), vocab=8192)
    with pytest.raises(ValueError, match="width"):
        token.rows(torch.tensor([0]), 0)
    for start in (995, -1):
        with pytest.raises(IndexError, match="range"):
            token.rows(torch.tensor([start]), 10)
