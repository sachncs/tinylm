"""Tests for corpus manifests and checksum verification."""

import hashlib
import json
import pathlib

import pytest

from tinylm import manifest


def write(path: pathlib.Path, data: bytes) -> pathlib.Path:
    path.write_bytes(data)
    return path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entry(path: pathlib.Path, data: bytes) -> manifest.Entry:
    return manifest.Entry(path.name, sha(data), len(data))


def test_digest_matches_hashlib_and_missing_file_raises(tmp_path: pathlib.Path):
    data = b"hello world\n" * 1000
    assert manifest.digest(write(tmp_path / "a.txt", data)) == sha(data)
    with pytest.raises(FileNotFoundError):
        manifest.digest(tmp_path / "missing.txt")


def test_verify_accepts_intact_file(tmp_path: pathlib.Path):
    path = write(tmp_path / "a.txt", b"abc\n")
    manifest.verify(path, entry(path, b"abc\n"))


@pytest.mark.parametrize(
    "data,expected,match",
    [
        (b"abc\n", manifest.Entry("a.txt", sha(b"other\n"), 4), "sha256"),
        (b"abcd", manifest.Entry("a.txt", sha(b"abcdefgh"), 8), "size"),
        (b"", manifest.Entry("a.txt", sha(b"x"), 1), "empty"),
        (
            b"\xff\xfe\xfa bad\n",
            manifest.Entry("a.txt", sha(b"\xff\xfe\xfa bad\n"), 8),
            "utf-8",
        ),
    ],
)
def test_verify_rejects_corruption(
    tmp_path: pathlib.Path, data: bytes, expected: manifest.Entry, match: str
):
    path = write(tmp_path / "a.txt", data)
    with pytest.raises(manifest.Error, match=match):
        manifest.verify(path, expected)


def test_build_save_load_verify_round_trip_is_sorted(tmp_path: pathlib.Path):
    write(tmp_path / "b.txt", b"three\n")
    write(tmp_path / "a.txt", b"one two\n")
    built = manifest.build(tmp_path, ["b.txt", "a.txt"])
    target = tmp_path / "manifest.json"
    manifest.save(built, target)
    assert [e["name"] for e in json.loads(target.read_text())["entries"]] == [
        "a.txt",
        "b.txt",
    ]
    loaded = manifest.load(target)
    assert loaded == built
    manifest.check(tmp_path, loaded)


def test_verify_directory_reports_missing_file(tmp_path: pathlib.Path):
    write(tmp_path / "a.txt", b"x\n")
    built = manifest.build(tmp_path, ["a.txt"])
    (tmp_path / "a.txt").unlink()
    with pytest.raises(manifest.Error, match="missing"):
        manifest.check(tmp_path, built)


@pytest.mark.parametrize(
    "body", [b"{not json", b'{"entries": [{"name": "a"}]}']
)
def test_load_rejects_malformed(tmp_path: pathlib.Path, body: bytes):
    with pytest.raises(manifest.Error, match="manifest"):
        manifest.load(write(tmp_path / "m.json", body))


def test_build_rejects_duplicates_and_traversal(tmp_path: pathlib.Path):
    write(tmp_path / "a.txt", b"x\n")
    with pytest.raises(ValueError, match="duplicate"):
        manifest.build(tmp_path, ["a.txt", "a.txt"])
    with pytest.raises(ValueError, match="relative"):
        manifest.build(tmp_path, ["../etc/passwd"])
