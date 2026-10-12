"""Tests for pinned, checksummed, idempotent dataset fetching."""

import pathlib

import pytest

from tinylm import manifest
from tinylm.fetch import Fetch, Pin

REVISION = "a" * 40


def pin(**overrides: object) -> Pin:
    values: dict[str, object] = {
        "name": "tiny",
        "repo": "org/tiny",
        "revision": REVISION,
        "patterns": ("*.txt",),
    }
    values.update(overrides)
    return Pin(**values)  # pyright: ignore[reportArgumentType]


class Fake:
    """Downloader that writes small files and counts its calls."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = (
            files
            if files is not None
            else {"a.txt": b"alpha\n", "sub/b.txt": b"beta\n"}
        )
        self.calls = 0

    def __call__(
        self,
        repo: str,
        revision: str,
        dest: pathlib.Path,
        patterns: tuple[str, ...],
    ) -> None:
        self.calls += 1
        for name, data in self.files.items():
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def test_fetch_downloads_verifies_and_writes_a_manifest(tmp_path: pathlib.Path):
    fake = Fake()
    fetch = Fetch(tmp_path / "data", tmp_path / "manifests", fake)
    fetch.run(pin())
    assert (
        tmp_path / "data" / "tiny" / "sub" / "b.txt"
    ).read_bytes() == b"beta\n"
    saved = manifest.load(tmp_path / "manifests" / "tiny.json")
    assert [e.name for e in saved.entries] == ["a.txt", "sub/b.txt"]
    manifest.check(tmp_path / "data" / "tiny", saved)


def test_second_run_is_a_no_op_when_the_data_verifies(tmp_path: pathlib.Path):
    fake = Fake()
    fetch = Fetch(tmp_path / "data", tmp_path / "manifests", fake)
    fetch.run(pin())
    fetch.run(pin())
    assert fake.calls == 1


def test_tampered_data_fails_loudly_and_is_not_silently_repaired(
    tmp_path: pathlib.Path,
):
    fake = Fake()
    fetch = Fetch(tmp_path / "data", tmp_path / "manifests", fake)
    fetch.run(pin())
    (tmp_path / "data" / "tiny" / "a.txt").write_bytes(b"EVIL\n")
    with pytest.raises(manifest.Error, match="sha256|size"):
        fetch.run(pin())
    assert fake.calls == 1


def test_failed_download_leaves_no_partial_data_or_manifest(
    tmp_path: pathlib.Path,
):
    def failing(*_args: object) -> None:
        raise ConnectionError("down")

    fetch = Fetch(tmp_path / "data", tmp_path / "manifests", failing)
    with pytest.raises(ConnectionError):
        fetch.run(pin())
    assert not (tmp_path / "data" / "tiny").exists()
    assert not list((tmp_path / "data").glob("*.partial"))
    assert not (tmp_path / "manifests" / "tiny.json").exists()


def test_empty_or_corrupt_downloads_are_rejected(tmp_path: pathlib.Path):
    with pytest.raises(manifest.Error, match="no files"):
        Fetch(tmp_path / "d1", tmp_path / "m", Fake({})).run(pin())
    with pytest.raises(manifest.Error, match="empty"):
        Fetch(tmp_path / "d2", tmp_path / "m", Fake({"a.txt": b""})).run(pin())
    assert not (tmp_path / "d1" / "tiny").exists()
    assert not (tmp_path / "d2" / "tiny").exists()


@pytest.mark.parametrize("revision", ["main", "abc123", "g" * 40, "A" * 40, ""])
def test_revision_must_be_a_full_lowercase_hex_commit(revision: str):
    with pytest.raises(ValueError, match="revision"):
        pin(revision=revision)


def test_pin_validates_other_fields():
    with pytest.raises(ValueError, match="name"):
        pin(name="../escape")
    with pytest.raises(ValueError, match="repo"):
        pin(repo="no-slash")
    with pytest.raises(ValueError, match="patterns"):
        pin(patterns=())
