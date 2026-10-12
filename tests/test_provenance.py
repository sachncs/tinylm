"""Tests for tracing a result back to its code and environment."""

import pathlib
import subprocess

import pytest
import torch

from tinylm.provenance import Error, Provenance


def git(cwd: pathlib.Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def repo(tmp: pathlib.Path) -> pathlib.Path:
    git(tmp, "init", "-q", "-b", "main")
    (tmp / "a.txt").write_text("one")
    git(tmp, "add", "-A")
    git(tmp, "commit", "-q", "-m", "first")
    return tmp


def test_capture_records_revision_versions_device_and_fingerprints(
    tmp_path: pathlib.Path,
):
    found = Provenance.capture(
        repo(tmp_path), device="cpu", config="c" * 64, data="d" * 64
    )
    assert len(found.revision) == 40 and not found.dirty
    assert found.torch == torch.__version__
    assert found.python.startswith("3.14")
    assert (
        found.device == "cpu"
        and found.config == "c" * 64
        and found.data == "d" * 64
    )


def test_dirty_working_tree_is_flagged(tmp_path: pathlib.Path):
    repo(tmp_path)
    (tmp_path / "a.txt").write_text("changed")
    assert Provenance.capture(tmp_path, "cpu", "c", "d").dirty
    (tmp_path / "untracked.txt").write_text("x")
    git(tmp_path, "checkout", "--", "a.txt")
    assert Provenance.capture(tmp_path, "cpu", "c", "d").dirty


def test_outside_a_git_repository_fails_loudly(tmp_path: pathlib.Path):
    with pytest.raises(Error, match="git"):
        Provenance.capture(tmp_path, "cpu", "c", "d")


def test_repository_without_commits_fails_loudly(tmp_path: pathlib.Path):
    git(tmp_path, "init", "-q", "-b", "main")
    with pytest.raises(Error, match="revision"):
        Provenance.capture(tmp_path, "cpu", "c", "d")
