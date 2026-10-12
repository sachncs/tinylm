"""Tests for atomic directory creation that never swallows errors."""

import pathlib

import pytest

from tinylm.atomic import Staging


def test_success_renames_into_place_and_leaves_no_partial(
    tmp_path: pathlib.Path,
):
    target = tmp_path / "out"
    with Staging(target) as staged:
        (staged / "a.txt").write_text("x")
        assert not target.exists()
    assert (target / "a.txt").read_text() == "x"
    assert not list(tmp_path.glob("*.partial"))


def test_failure_removes_the_partial_and_reraises_the_original(
    tmp_path: pathlib.Path,
):
    target = tmp_path / "out"
    with pytest.raises(RuntimeError, match="boom"), Staging(target) as staged:
        (staged / "a.txt").write_text("x")
        raise RuntimeError("boom")
    assert not target.exists() and not list(tmp_path.glob("*.partial"))


def test_existing_target_is_refused_before_any_work(tmp_path: pathlib.Path):
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(FileExistsError), Staging(target):
        pytest.fail("body must not run")
    assert list(target.iterdir()) == []


def test_a_stale_partial_from_a_crash_is_replaced(tmp_path: pathlib.Path):
    stale = tmp_path / "out.partial"
    stale.mkdir()
    (stale / "old.txt").write_text("old")
    with Staging(tmp_path / "out") as staged:
        assert not (staged / "old.txt").exists()
    assert (tmp_path / "out").is_dir()


def test_missing_parent_is_created(tmp_path: pathlib.Path):
    with Staging(tmp_path / "deep" / "er" / "out") as staged:
        (staged / "f").write_text("1")
    assert (tmp_path / "deep" / "er" / "out" / "f").read_text() == "1"
