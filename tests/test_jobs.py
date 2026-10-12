"""Tests for running experiments as isolated, loud subprocesses."""

import pathlib
import subprocess
import sys

import pytest

from tinylm import jobs


def test_command_names_the_run_and_passes_every_override_in_order():
    command = jobs.command(pathlib.Path("c.toml"), "x01", ("a.b=1", "c.d=2"))
    assert command == [
        sys.executable,
        "scripts/run.py",
        "c.toml",
        "--name",
        "x01",
        "--resume",
        "--set",
        "a.b=1",
        "--set",
        "c.d=2",
    ]
    assert jobs.command(pathlib.Path("c.toml"), "x", ())[-3:] == [
        "--name",
        "x",
        "--resume",
    ]


def test_run_all_executes_in_order_and_wraps_with_caffeinate_when_present(
    monkeypatch: pytest.MonkeyPatch,
):
    seen: list[list[str]] = []

    def fake(command: list[str], check: bool) -> None:
        assert check is True
        seen.append(command)

    monkeypatch.setattr(subprocess, "run", fake)

    def present(name: str) -> str | None:
        return "/usr/bin/caffeinate"

    def absent(name: str) -> str | None:
        return None

    monkeypatch.setattr(jobs.shutil, "which", present)
    jobs.run_all([["one"], ["two"]])
    assert seen == [
        ["/usr/bin/caffeinate", "-i", "one"],
        ["/usr/bin/caffeinate", "-i", "two"],
    ]
    seen.clear()
    monkeypatch.setattr(jobs.shutil, "which", absent)
    jobs.run_all([["one"]])
    assert seen == [["one"]]


def test_a_failing_job_stops_the_batch_and_raises_with_its_exit_status(
    tmp_path: pathlib.Path,
):
    marker = tmp_path / "ran-after-failure"
    ok = [sys.executable, "-c", "pass"]
    bad = [sys.executable, "-c", "raise SystemExit(3)"]
    never = [sys.executable, "-c", f"open({str(marker)!r}, 'w')"]
    with pytest.raises(subprocess.CalledProcessError) as caught:
        jobs.run_all([ok, bad, never])
    assert caught.value.returncode == 3
    assert not marker.exists()
