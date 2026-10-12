"""Tests that the versioned commit gate really blocks bad commits."""

import os
import pathlib
import shutil
import subprocess

HOOKS = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "hooks"


def commit(repo: pathlib.Path, gate: str) -> subprocess.CompletedProcess[str]:
    (repo / "f.txt").write_text(gate)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    env = {**os.environ, "GATE": gate, "HOME": str(repo)}
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-m",
            "x",
        ],
        check=False,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def repo(tmp: pathlib.Path) -> pathlib.Path:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp, check=True)
    shutil.copytree(HOOKS, tmp / "hooks")
    subprocess.run(
        ["git", "config", "core.hooksPath", "hooks"], cwd=tmp, check=True
    )
    return tmp


def test_failing_gate_blocks_the_commit_and_says_why(tmp_path: pathlib.Path):
    result = commit(repo(tmp_path), "false")
    assert result.returncode != 0
    assert "commit blocked" in result.stderr
    log = subprocess.run(
        ["git", "log", "--oneline"],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert log.returncode != 0 or log.stdout.strip() == ""


def test_passing_gate_allows_the_commit_and_writes_the_external_backup(
    tmp_path: pathlib.Path,
):
    result = commit(repo(tmp_path), "true")
    assert result.returncode == 0, result.stderr
    assert (
        tmp_path / "Documents" / "Codex" / "backups" / "tinylm.bundle"
    ).is_file()
