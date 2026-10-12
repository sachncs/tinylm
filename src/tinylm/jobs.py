"""Runs experiments as separate processes that fail loudly."""

import collections.abc
import pathlib
import shutil
import subprocess
import sys


def command(base: pathlib.Path, name: str, sets: tuple[str, ...]) -> list[str]:
    """The command line that runs one named variant of a base experiment.

    `--resume` makes a re-run after a crash continue from the checkpoint
    when there is one and start fresh when there is none.
    """
    line = [
        sys.executable,
        "scripts/run.py",
        str(base),
        "--name",
        name,
        "--resume",
    ]
    for override in sets:
        line += ["--set", override]
    return line


def run_all(commands: collections.abc.Iterable[list[str]]) -> None:
    """Runs commands one after another, each in its own process.

    Separate processes keep memory use and crashes of one run from touching
    the next. On macOS each command is wrapped in `caffeinate -i` so the
    machine cannot idle-sleep mid-run. The first failing command raises
    CalledProcessError and the rest are not started.
    """
    keep_awake = shutil.which("caffeinate")
    for line in commands:
        subprocess.run(
            [keep_awake, "-i", *line] if keep_awake else line, check=True
        )
