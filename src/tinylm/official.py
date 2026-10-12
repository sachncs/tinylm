"""Runs the official BabyLM zero-shot pipeline and parses its report."""

import collections.abc
import dataclasses
import pathlib
import subprocess
import sys

TASKS = ("blimp", "ewok", "entity_tracking")
BACKENDS = ("causal", "mlm", "mntp")
MARKER = "### AVERAGE ACCURACY"
MODULE = "evaluation_pipeline.sentence_zero_shot.run"
PERCENT = 100.0

Runner = collections.abc.Callable[[list[str], pathlib.Path], str]


class Error(Exception):
    """Raised when the official pipeline's input or report is invalid."""


def execute(command: list[str], cwd: pathlib.Path) -> str:
    """Runs a command and returns its stdout, raising on a non-zero exit."""
    done = subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True
    )
    return done.stdout


@dataclasses.dataclass(frozen=True, slots=True)
class Score:
    """An official score.

    Attributes:
        task: Official task name.
        average: Average accuracy in percent, as the pipeline reports it.
        groups: Accuracy in percent per sub-group.
    """

    task: str
    average: float
    groups: dict[str, float]


@dataclasses.dataclass(frozen=True, slots=True)
class Official:
    """A checkout of the official evaluator.

    Attributes:
        root: The `strict` folder of babylm-org/babylm-eval.
        runner: Callable that runs a command in a directory and returns
            stdout; replaceable in tests.
    """

    root: pathlib.Path
    runner: Runner = execute

    def command(
        self,
        model: pathlib.Path,
        task: str,
        data: pathlib.Path,
        output: pathlib.Path,
        backend: str = "causal",
    ) -> list[str]:
        """Builds the command line of one official evaluation.

        Raises:
            Error: If the task or backend is not supported.
        """
        if task not in TASKS:
            raise Error(f"task must be one of {TASKS}, got {task!r}")
        if backend not in BACKENDS:
            raise Error(f"backend must be one of {BACKENDS}, got {backend!r}")
        return [
            sys.executable, "-m", MODULE,
            "--model_path_or_name", str(model),
            "--backend", backend,
            "--task", task,
            "--data_path", str(data),
            "--output_dir", str(output),
        ]  # fmt: skip

    def parse(self, task: str, text: str) -> Score:
        """Reads the report the pipeline prints.

        Raises:
            Error: If the average is missing, not a number or out of range.
        """
        lines = text.splitlines()
        if MARKER not in lines:
            raise Error("no average accuracy in the official report")
        after = lines[lines.index(MARKER) + 1 :]
        try:
            average = float(after[0])
        except (IndexError, ValueError) as error:
            raise Error(
                "average accuracy is missing or not a number"
            ) from error
        if not 0.0 <= average <= PERCENT:
            raise Error(f"average accuracy {average} is out of range")
        groups: dict[str, float] = {}
        for line in lines[: lines.index(MARKER)]:
            name, colon, value = line.partition(":")
            if colon:
                try:
                    groups[name.strip()] = float(value)
                except ValueError:
                    continue
        return Score(task, average, groups)

    def evaluate(
        self,
        model: pathlib.Path,
        task: str,
        data: pathlib.Path,
        output: pathlib.Path,
        backend: str = "causal",
    ) -> Score:
        """Runs the official pipeline on an exported model.

        Raises:
            Error: If arguments or the report are invalid.
            subprocess.CalledProcessError: If the pipeline itself fails.
        """
        text = self.runner(
            self.command(model, task, data, output, backend), self.root
        )
        return self.parse(task, text)
