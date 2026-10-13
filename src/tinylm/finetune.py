"""Runs the official GLUE fine-tuning pipeline and reads its results."""

import collections.abc
import dataclasses
import json
import os
import pathlib
import subprocess
import sys

MODULE = "evaluation_pipeline.finetune.run"
RESULTS = "results.txt"
Runner = collections.abc.Callable[[list[str], pathlib.Path], str]


class Error(Exception):
    """Raised when fine-tuning input or output is invalid."""


def execute(command: list[str], cwd: pathlib.Path) -> str:
    """Runs a command with Weights & Biases disabled; raises on failure."""
    env = {**os.environ, "WANDB_MODE": "disabled"}
    done = subprocess.run(
        command, cwd=cwd, check=True, capture_output=True, text=True, env=env
    )
    return done.stdout


@dataclasses.dataclass(frozen=True, slots=True)
class Task:
    """One GLUE task.

    Attributes:
        name: File prefix and official task name.
        labels: Number of classes.
        metrics: Metrics to report.
    """

    name: str
    labels: int
    metrics: tuple[str, ...]


TASKS = (
    Task("boolq", 2, ("accuracy",)),
    Task("mnli", 3, ("accuracy",)),
    Task("mrpc", 2, ("accuracy", "f1")),
    Task("multirc", 2, ("accuracy",)),
    Task("qqp", 2, ("accuracy", "f1")),
    Task("rte", 2, ("accuracy",)),
    Task("wsc", 2, ("accuracy",)),
)


@dataclasses.dataclass(frozen=True, slots=True)
class Finetune:
    """A checkout of the official evaluator and the GLUE files.

    Attributes:
        root: The `strict` folder of babylm-org/babylm-eval.
        data: Folder holding `<task>.train.jsonl` and `<task>.valid.jsonl`.
        epochs: Fine-tuning epochs.
        batch: Batch size.
        length: Maximum sequence length before truncation.
        rate: Peak learning rate.
        seed: Seed of the fine-tuning run.
        runner: Runs a command in a directory and returns stdout; replaceable
            in tests.
    """

    root: pathlib.Path
    data: pathlib.Path
    epochs: int = 10
    batch: int = 32
    length: int = 256
    rate: float = 3e-5
    seed: int = 42
    runner: Runner = execute

    def command(
        self, model: pathlib.Path, task: Task, output: pathlib.Path
    ) -> list[str]:
        """Builds the command line for one task.

        Last-token pooling needs left padding; the exported model re-packs
        rows by their mask, so that is exact.
        """
        train = self.data / f"{task.name}.train.jsonl"
        valid = self.data / f"{task.name}.valid.jsonl"
        return [
            sys.executable, "-m", MODULE,
            "--model_name_or_path", str(model),
            "--train_data", str(train),
            "--valid_data", str(valid),
            "--task", task.name,
            "--num_labels", str(task.labels),
            "--metrics", *task.metrics,
            "--metric_for_valid", task.metrics[0],
            "--results_dir", str(output),
            "--padding_side", "left",
            "--sequence_length", str(self.length),
            "--batch_size", str(self.batch),
            "--num_epochs", str(self.epochs),
            "--learning_rate", str(self.rate),
            "--seed", str(self.seed),
            "--take_final",
        ]  # fmt: skip

    def parse(self, text: str) -> dict[str, float]:
        """Reads `name: value` lines.

        Raises:
            Error: If there is no metric line or a value is not a number in
                [0, 1].
        """
        found: dict[str, float] = {}
        for line in text.splitlines():
            name, colon, value = line.partition(":")
            if not colon:
                continue
            try:
                number = float(value)
            except ValueError as error:
                raise Error(
                    f"metric {name.strip()!r} is not a number"
                ) from error
            if not 0.0 <= number <= 1.0:
                raise Error(
                    f"metric {name.strip()!r} = {number} is out of range"
                )
            found[name.strip()] = number
        if not found:
            raise Error("no metrics in the results")
        return found

    def majority(self, task: Task) -> float:
        """Accuracy of always predicting the commonest validation label.

        Raises:
            Error: If the validation file is empty.
        """
        path = self.data / f"{task.name}.valid.jsonl"
        counts: dict[object, int] = {}
        for line in path.read_text().splitlines():
            label = json.loads(line)["label"]
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            raise Error(f"{path} is empty")
        return max(counts.values()) / sum(counts.values())

    def evaluate(
        self, model: pathlib.Path, task: Task, output: pathlib.Path
    ) -> dict[str, float]:
        """Fine-tunes on one task and returns its validation metrics.

        Raises:
            subprocess.CalledProcessError: If the pipeline fails.
            Error: If it wrote no usable results file.
        """
        self.runner(self.command(model, task, output), self.root)
        path = output / model.name / "main" / "finetune" / task.name / RESULTS
        if not path.is_file():
            raise Error(f"no results file at {path}")
        return self.parse(path.read_text())
