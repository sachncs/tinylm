"""Tests for the GLUE fine-tuning wrapper."""

import json
import pathlib
import subprocess
import sys

import pytest

from tinylm.finetune import Error, Finetune, Runner, Task


def quiet(command: list[str], cwd: pathlib.Path) -> str:
    return ""


def make(tmp_path: pathlib.Path, runner: Runner = quiet) -> Finetune:
    return Finetune(
        root=tmp_path / "strict", data=tmp_path / "glue", runner=runner
    )


def test_command_passes_every_setting_and_left_pads_for_last_token_pooling(
    tmp_path: pathlib.Path,
):
    command = make(tmp_path).command(
        tmp_path / "model", Task("mnli", 3, ("accuracy",)), tmp_path / "out"
    )
    assert command[:3] == [
        sys.executable,
        "-m",
        "evaluation_pipeline.finetune.run",
    ]
    pairs = dict(zip(command[3::2], command[4::2], strict=False))
    assert pairs["--task"] == "mnli"
    assert pairs["--num_labels"] == "3"
    assert pairs["--padding_side"] == "left"
    assert pairs["--train_data"].endswith("mnli.train.jsonl")
    assert pairs["--valid_data"].endswith("mnli.valid.jsonl")
    assert "--take_final" in command


def test_parse_reads_every_metric_line(tmp_path: pathlib.Path):
    assert make(tmp_path).parse("accuracy: 0.5\nf1: 0.25") == {
        "accuracy": 0.5,
        "f1": 0.25,
    }


@pytest.mark.parametrize(
    "text", ["", "accuracy: high", "accuracy: 1.5", "accuracy: -0.1", "nothing"]
)
def test_parse_rejects_missing_or_implausible_results(
    tmp_path: pathlib.Path, text: str
):
    with pytest.raises(Error):
        make(tmp_path).parse(text)


def test_majority_baseline_is_the_share_of_the_commonest_label(
    tmp_path: pathlib.Path,
):
    (tmp_path / "glue").mkdir()
    rows = [{"label": 1}] * 3 + [{"label": 0}]
    (tmp_path / "glue" / "rte.valid.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    assert make(tmp_path).majority(Task("rte", 2, ("accuracy",))) == 0.75


def test_majority_baseline_of_an_empty_file_raises(tmp_path: pathlib.Path):
    (tmp_path / "glue").mkdir()
    (tmp_path / "glue" / "rte.valid.jsonl").write_text("")
    with pytest.raises(Error, match="empty"):
        make(tmp_path).majority(Task("rte", 2, ("accuracy",)))


def test_evaluate_runs_the_task_and_reads_its_results_file(
    tmp_path: pathlib.Path,
):
    seen: list[list[str]] = []

    def runner(command: list[str], cwd: pathlib.Path) -> str:
        seen.append(command)
        results = tmp_path / "out" / "model" / "main" / "finetune" / "wsc"
        results.mkdir(parents=True)
        (results / "results.txt").write_text("accuracy: 0.6")
        return ""

    score = make(tmp_path, runner).evaluate(
        tmp_path / "model", Task("wsc", 2, ("accuracy",)), tmp_path / "out"
    )
    assert score == {"accuracy": 0.6}
    assert len(seen) == 1


def test_a_missing_results_file_fails_loudly(tmp_path: pathlib.Path):
    with pytest.raises(Error, match="results"):
        make(tmp_path).evaluate(
            tmp_path / "model", Task("wsc", 2, ("accuracy",)), tmp_path / "out"
        )


def test_a_failing_process_raises_instead_of_returning_a_score(
    tmp_path: pathlib.Path,
):
    def runner(command: list[str], cwd: pathlib.Path) -> str:
        raise subprocess.CalledProcessError(2, command)

    with pytest.raises(subprocess.CalledProcessError):
        make(tmp_path, runner).evaluate(
            tmp_path / "model", Task("wsc", 2, ("accuracy",)), tmp_path / "out"
        )
