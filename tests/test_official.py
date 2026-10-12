"""Tests for the wrapper around the official evaluation pipeline."""

import pathlib
import subprocess

import pytest

from tinylm.official import Error, Official, Score

SAMPLE = """\
loading model...
quantifiers: 69.75
argument_structure: 67.86
subject_verb_agreement: 67.00

### AVERAGE ACCURACY
66.99
"""


def test_parse_reads_groups_and_the_average():
    score = Official(pathlib.Path("x")).parse("blimp", SAMPLE)
    assert score == Score(
        "blimp",
        66.99,
        {
            "quantifiers": 69.75,
            "argument_structure": 67.86,
            "subject_verb_agreement": 67.00,
        },
    )


@pytest.mark.parametrize(
    "text,match",
    [
        ("no report here", "average"),
        ("### AVERAGE ACCURACY\n", "average"),
        ("### AVERAGE ACCURACY\nhigh\n", "average"),
        ("### AVERAGE ACCURACY\n150.0\n", "range"),
        ("", "average"),
    ],
)
def test_parse_rejects_missing_or_implausible_reports(text: str, match: str):
    with pytest.raises(Error, match=match):
        Official(pathlib.Path("x")).parse("blimp", text)


def test_command_targets_the_official_module_with_every_argument():
    command = Official(pathlib.Path("/e")).command(
        pathlib.Path("/m"), "blimp", pathlib.Path("/d"), pathlib.Path("/o")
    )
    assert command[1:3] == ["-m", "evaluation_pipeline.sentence_zero_shot.run"]
    pairs = dict(zip(command[3::2], command[4::2], strict=False))
    assert pairs["--model_path_or_name"] == "/m" and pairs["--task"] == "blimp"
    assert pairs["--data_path"] == "/d" and pairs["--backend"] == "causal"
    assert pairs["--output_dir"] == "/o"


def test_evaluate_uses_the_runner_and_parses_its_output(tmp_path: pathlib.Path):
    seen: list[tuple[list[str], pathlib.Path]] = []

    def runner(command: list[str], cwd: pathlib.Path) -> str:
        seen.append((command, cwd))
        return SAMPLE

    official = Official(tmp_path, runner=runner)
    score = official.evaluate(
        tmp_path / "m", "blimp", tmp_path / "d", tmp_path / "o"
    )
    assert score.average == 66.99 and seen[0][1] == tmp_path


def test_a_failing_official_process_raises_instead_of_returning_a_number(
    tmp_path: pathlib.Path,
):
    with pytest.raises(subprocess.CalledProcessError):
        Official(tmp_path).evaluate(
            tmp_path / "m", "blimp", tmp_path / "d", tmp_path / "o"
        )


def test_unknown_task_and_backend_are_rejected():
    official = Official(pathlib.Path("x"))
    with pytest.raises(Error, match="task"):
        official.command(
            pathlib.Path("m"), "vibes", pathlib.Path("d"), pathlib.Path("o")
        )
    with pytest.raises(Error, match="backend"):
        official.command(
            pathlib.Path("m"),
            "blimp",
            pathlib.Path("d"),
            pathlib.Path("o"),
            "magic",
        )
