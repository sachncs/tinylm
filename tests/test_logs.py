"""Tests for logging setup and the messages the library emits."""

import logging
import pathlib

import pytest

from tinylm import logs


@pytest.fixture(autouse=True)
def clean_logger():
    yield
    logs.reset()


def test_setup_writes_to_the_file_with_time_level_and_name(
    tmp_path: pathlib.Path,
):
    target = tmp_path / "run.log"
    logs.setup(file=target)
    logging.getLogger("tinylm.test").info("hello %d", 7)
    logging.getLogger("tinylm.test").warning("careful")
    text = target.read_text()
    assert "INFO tinylm.test hello 7" in text
    assert "WARNING tinylm.test careful" in text
    assert text.splitlines()[0][:4].isdigit()


def test_setup_is_idempotent_and_reset_removes_its_handlers(
    tmp_path: pathlib.Path,
):
    logs.setup(file=tmp_path / "a.log")
    logs.setup(file=tmp_path / "a.log")
    logger = logging.getLogger("tinylm")
    assert len(logger.handlers) == 2
    logs.reset()
    assert logger.handlers == []


def test_invalid_level_fails_loudly(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="level"):
        logs.setup(level="LOUD")


def test_errors_inside_handlers_are_not_swallowed(tmp_path: pathlib.Path):
    assert logging.raiseExceptions is True
    logs.setup(file=tmp_path / "x.log")
    assert logging.getLogger("tinylm").propagate is False
