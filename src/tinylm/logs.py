"""Logging setup: one format, console plus an optional run log file."""

import logging
import pathlib
import sys

NAME = "tinylm"
FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def setup(level: str = "INFO", file: pathlib.Path | None = None) -> None:
    """Configures the library logger; calling it again replaces the handlers.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR.
        file: Optional path that also receives every record.

    Raises:
        ValueError: If level is not one of the supported names.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    reset()
    logger = logging.getLogger(NAME)
    logger.setLevel(level)
    logger.propagate = False
    formatter = logging.Formatter(FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if file is not None:
        file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def reset() -> None:
    """Removes and closes every handler installed by setup()."""
    logger = logging.getLogger(NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
