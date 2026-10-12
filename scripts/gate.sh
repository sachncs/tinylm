#!/bin/sh
# Quality gate: every check must pass before a commit.
set -e
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q --cov=tinylm --cov-report=term-missing:skip-covered --cov-fail-under=97
