"""Where a result came from: code revision, environment and inputs."""

import dataclasses
import pathlib
import platform
import subprocess

import numpy
import torch


class Error(Exception):
    """Raised when provenance cannot be established; results need it."""


def git(cwd: pathlib.Path, *args: str) -> str:
    """Runs git and returns stdout.

    Raises:
        Error: If git is missing, fails, or cwd is not a repository.
    """
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise Error(
            f"git {' '.join(args)} failed in {cwd}: {detail.strip()}"
        ) from error
    return done.stdout


@dataclasses.dataclass(frozen=True, slots=True)
class Provenance:
    """Everything needed to say which code and inputs produced a number.

    Attributes:
        revision: Full git commit hash of the code.
        dirty: True if the working tree differed from that commit.
        python: Interpreter version.
        torch: PyTorch version.
        numpy: NumPy version.
        platform: Operating system and CPU description.
        device: Compute device name.
        config: Fingerprint of the experiment configuration.
        data: Fingerprint of the training data.
    """

    revision: str
    dirty: bool
    python: str
    torch: str
    numpy: str
    platform: str
    device: str
    config: str
    data: str

    @classmethod
    def capture(
        cls, root: pathlib.Path, device: str, config: str, data: str
    ) -> Provenance:
        """Reads the current code revision and environment.

        Raises:
            Error: If root is not a git repository with at least one commit.
        """
        git(root, "rev-parse", "--is-inside-work-tree")
        try:
            revision = git(root, "rev-parse", "HEAD").strip()
        except Error as error:
            raise Error(
                f"cannot read the revision of {root}: {error}"
            ) from error
        dirty = bool(git(root, "status", "--porcelain").strip())
        return cls(
            revision=revision,
            dirty=dirty,
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=numpy.__version__,
            platform=platform.platform(),
            device=device,
            config=config,
            data=data,
        )
