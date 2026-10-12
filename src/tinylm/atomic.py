"""Atomic directory creation: all of it appears at once, or none of it."""

import pathlib
import shutil
import types


class Staging:
    """Context manager that builds a directory next to its final place.

    The body writes into a sibling `<name>.partial` folder. On success the
    folder is renamed to the target in one step; on any exception it is
    removed and the original exception propagates. Failures while cleaning
    up are never swallowed: they surface chained to the original error.
    """

    def __init__(self, target: pathlib.Path) -> None:
        self.target = target
        self.partial = target.with_name(target.name + ".partial")

    def __enter__(self) -> pathlib.Path:
        """Creates a fresh partial folder; refuses an existing target.

        Raises:
            FileExistsError: If the target already exists.
        """
        if self.target.exists():
            raise FileExistsError(self.target)
        if self.partial.exists():
            shutil.rmtree(self.partial)
        self.partial.mkdir(parents=True)
        return self.partial

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: types.TracebackType | None,
    ) -> None:
        """Commits on success, removes the partial folder on failure."""
        if kind is None:
            self.partial.rename(self.target)
        else:
            shutil.rmtree(self.partial)
