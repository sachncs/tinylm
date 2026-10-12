"""Guarantees that a block of code runs without touching the network."""

import contextlib
import os
import socket
import threading
import types
from unittest import mock

FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
ON = "1"
ACTIVE = threading.Lock()


class Error(Exception):
    """Raised when code inside Offline tries to open a network connection."""


def refuse(*_args: object, **_kwargs: object) -> None:
    """Replacement for socket connect calls inside Offline."""
    raise Error("network access attempted while Offline is active")


class Offline:
    """Context manager that disables the network for its body.

    Sets the Hugging Face offline flags and replaces socket connection calls
    with a function that raises Error. Everything is restored on exit, also
    when the body raises. It patches process-global functions, so it cannot be
    nested or used from two threads at once: a second entry raises Error
    instead of silently restoring the wrong state.
    """

    def __init__(self) -> None:
        self.saved: dict[str, str | None] = {}
        self.stack = contextlib.ExitStack()

    def __enter__(self) -> Offline:
        """Switches the offline flags on and blocks sockets.

        Raises:
            Error: If another Offline block is already active.
        """
        if not ACTIVE.acquire(blocking=False):
            raise Error("Offline is already active; it patches global state")
        try:
            for flag in FLAGS:
                self.saved[flag] = os.environ.get(flag)
                os.environ[flag] = ON
            self.stack.enter_context(
                mock.patch.object(socket.socket, "connect", refuse)
            )
            self.stack.enter_context(
                mock.patch.object(socket, "create_connection", refuse)
            )
        except BaseException:
            self.restore()
            raise
        return self

    def restore(self) -> None:
        """Undoes the patches and flags, then releases the lock."""
        self.stack.close()
        for flag, old in self.saved.items():
            if old is None:
                os.environ.pop(flag, None)
            else:
                os.environ[flag] = old
        ACTIVE.release()

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: types.TracebackType | None,
    ) -> None:
        """Restores the environment and the socket functions."""
        self.restore()
