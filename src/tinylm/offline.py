"""Guarantees that a block of code runs without touching the network."""

import os
import socket
import types

FLAGS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
ON = "1"


class Blocked(Exception):
    """Raised when code inside Offline tries to open a network connection."""


def refuse(*_args: object, **_kwargs: object) -> None:
    """Replacement for socket connect calls inside Offline."""
    raise Blocked("network access attempted while Offline is active")


class Offline:
    """Context manager that disables the network for its body.

    Sets the Hugging Face offline flags and replaces socket connection calls
    with a function that raises Blocked. Everything is restored on exit, also
    when the body raises.
    """

    def __init__(self) -> None:
        self.saved_env: dict[str, str | None] = {}
        self.saved_connect: object = None
        self.saved_create: object = None

    def __enter__(self) -> Offline:
        """Switches the offline flags on and blocks sockets."""
        for flag in FLAGS:
            self.saved_env[flag] = os.environ.get(flag)
            os.environ[flag] = ON
        self.saved_connect = socket.socket.connect
        self.saved_create = socket.create_connection
        socket.socket.connect = refuse  # pyright: ignore[reportAttributeAccessIssue]
        socket.create_connection = refuse  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        trace: types.TracebackType | None,
    ) -> None:
        """Restores the environment and the socket functions."""
        socket.socket.connect = self.saved_connect  # pyright: ignore[reportAttributeAccessIssue]
        socket.create_connection = self.saved_create  # pyright: ignore[reportAttributeAccessIssue]
        for flag, old in self.saved_env.items():
            if old is None:
                os.environ.pop(flag, None)
            else:
                os.environ[flag] = old
