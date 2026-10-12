"""Device selection that fails loudly instead of falling back silently."""

import dataclasses
import os

import torch

NAMES = ("cpu", "mps", "cuda")
FALLBACK = "PYTORCH_ENABLE_MPS_FALLBACK"


class Error(Exception):
    """Raised when the requested device cannot be used as asked."""


@dataclasses.dataclass(frozen=True, slots=True)
class Device:
    """A named compute device with explicit synchronisation.

    Attributes:
        name: "cpu", "mps" or "cuda".
    """

    name: str

    def resolve(self) -> torch.device:
        """Returns the torch device, or raises with the exact reason.

        Raises:
            Error: If the name is unknown, the accelerator is missing, or
                the MPS CPU-fallback switch is on (it would silently run
                unsupported operations on the CPU and hide the slowdown).
        """
        if self.name not in NAMES:
            raise Error(f"device must be one of {NAMES}, got {self.name!r}")
        if self.name == "mps":
            if not torch.backends.mps.is_built():
                raise Error("PyTorch was not built with MPS support")
            if not torch.backends.mps.is_available():
                raise Error("MPS is built but not available on this machine")
            if os.environ.get(FALLBACK, "0") not in ("", "0"):
                raise Error(
                    f"{FALLBACK} is set: refusing silent CPU fallback; unset it"
                )
        if self.name == "cuda" and not torch.cuda.is_available():
            raise Error("CUDA is not available on this machine")
        return torch.device(self.name)

    def sync(self) -> None:
        """Waits until all queued device work has finished."""
        if self.name == "mps":
            torch.mps.synchronize()
        elif self.name == "cuda":
            torch.cuda.synchronize()

    def release(self) -> None:
        """Returns cached device memory to the system."""
        if self.name == "mps":
            torch.mps.empty_cache()
        elif self.name == "cuda":
            torch.cuda.empty_cache()
