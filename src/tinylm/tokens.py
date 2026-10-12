"""Compact, memory-mapped token storage that needs no network."""

import dataclasses
import json
import pathlib
import shutil

import numpy
import torch

from tinylm import manifest

VERSION = 1
BIN_NAME = "tokens.bin"
META_NAME = "meta.json"
SMALL_VOCAB = 1 << 16
META_KEYS = ("version", "dtype", "length", "vocab", "sha256")


class Error(Exception):
    """Raised when a stored token file is missing parts, corrupt or foreign."""


def dtype_for(vocab: int) -> type[numpy.unsignedinteger]:
    """Returns the smallest unsigned dtype that can hold ids below vocab."""
    return numpy.uint16 if vocab <= SMALL_VOCAB else numpy.uint32


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Token:
    """A 1-D stream of token ids and the vocabulary size they come from.

    Attributes:
        ids: Unsigned integer array of token ids; may be a read-only memmap.
        vocab: Vocabulary size; every id is below it.
        check: Scan ids for range errors on construction. Loading from disk
            skips the scan (use verify=True there for a checksum instead).
    """

    ids: numpy.ndarray
    vocab: int
    check: dataclasses.InitVar[bool] = True

    def __post_init__(self, check: bool) -> None:
        """Validates shape, dtype, vocabulary and (optionally) id range."""
        if self.vocab <= 0:
            raise ValueError("vocab must be positive")
        if self.ids.ndim != 1:
            raise ValueError("ids must be 1-D")
        if self.ids.size == 0:
            raise ValueError("ids must not be empty")
        if self.ids.dtype.kind not in "ui":
            raise ValueError("ids must have an integer dtype")
        if check:
            if int(self.ids.min()) < 0:
                raise ValueError("ids contain negative values")
            if int(self.ids.max()) >= self.vocab:
                raise ValueError("ids contain values outside vocab")

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, vocab: int) -> Token:
        """Builds a Token from a 1-D integer tensor.

        Raises:
            ValueError: If the tensor is not 1-D or empty, has negative ids
                or ids outside vocab.
        """
        if tensor.ndim != 1:
            raise ValueError("ids must be 1-D")
        if tensor.numel() == 0:
            raise ValueError("ids must not be empty")
        if vocab <= 0:
            raise ValueError("vocab must be positive")
        if int(tensor.min()) < 0:
            raise ValueError("ids contain negative values")
        if int(tensor.max()) >= vocab:
            raise ValueError("ids contain values outside vocab")
        return cls(tensor.numpy().astype(dtype_for(vocab)), vocab)

    def __len__(self) -> int:
        """Number of tokens in the stream."""
        return int(self.ids.size)

    def rows(self, starts: torch.Tensor, width: int) -> torch.Tensor:
        """Gathers windows ids[start : start + width] as a long tensor.

        Only the requested windows are copied; the stream is never copied.

        Raises:
            ValueError: If width is not positive.
            IndexError: If any window leaves the stream.
        """
        if width <= 0:
            raise ValueError("width must be positive")
        first = starts.to(torch.int64).numpy()
        if first.size and (first.min() < 0 or first.max() + width > len(self)):
            raise IndexError("a window is out of range")
        index = first[:, None] + numpy.arange(width)[None, :]
        return torch.from_numpy(self.ids[index].astype(numpy.int64))

    def store(self, directory: pathlib.Path) -> None:
        """Writes the stream atomically as tokens.bin plus meta.json.

        Raises:
            FileExistsError: If directory already exists.
        """
        if directory.exists():
            raise FileExistsError(directory)
        staging = directory.with_name(directory.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            array = numpy.ascontiguousarray(
                self.ids, dtype=dtype_for(self.vocab)
            )
            (staging / BIN_NAME).write_bytes(array.tobytes())
            meta = {
                "version": VERSION,
                "dtype": array.dtype.name,
                "length": int(array.size),
                "vocab": self.vocab,
                "sha256": manifest.digest(staging / BIN_NAME),
            }
            (staging / META_NAME).write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n"
            )
            staging.rename(directory)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @classmethod
    def load(cls, directory: pathlib.Path, verify: bool = False) -> Token:
        """Opens a stored stream as a read-only memory map.

        Args:
            directory: Folder written by store().
            verify: Recompute the SHA-256 and compare it with the metadata.

        Raises:
            FileNotFoundError: If the directory or its files are missing.
            Error: If the metadata is malformed or of another version, the
                file size disagrees with it, or verification fails.
        """
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        binary = directory / BIN_NAME
        meta_path = directory / META_NAME
        if not binary.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"{directory} lacks {BIN_NAME} or {META_NAME}"
            )
        try:
            meta = json.loads(meta_path.read_text())
            missing = [key for key in META_KEYS if key not in meta]
            dtype = numpy.dtype(meta["dtype"])
        except (ValueError, KeyError, TypeError) as error:
            raise Error(f"malformed meta in {directory}") from error
        if missing:
            raise Error(f"malformed meta in {directory}: missing {missing}")
        if meta["version"] != VERSION:
            raise Error(f"unsupported version {meta['version']} in {directory}")
        if binary.stat().st_size != meta["length"] * dtype.itemsize:
            raise Error(f"size of {binary} disagrees with meta")
        if verify and manifest.digest(binary) != meta["sha256"]:
            raise Error(f"sha256 mismatch for {binary}")
        ids = numpy.memmap(
            binary, dtype=dtype, mode="r", shape=(meta["length"],)
        )
        return cls(ids, meta["vocab"], check=False)
