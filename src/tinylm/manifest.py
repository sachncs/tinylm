"""Corpus manifests: pinned file lists with checksums and sizes."""

import dataclasses
import hashlib
import json
import pathlib

CHUNK_BYTES = 1 << 20


class IntegrityError(Exception):
    """Raised when a file or manifest fails verification."""


@dataclasses.dataclass(frozen=True)
class Entry:
    """One corpus file.

    Attributes:
        name: Path relative to the corpus directory.
        sha256: Hex SHA-256 of the file contents.
        num_bytes: File size in bytes.
    """

    name: str
    sha256: str
    num_bytes: int


@dataclasses.dataclass(frozen=True)
class Manifest:
    """A sorted collection of entries."""

    entries: tuple[Entry, ...]


def file_digest(path: pathlib.Path) -> str:
    """Returns the hex SHA-256 of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: pathlib.Path, entry: Entry) -> None:
    """Checks a file against its entry.

    Raises:
        IntegrityError: If the file is empty, truncated, not valid UTF-8, or
            has the wrong checksum.
    """
    size = path.stat().st_size
    if size == 0:
        raise IntegrityError(f"{entry.name} is empty")
    if size != entry.num_bytes:
        raise IntegrityError(
            f"{entry.name} size {size} != expected {entry.num_bytes}"
        )
    try:
        with path.open(encoding="utf-8") as handle:
            while handle.read(CHUNK_BYTES):
                pass
    except UnicodeDecodeError as error:
        raise IntegrityError(f"{entry.name} is not valid utf-8") from error
    if file_digest(path) != entry.sha256:
        raise IntegrityError(f"{entry.name} sha256 mismatch")


def build(directory: pathlib.Path, names: list[str]) -> Manifest:
    """Builds a manifest for the named files under directory.

    Raises:
        ValueError: If a name is duplicated or is not a relative path inside
            directory.
    """
    if len(set(names)) != len(names):
        raise ValueError("duplicate file names")
    root = directory.resolve()
    entries: list[Entry] = []
    for name in names:
        path = (root / name).resolve()
        if pathlib.PurePath(name).is_absolute() or not path.is_relative_to(
            root
        ):
            raise ValueError(f"{name!r} must be a relative path in the corpus")
        entries.append(Entry(name, file_digest(path), path.stat().st_size))
    return Manifest(tuple(sorted(entries, key=lambda e: e.name)))


def save(manifest: Manifest, path: pathlib.Path) -> None:
    """Writes a manifest as stable, sorted JSON."""
    payload = {"entries": [dataclasses.asdict(e) for e in manifest.entries]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load(path: pathlib.Path) -> Manifest:
    """Reads a manifest.

    Raises:
        IntegrityError: If the file is not a well-formed manifest.
    """
    try:
        payload = json.loads(path.read_text())
        entries = tuple(Entry(**item) for item in payload["entries"])
    except (ValueError, KeyError, TypeError) as error:
        raise IntegrityError(f"malformed manifest {path}") from error
    return Manifest(entries)


def verify_directory(directory: pathlib.Path, manifest: Manifest) -> None:
    """Verifies every manifest entry exists and is intact.

    Raises:
        IntegrityError: If any file is missing or fails verification.
    """
    for entry in manifest.entries:
        path = directory / entry.name
        if not path.is_file():
            raise IntegrityError(f"{entry.name} is missing")
        verify_file(path, entry)
