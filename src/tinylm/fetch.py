"""Downloads pinned dataset snapshots once and verifies them forever after."""

import collections.abc
import dataclasses
import pathlib
import re
import shutil

from tinylm import manifest

REVISION = re.compile(r"^[0-9a-f]{40}$")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
IGNORED = (".cache",)

Downloader = collections.abc.Callable[
    [str, str, pathlib.Path, tuple[str, ...]], None
]


@dataclasses.dataclass(frozen=True, slots=True)
class Pin:
    """One dataset snapshot, identified by an immutable commit hash.

    Attributes:
        name: Folder name under the data root; a plain file-name component.
        repo: Hugging Face dataset id, "owner/name".
        revision: Full 40-character lowercase hex commit; moving refs such as
            "main" are rejected so the data can never change underneath us.
        patterns: Glob patterns of the files to download.
    """

    name: str
    repo: str
    revision: str
    patterns: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates every field."""
        if not NAME.match(self.name):
            raise ValueError(f"name {self.name!r} is not a plain file name")
        if self.repo.count("/") != 1 or not all(self.repo.split("/")):
            raise ValueError(f"repo {self.repo!r} must be 'owner/name'")
        if not REVISION.match(self.revision):
            raise ValueError("revision must be a full lowercase hex commit")
        if not self.patterns:
            raise ValueError("patterns must not be empty")


def download(
    repo: str, revision: str, dest: pathlib.Path, patterns: tuple[str, ...]
) -> None:
    """Downloads a pinned snapshot with huggingface_hub."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415  # pyright: ignore[reportUnknownVariableType]

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        local_dir=dest,
        allow_patterns=list(patterns),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class Fetch:
    """Fetches pins into a data root and records their manifests.

    Attributes:
        root: Folder that receives one sub-folder per pin.
        manifests: Folder for the committed checksum manifests.
        downloader: Callable doing the actual download; replaceable in tests.
    """

    root: pathlib.Path
    manifests: pathlib.Path
    downloader: Downloader = download

    def run(self, pin: Pin) -> None:
        """Makes pin available locally and verified.

        If the folder exists it is only verified (never repaired). Otherwise
        the snapshot is downloaded into a sibling .partial folder, hashed,
        written to a manifest and renamed into place, so a failure leaves
        neither data nor manifest behind.

        Raises:
            manifest.Error: If existing data fails verification, or the
                download is empty or contains an empty file.
        """
        dest = self.root / pin.name
        record = self.manifests / f"{pin.name}.json"
        if dest.exists():
            manifest.check(dest, manifest.load(record))
            return
        staging = dest.with_name(dest.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            self.downloader(pin.repo, pin.revision, staging, pin.patterns)
            names = sorted(
                str(path.relative_to(staging))
                for path in staging.rglob("*")
                if path.is_file()
                and not set(path.relative_to(staging).parts) & set(IGNORED)
            )
            if not names:
                raise manifest.Error(f"{pin.repo} download produced no files")
            built = manifest.build(staging, names)
            for entry in built.entries:
                manifest.verify(staging / entry.name, entry)
            self.manifests.mkdir(parents=True, exist_ok=True)
            manifest.save(built, record)
            staging.rename(dest)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            record.unlink(missing_ok=True)
            raise
