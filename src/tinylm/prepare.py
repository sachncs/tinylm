"""Turns a raw BabyLM corpus directory into offline token stores."""

import array
import collections.abc
import dataclasses
import hashlib
import itertools
import json
import pathlib
import tempfile

import numpy
from tokenizers import Tokenizer

from tinylm import manifest, tokenizer
from tinylm.atomic import Staging
from tinylm.clean import SOURCES, Options, apply, dedupe
from tinylm.tokens import Token, dtype_for

VERSION = 2
SUFFIX = ".train.txt"
BOS = tokenizer.SPECIAL_TOKENS.index("<bos>")
DEDUPE_WORDS = 20
ENCODE_BATCH = 10_000


@dataclasses.dataclass(frozen=True, slots=True)
class Stats:
    """Counts and lineage describing a prepared corpus.

    Attributes:
        train_lines: Cleaned training lines.
        dev_lines: Cleaned held-out lines.
        train_tokens: Tokens in the training stream.
        dev_tokens: Tokens in the held-out stream.
        dev_chars: Characters (with newlines) in the held-out text.
        dev_dropped: Held-out lines removed because they also occur in train.
        inputs: Name and SHA-256 of every raw input file.
        settings: The Prepare settings that produced this output.
        tokenizer_sha256: SHA-256 of tokenizer.json.
        version: Layout version of the output directory.
    """

    train_lines: int
    dev_lines: int
    train_tokens: int
    dev_tokens: int
    dev_chars: int
    dev_dropped: int
    inputs: tuple[manifest.Entry, ...]
    settings: dict[str, object]
    tokenizer_sha256: str
    version: int = VERSION


@dataclasses.dataclass(frozen=True, slots=True)
class Prepare:
    """Cleans, splits, tokenises and stores a corpus.

    Attributes:
        vocab: Tokenizer vocabulary size.
        kind: "bpe" or "unigram".
        block: Consecutive lines per split block.
        every: Every n-th block is held out as dev data.
        options: Cleaning options.
        unique: Held-out lines with at least this many words must not occur
            verbatim in the training lines; such lines are dropped from dev.
    """

    vocab: int
    kind: str
    block: int
    every: int
    options: Options
    unique: int = 6

    def __post_init__(self) -> None:
        """Validates the split settings."""
        if self.block <= 0:
            raise ValueError("block must be positive")
        if self.every <= 0:
            raise ValueError("every must be positive")
        if self.unique <= 0:
            raise ValueError("unique must be positive")

    def sources(self, corpus: pathlib.Path) -> dict[str, pathlib.Path]:
        """Maps each known source name to its file, in sorted order.

        Raises:
            FileNotFoundError: If corpus is not a directory.
            ValueError: If no known source file is present.
        """
        if not corpus.is_dir():
            raise FileNotFoundError(corpus)
        paths = {
            p.name.removesuffix(SUFFIX): p
            for p in sorted(corpus.glob(f"*{SUFFIX}"))
        }
        unknown = sorted(set(paths) - set(SOURCES))
        if unknown or not paths:
            raise ValueError(f"no known source files (unknown: {unknown})")
        return dict(sorted(paths.items()))

    def lines(
        self, path: pathlib.Path, name: str, seen: set[bytes]
    ) -> collections.abc.Iterator[str]:
        """Streams the cleaned, de-duplicated lines of one source file.

        Raises:
            UnicodeDecodeError: If the file is not valid UTF-8.
        """
        with path.open(encoding="utf-8") as handle:
            raw = (line.rstrip("\n") for line in handle)
            yield from dedupe(
                apply(raw, name, self.options),
                min_words=DEDUPE_WORDS,
                seen=seen,
            )

    def read(self, corpus: pathlib.Path) -> dict[str, list[str]]:
        """Cleans every source into memory; for small inputs and tests.

        Raises:
            FileNotFoundError: If corpus is not a directory.
            ValueError: If no known source file is present.
            UnicodeDecodeError: If a file is not valid UTF-8.
        """
        seen: set[bytes] = set()
        return {
            name: list(self.lines(path, name, seen))
            for name, path in self.sources(corpus).items()
        }

    def encode(self, tok: Tokenizer, path: pathlib.Path) -> Token:
        """Encodes a text file line by line, with BOS after every line.

        The official evaluator prepends BOS to each sentence, so training
        with the same boundary symbol removes a train/eval mismatch. The file
        is streamed in batches, so memory stays small.
        """
        kind = dtype_for(tok.get_vocab_size())
        pieces: list[numpy.ndarray] = []
        with path.open(encoding="utf-8") as handle:
            while batch := [
                line.rstrip("\n")
                for line in itertools.islice(handle, ENCODE_BATCH)
            ]:
                flat: list[int] = []
                for encoding in tok.encode_batch(batch):
                    if encoding.ids:
                        flat.extend(encoding.ids)
                        flat.append(BOS)
                pieces.append(numpy.array(flat, dtype=kind))
        return Token(numpy.concatenate(pieces), tok.get_vocab_size())

    def run(self, corpus: pathlib.Path, out: pathlib.Path) -> Stats:
        """Prepares corpus into out/{train,dev,tokenizer.json,stats.json}.

        Streams the corpus: cleaned lines go to temporary files and only
        64-bit fingerprints of long training lines are kept in memory (for
        the dev decontamination). The tokenizer is trained on training lines
        only. Output is written to a sibling directory first and renamed on
        success, so a failed run leaves nothing behind.

        Raises:
            FileExistsError: If out already exists.
            FileNotFoundError: If corpus does not exist.
            ValueError: If the corpus has no known sources or no text.
        """
        if out.exists():
            raise FileExistsError(out)
        files = self.sources(corpus)
        names = sorted(f"{name}{SUFFIX}" for name in files)
        inputs = manifest.build(corpus, names).entries
        with tempfile.TemporaryDirectory(dir=out.parent) as work_name:
            work = pathlib.Path(work_name)
            train_path, dev_path, kept_path = (
                work / "train.txt",
                work / "dev.txt",
                work / "dev-kept.txt",
            )
            counts = self.route(files, train_path, dev_path)
            known, n_train, n_dev = counts
            if n_train == 0:
                raise ValueError(
                    "corpus has no text after cleaning and splitting"
                )
            n_kept, chars = self.decontaminate(dev_path, kept_path, known)
            if n_kept == 0:
                raise ValueError("corpus has no held-out text after cleaning")
            with train_path.open(encoding="utf-8") as handle:
                tok = tokenizer.train(
                    (line.rstrip("\n") for line in handle),
                    self.vocab,
                    self.kind,
                )
            train_stream = self.encode(tok, train_path)
            dev_stream = self.encode(tok, kept_path)
            with Staging(out) as staged:
                tokenizer.save(tok, staged / "tokenizer.json")
                stats = Stats(
                    n_train,
                    n_kept,
                    len(train_stream),
                    len(dev_stream),
                    chars,
                    n_dev - n_kept,
                    inputs,
                    dataclasses.asdict(self),
                    manifest.digest(staged / "tokenizer.json"),
                )
                train_stream.store(staged / "train")
                dev_stream.store(staged / "dev")
                (staged / "stats.json").write_text(
                    json.dumps(
                        dataclasses.asdict(stats), indent=2, sort_keys=True
                    )
                    + "\n"
                )
        return stats

    def route(
        self,
        files: dict[str, pathlib.Path],
        train_path: pathlib.Path,
        dev_path: pathlib.Path,
    ) -> tuple[numpy.ndarray, int, int]:
        """Splits cleaned lines into train and dev files by whole blocks.

        Returns:
            Sorted fingerprints of training lines with at least `unique`
            words, and the training and dev line counts.
        """
        seen: set[bytes] = set()
        hashes = array.array("Q")
        n_train = n_dev = 0
        with (
            train_path.open("w", encoding="utf-8") as train,
            dev_path.open("w", encoding="utf-8") as dev,
        ):
            for name, path in files.items():
                for index, line in enumerate(self.lines(path, name, seen)):
                    if (index // self.block) % self.every == self.every - 1:
                        dev.write(line + "\n")
                        n_dev += 1
                    else:
                        train.write(line + "\n")
                        n_train += 1
                        if len(line.split()) >= self.unique:
                            hashes.append(fingerprint(line))
        return (
            numpy.unique(numpy.frombuffer(hashes, dtype=numpy.uint64)),
            n_train,
            n_dev,
        )

    def decontaminate(
        self,
        dev_path: pathlib.Path,
        kept_path: pathlib.Path,
        known: numpy.ndarray,
    ) -> tuple[int, int]:
        """Writes the dev lines that do not occur verbatim in training.

        Lines shorter than `unique` words are kept even if they repeat (a
        reply such as "yeah." legitimately appears everywhere).

        Returns:
            The number of dev lines kept and their character count.
        """
        kept = chars = 0
        with (
            dev_path.open(encoding="utf-8") as source,
            kept_path.open("w", encoding="utf-8") as target,
        ):
            for raw in source:
                line = raw.rstrip("\n")
                if len(line.split()) >= self.unique:
                    where = numpy.searchsorted(
                        known, numpy.uint64(fingerprint(line))
                    )
                    if where < known.size and known[where] == fingerprint(line):
                        continue
                target.write(line + "\n")
                kept += 1
                chars += len(line) + 1
        return kept, chars


def fingerprint(line: str) -> int:
    """A 64-bit fingerprint of a line (collisions are negligible here)."""
    digest = hashlib.blake2b(line.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")
