"""Turns a raw BabyLM corpus directory into offline token stores."""

import dataclasses
import json
import pathlib

from tokenizers import Tokenizer

from tinylm import manifest, packing, tokenizer
from tinylm.atomic import Staging
from tinylm.clean import SOURCES, Options, apply, dedupe
from tinylm.tokens import Token

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

    def read(self, corpus: pathlib.Path) -> dict[str, list[str]]:
        """Reads and cleans every source file in a corpus directory.

        Raises:
            FileNotFoundError: If corpus is not a directory.
            ValueError: If no known source file is present.
            UnicodeDecodeError: If a file is not valid UTF-8.
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
        seen: set[bytes] = set()
        return {
            name: list(
                dedupe(
                    apply(
                        path.read_text(encoding="utf-8").splitlines(),
                        name,
                        self.options,
                    ),
                    min_words=DEDUPE_WORDS,
                    seen=seen,
                )
            )
            for name, path in sorted(paths.items())
        }

    def split(self, lines: list[str]) -> tuple[list[str], list[str]]:
        """Splits lines into (train, dev) by whole consecutive blocks."""
        train: list[str] = []
        dev: list[str] = []
        for start in range(0, len(lines), self.block):
            chunk = lines[start : start + self.block]
            is_dev = (start // self.block) % self.every == self.every - 1
            (dev if is_dev else train).extend(chunk)
        return train, dev

    def encode(self, tok: Tokenizer, lines: list[str]) -> Token:
        """Encodes lines into one stream with BOS between lines.

        The official evaluator prepends BOS to each sentence, so training
        with the same boundary symbol removes a train/eval mismatch.
        """
        ids: list[list[int]] = []
        for start in range(0, len(lines), ENCODE_BATCH):
            batch = tok.encode_batch(lines[start : start + ENCODE_BATCH])
            ids.extend(encoding.ids for encoding in batch)
        return Token.from_tensor(packing.join(ids, BOS), tok.get_vocab_size())

    def run(self, corpus: pathlib.Path, out: pathlib.Path) -> Stats:
        """Prepares corpus into out/{train,dev,tokenizer.json,stats.json}.

        The tokenizer is trained on training lines only. Output is written
        to a sibling directory first and renamed on success, so a failed run
        leaves nothing behind.

        Raises:
            FileExistsError: If out already exists.
            FileNotFoundError: If corpus does not exist.
            ValueError: If the corpus has no known sources or no text.
        """
        if out.exists():
            raise FileExistsError(out)  # fail before reading anything
        sources = self.read(corpus)
        train_lines: list[str] = []
        dev_lines: list[str] = []
        for name in sorted(sources):
            train, dev = self.split(sources[name])
            train_lines.extend(train)
            dev_lines.extend(dev)
        if not train_lines:
            raise ValueError("corpus has no text after cleaning and splitting")
        known = set(train_lines)
        kept = [
            line
            for line in dev_lines
            if len(line.split()) < self.unique or line not in known
        ]
        if not kept:
            raise ValueError("corpus has no held-out text after cleaning")
        tok = tokenizer.train(train_lines, self.vocab, self.kind)
        train_stream = self.encode(tok, train_lines)
        dev_stream = self.encode(tok, kept)
        names = sorted(f"{name}{SUFFIX}" for name in sources)
        inputs = manifest.build(corpus, names).entries
        with Staging(out) as staged:
            tokenizer.save(tok, staged / "tokenizer.json")
            stats = Stats(
                len(train_lines),
                len(kept),
                len(train_stream),
                len(dev_stream),
                sum(len(line) + 1 for line in kept),
                len(dev_lines) - len(kept),
                inputs,
                dataclasses.asdict(self),
                manifest.digest(staged / "tokenizer.json"),
            )
            train_stream.store(staged / "train")
            dev_stream.store(staged / "dev")
            (staged / "stats.json").write_text(
                json.dumps(dataclasses.asdict(stats), indent=2, sort_keys=True)
                + "\n"
            )
        return stats
