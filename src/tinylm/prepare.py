"""Turns a raw BabyLM corpus directory into tokenised train/dev streams."""

import dataclasses
import json
import pathlib

import torch
from tokenizers import Tokenizer

from tinylm import clean, packing, tokenizer

SUFFIX = ".train.txt"
BOS_ID = tokenizer.SPECIAL_TOKENS.index("<bos>")
DEDUP_MIN_WORDS = 20
ENCODE_BATCH = 10_000


@dataclasses.dataclass(frozen=True)
class Settings:
    """Preparation settings.

    Attributes:
        vocab_size: Tokenizer vocabulary size.
        tokenizer_kind: "bpe" or "unigram".
        dev_block_lines: Consecutive lines per split block.
        dev_every_blocks: Every n-th block is held out as dev data.
        options: Cleaning options.
    """

    vocab_size: int
    tokenizer_kind: str
    dev_block_lines: int
    dev_every_blocks: int
    options: clean.Options


@dataclasses.dataclass(frozen=True)
class Stats:
    """Counts describing a prepared corpus."""

    train_lines: int
    dev_lines: int
    train_tokens: int
    dev_tokens: int
    dev_chars: int


def read_sources(
    corpus: pathlib.Path, options: clean.Options
) -> dict[str, list[str]]:
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
    unknown = sorted(set(paths) - set(clean.SOURCES))
    if unknown or not paths:
        raise ValueError(f"no known source files (unknown: {unknown})")
    return {
        name: list(
            clean.drop_duplicate_long_lines(
                clean.clean_lines(
                    path.read_text(encoding="utf-8").splitlines(), name, options
                ),
                min_words=DEDUP_MIN_WORDS,
            )
        )
        for name, path in paths.items()
    }


def split_blocks(
    lines: list[str], block_lines: int, every: int
) -> tuple[list[str], list[str]]:
    """Splits lines into (train, dev) by whole consecutive blocks."""
    train: list[str] = []
    dev: list[str] = []
    for start in range(0, len(lines), block_lines):
        block = lines[start : start + block_lines]
        is_dev = (start // block_lines) % every == every - 1
        (dev if is_dev else train).extend(block)
    return train, dev


def encode_stream(tok: Tokenizer, lines: list[str]) -> torch.Tensor:
    """Encodes lines into one int32 stream with BOS between lines.

    The official evaluator prepends BOS to each sentence, so training with
    the same boundary symbol removes a train/eval mismatch.
    """
    ids: list[list[int]] = []
    for start in range(0, len(lines), ENCODE_BATCH):
        batch = tok.encode_batch(lines[start : start + ENCODE_BATCH])
        ids.extend(encoding.ids for encoding in batch)
    return packing.build_stream(ids, BOS_ID).to(torch.int32)


def prepare(
    corpus: pathlib.Path, out_dir: pathlib.Path, settings: Settings
) -> Stats:
    """Cleans, splits, tokenises and saves a corpus.

    The tokenizer is trained on training lines only. Outputs in out_dir:
    tokenizer.json, train.pt, dev.pt (int32 streams) and stats.json.

    Raises:
        ValueError: If settings are invalid or the corpus has no text.
        FileNotFoundError: If corpus does not exist.
    """
    if settings.dev_block_lines <= 0:
        raise ValueError("dev_block_lines must be positive")
    if settings.dev_every_blocks <= 0:
        raise ValueError("dev_every_blocks must be positive")
    sources = read_sources(corpus, settings.options)
    train_lines: list[str] = []
    dev_lines: list[str] = []
    for name in sorted(sources):
        train, dev = split_blocks(
            sources[name], settings.dev_block_lines, settings.dev_every_blocks
        )
        train_lines.extend(train)
        dev_lines.extend(dev)
    if not train_lines:
        raise ValueError("corpus has no text after cleaning")
    tok = tokenizer.train(
        train_lines, settings.vocab_size, settings.tokenizer_kind
    )
    train_stream = encode_stream(tok, train_lines)
    dev_stream = encode_stream(tok, dev_lines)
    stats = Stats(
        len(train_lines),
        len(dev_lines),
        train_stream.numel(),
        dev_stream.numel(),
        sum(len(line) + 1 for line in dev_lines),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tok, out_dir / "tokenizer.json")
    torch.save(train_stream, out_dir / "train.pt")
    torch.save(dev_stream, out_dir / "dev.pt")
    (out_dir / "stats.json").write_text(
        json.dumps(dataclasses.asdict(stats), indent=2, sort_keys=True) + "\n"
    )
    return stats
