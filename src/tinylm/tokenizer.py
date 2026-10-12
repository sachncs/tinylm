"""Deterministic tokenizer training (byte-level BPE or Unigram)."""

import collections.abc
import itertools
import json
import pathlib

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SCORE_DECIMALS = 6
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>", "<mask>")
KINDS = ("bpe", "unigram")
BYTE_ALPHABET_SIZE = 256
MIN_VOCAB = {
    "bpe": BYTE_ALPHABET_SIZE + len(SPECIAL_TOKENS) + 1,
    "unigram": len(SPECIAL_TOKENS) + 1,
}


def train(
    corpus: collections.abc.Iterable[str], vocab_size: int, kind: str
) -> Tokenizer:
    """Trains a tokenizer on non-blank lines of corpus.

    Args:
        corpus: Iterable of text lines.
        vocab_size: Target vocabulary size including special tokens.
        kind: "bpe" (byte-level) or "unigram".

    Returns:
        A trained tokenizer whose special tokens occupy ids 0..4.

    Raises:
        ValueError: If kind is unknown, vocab_size is too small, or the
            corpus has no non-blank lines.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if vocab_size < MIN_VOCAB[kind]:
        raise ValueError(f"vocab_size must be at least {MIN_VOCAB[kind]}")
    stream = (line for line in corpus if line.strip())
    first = next(stream, None)
    if first is None:
        raise ValueError("corpus has no non-blank lines")
    lines = itertools.chain([first], stream)
    special = list(SPECIAL_TOKENS)
    if kind == "bpe":
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
    else:
        tok = Tokenizer(models.Unigram())
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special,
            unk_token="<unk>",
            show_progress=False,
        )
    tok.train_from_iterator(lines, trainer)
    return canonicalize(tok) if kind == "unigram" else tok


def canonicalize(tok: Tokenizer) -> Tokenizer:
    """Makes unigram training reproducible.

    The Rust trainer iterates a hash map, so pieces with identical scores
    can swap ids between runs and scores differ in the last float digit.
    Scores are rounded and ties ordered by piece text. Special tokens keep
    ids 0..4.
    """
    spec = json.loads(tok.to_str())
    vocab = spec["model"]["vocab"]
    head = vocab[: len(SPECIAL_TOKENS)]
    tail = [
        [piece, round(score, SCORE_DECIMALS)]
        for piece, score in vocab[len(SPECIAL_TOKENS) :]
    ]
    tail.sort(key=lambda p: (-p[1], p[0]))
    spec["model"]["vocab"] = head + tail
    return Tokenizer.from_str(json.dumps(spec))


def save(tok: Tokenizer, path: pathlib.Path) -> None:
    """Writes a tokenizer to a JSON file."""
    tok.save(str(path))


def load(path: pathlib.Path) -> Tokenizer:
    """Reads a tokenizer from a JSON file.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    return Tokenizer.from_file(str(path))
