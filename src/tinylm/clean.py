"""Per-source text cleaning for the BabyLM corpora."""

import collections.abc
import dataclasses
import hashlib
import re
import unicodedata

SOURCES = (
    "bnc_spoken",
    "childes",
    "gutenberg",
    "open_subtitles",
    "simple_wiki",
    "switchboard",
)
TAGGED_SOURCES = ("childes", "switchboard")
MIN_CASE_LETTERS = 3

TYPOGRAPHY = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)
SPEAKER_TAG = re.compile(r"^(?:\*([A-Z0-9]{2,4})|([AB])):\t+")
STAGE_DIRECTION = re.compile(r"^\[.*\]$")
WIKI_HEADING = re.compile(r"^(?:=\s*)+(.+?)(?:\s*=)+$")
STANDALONE_I = re.compile(r"\bi\b")
UNSPACED = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclasses.dataclass(frozen=True)
class Options:
    """Cleaning switches, each an ablation axis.

    Attributes:
        keep_speaker_tags: Keep "MOT: " style prefixes instead of stripping.
        recase: Convert ALL-CAPS lines to sentence case.
        drop_stage_directions: Drop fully bracketed lines like "[laughs.]".
    """

    keep_speaker_tags: bool = False
    recase: bool = True
    drop_stage_directions: bool = True


def words(text: str) -> int:
    """Counts words: spaced tokens, plus each character of unspaced scripts.

    Han, Hiragana and Katakana are written without spaces, so every such
    character counts as one word; everything else is split on whitespace.
    """
    unspaced = len(UNSPACED.findall(text))
    rest = UNSPACED.sub(" ", text)
    return unspaced + len(rest.split())


def typography(text: str) -> str:
    """NFC-normalises, straightens quotes/dashes, drops controls and spaces."""
    text = unicodedata.normalize("NFC", text).translate(TYPOGRAPHY)
    text = "".join(
        ch for ch in text if ch.isspace() or unicodedata.category(ch) != "Cc"
    )
    return " ".join(text.split())


def recase(text: str) -> str:
    """Converts an all-uppercase line to sentence case.

    Lines with fewer than MIN_CASE_LETTERS letters, or any lowercase letter,
    are returned unchanged.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < MIN_CASE_LETTERS or not all(c.isupper() for c in letters):
        return text
    lowered = STANDALONE_I.sub("I", text.lower())
    first = next(i for i, ch in enumerate(lowered) if ch.isalpha())
    return lowered[:first] + lowered[first].upper() + lowered[first + 1 :]


def apply(
    lines: collections.abc.Iterable[str], source: str, options: Options
) -> collections.abc.Iterator[str]:
    """Cleans lines of one source, dropping those that become empty.

    Raises:
        ValueError: If source is not a known corpus source.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    for raw in lines:
        line = raw.rstrip("\n")
        prefix = ""
        match = SPEAKER_TAG.match(line)
        if match and source in TAGGED_SOURCES:
            if options.keep_speaker_tags:
                prefix = f"{match.group(1) or match.group(2)}: "
            line = line[match.end() :]
        if options.drop_stage_directions and STAGE_DIRECTION.match(
            line.strip()
        ):
            continue
        heading = WIKI_HEADING.match(line.strip())
        if source == "simple_wiki" and heading:
            line = heading.group(1)
        line = typography(line)
        if options.recase:
            line = recase(line)
        if line:
            yield prefix + line


def dedupe(
    lines: collections.abc.Iterable[str],
    min_words: int,
    seen: set[bytes] | None = None,
) -> collections.abc.Iterator[str]:
    """Drops repeats of lines with at least min_words words.

    Short lines such as "Yeah." are legitimately repeated and are kept.
    Pass the same `seen` set to several calls to deduplicate across them.

    Raises:
        ValueError: If min_words is not positive.
    """
    if min_words <= 0:
        raise ValueError("min_words must be positive")
    seen = set() if seen is None else seen
    for line in lines:
        if words(line) >= min_words:
            key = hashlib.blake2b(line.encode(), digest_size=16).digest()
            if key in seen:
                continue
            seen.add(key)
        yield line
