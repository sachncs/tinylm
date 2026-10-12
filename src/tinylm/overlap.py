"""Detects benchmark text leaking into the training corpus."""

import collections.abc
import dataclasses
import re

WORD = re.compile(r"[a-z0-9']+")


class Error(Exception):
    """Raised when evaluation text overlaps the training corpus too much."""


@dataclasses.dataclass(frozen=True, slots=True)
class Overlap:
    """Word n-gram overlap between training text and evaluation sentences.

    Attributes:
        n: Words per n-gram.
    """

    n: int

    def __post_init__(self) -> None:
        """Validates n."""
        if self.n <= 0:
            raise ValueError("n must be positive")

    def ngrams(self, text: str) -> set[tuple[str, ...]]:
        """Returns the set of lowercase word n-grams of text."""
        words = WORD.findall(text.lower())
        return {
            tuple(words[i : i + self.n]) for i in range(len(words) - self.n + 1)
        }

    def rate(
        self,
        train: collections.abc.Iterable[str],
        sentences: collections.abc.Sequence[str],
    ) -> float:
        """Fraction of eligible sentences sharing any n-gram with training.

        Sentences with fewer than n words are not eligible and are skipped.

        Raises:
            ValueError: If either input is empty or no sentence is eligible.
        """
        if not sentences:
            raise ValueError("sentences is empty")
        eligible = [g for g in (self.ngrams(s) for s in sentences) if g]
        if not eligible:
            raise ValueError(
                f"no eligible sentences with at least {self.n} words"
            )
        wanted: set[tuple[str, ...]] = set()
        for grams in eligible:
            wanted |= grams
        seen: set[tuple[str, ...]] = set()
        lines = 0
        for line in train:
            lines += 1
            seen |= self.ngrams(line) & wanted
        if lines == 0:
            raise ValueError("train is empty")
        return sum(1 for grams in eligible if grams & seen) / len(eligible)

    def check(
        self,
        train: collections.abc.Iterable[str],
        sentences: collections.abc.Sequence[str],
        threshold: float,
    ) -> float:
        """Returns the overlap rate, raising if it exceeds threshold.

        Raises:
            ValueError: If threshold is outside [0, 1].
            Error: If the overlap rate exceeds threshold.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        found = self.rate(train, sentences)
        if found > threshold:
            raise Error(
                f"overlap {found:.3f} exceeds threshold {threshold:.3f}"
            )
        return found
