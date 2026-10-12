"""In-process minimal-pair benchmark scoring (BLiMP-style)."""

import dataclasses
import json
import pathlib

import torch
from tokenizers import Tokenizer
from torch.nn import functional

from tinylm import model, tokenizer

BOS = tokenizer.SPECIAL_TOKENS.index("<bos>")
PAD = tokenizer.SPECIAL_TOKENS.index("<pad>")
BATCH = 64


class Error(Exception):
    """Raised when benchmark files are missing fields or malformed."""


@dataclasses.dataclass(frozen=True, slots=True)
class Pair:
    """A grammatical sentence and its ungrammatical counterpart."""

    good: str
    bad: str

    @classmethod
    def read(cls, path: pathlib.Path) -> list[Pair]:
        """Reads minimal pairs from a JSON-lines file.

        Raises:
            Error: If the file is empty, has malformed JSON, or a row lacks
                sentence_good or sentence_bad.
        """
        pairs: list[Pair] = []
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise Error(
                    f"{path.name}: malformed JSON on line {number}"
                ) from error
            for key in ("sentence_good", "sentence_bad"):
                if key not in row:
                    raise Error(f"{path.name}: line {number} lacks {key}")
            pairs.append(cls(row["sentence_good"], row["sentence_bad"]))
        if not pairs:
            raise Error(f"{path.name} is empty")
        return pairs


@dataclasses.dataclass(frozen=True, slots=True)
class Confidence:
    """Precision and coverage of confident answers at one margin threshold.

    Attributes:
        tau: Minimum |log-probability margin| for an answer to be confident.
        total: Number of items.
        confident: Items with |margin| >= tau.
        correct: Confident items whose margin is positive.
        precision: correct / confident, or None if nothing is confident.
        coverage: confident / total.
    """

    tau: float
    total: int
    confident: int
    correct: int
    precision: float | None
    coverage: float

    @classmethod
    def of(cls, margins: list[float], tau: float) -> Confidence:
        """Summarises margins at threshold tau.

        Raises:
            ValueError: If margins is empty or tau is negative or NaN.
        """
        if not margins:
            raise ValueError("margins must not be empty")
        if not tau >= 0.0:
            raise ValueError("tau must be a non-negative number")
        sure = [m for m in margins if abs(m) >= tau]
        right = sum(m > 0 for m in sure)
        return cls(
            tau=tau,
            total=len(margins),
            confident=len(sure),
            correct=right,
            precision=right / len(sure) if sure else None,
            coverage=len(sure) / len(margins),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Report:
    """Accuracy per paradigm, their unweighted mean, and pooled confidence."""

    by_paradigm: dict[str, float]
    mean: float
    confidence: dict[float, Confidence]


@dataclasses.dataclass(frozen=True, slots=True)
class Bench:
    """Scores sentences with a model and evaluates minimal-pair files.

    Attributes:
        net: Model.
        tok: Tokenizer matching the model.
        device: Torch device string.
    """

    net: model.TinyLM
    tok: Tokenizer
    device: str

    def score(self, sentences: list[str]) -> list[float]:
        """Total log-probability of each sentence given a leading BOS.

        Raises:
            ValueError: If a sentence is empty or too long for the model.
        """
        self.net.to(self.device).eval()
        encoded: list[list[int]] = []
        for sentence in sentences:
            ids = self.tok.encode(sentence).ids
            if not ids:
                raise ValueError("empty sentence")
            if len(ids) + 1 > self.net.config.max_seq_len:
                raise ValueError("sentence exceeds max_seq_len")
            encoded.append(ids)
        totals: list[float] = []
        for start in range(0, len(encoded), BATCH):
            chunk = encoded[start : start + BATCH]
            width = max(len(ids) for ids in chunk) + 1
            batch = torch.full((len(chunk), width), PAD, dtype=torch.long)
            lengths = torch.tensor([len(ids) for ids in chunk])
            for row, ids in enumerate(chunk):
                batch[row, 0] = BOS
                batch[row, 1 : len(ids) + 1] = torch.tensor(ids)
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = self.net(batch[:, :-1]).float()
            logp = functional.log_softmax(logits, dim=-1)
            picked = logp.gather(-1, batch[:, 1:].unsqueeze(-1)).squeeze(-1)
            positions = torch.arange(width - 1, device=self.device).unsqueeze(0)
            mask = positions < lengths.to(self.device).unsqueeze(1)
            totals.extend(float(t) for t in (picked * mask).sum(dim=1).cpu())
        return totals

    def margins(self, pairs: list[Pair]) -> list[float]:
        """Returns score(good) - score(bad) for every pair.

        Raises:
            ValueError: If pairs is empty.
        """
        if not pairs:
            raise ValueError("pairs must not be empty")
        good = self.score([p.good for p in pairs])
        bad = self.score([p.bad for p in pairs])
        return [g - b for g, b in zip(good, bad, strict=True)]

    def accuracy(self, pairs: list[Pair]) -> float:
        """Fraction of pairs where the good sentence scores strictly higher."""
        margins = self.margins(pairs)
        return sum(m > 0 for m in margins) / len(margins)

    def evaluate(
        self, directory: pathlib.Path, taus: tuple[float, ...] = ()
    ) -> Report:
        """Scores every .jsonl file in a directory as one paradigm.

        Args:
            directory: Folder of minimal-pair files.
            taus: Margin thresholds for pooled precision and coverage.

        Raises:
            FileNotFoundError: If directory does not exist.
            Error: If it holds no .jsonl files or a file is invalid.
        """
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files = sorted(directory.glob("*.jsonl"))
        if not files:
            raise Error(f"no .jsonl files in {directory}")
        margins = {path.stem: self.margins(Pair.read(path)) for path in files}
        by_paradigm = {
            name: sum(m > 0 for m in values) / len(values)
            for name, values in margins.items()
        }
        pooled = [m for values in margins.values() for m in values]
        return Report(
            by_paradigm,
            sum(by_paradigm.values()) / len(by_paradigm),
            {tau: Confidence.of(pooled, tau) for tau in taus},
        )
