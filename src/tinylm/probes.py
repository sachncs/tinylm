"""Recall and superposition probes for trained language models."""

import dataclasses
import typing

import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional

from tinylm import model, seeding, tokenizer
from tinylm.layers import Layers

BOS = tokenizer.SPECIAL_TOKENS.index("<bos>")
FIRST = len(tokenizer.SPECIAL_TOKENS)
MIN_TOKENS = 4
EXTRA = 4
MATRIX_NDIM = 2
LIMIT = 2000
ALIVE = 0.005
ACTIVE = 1e-4
LEARNING_RATE = 5e-3


@dataclasses.dataclass(frozen=True, slots=True)
class Recalled:
    """How much of held-back text a model reproduces from its first half.

    Attributes:
        token: Share of continuation tokens whose top-1 prediction matches.
        exact: Share of lines whose whole continuation matches.
        lines: Number of lines probed.
    """

    token: float
    exact: float
    lines: int


@dataclasses.dataclass(frozen=True, slots=True)
class Memory:
    """Probes how much training (or held-out) text a model has memorised.

    Attributes:
        net: Model.
        tok: Tokenizer matching the model.
        device: Torch device string.
    """

    net: model.TinyLM
    tok: Tokenizer
    device: str

    def recall(self, lines: list[str]) -> Recalled:
        """Feeds each line's first half and checks the second half.

        Greedy decoding reproduces a continuation exactly when the top-1
        prediction matches at every step given the true prefix, so teacher
        forcing measures both token-level and exact recall.

        Raises:
            ValueError: If lines is empty or a line has too few tokens.
        """
        if not lines:
            raise ValueError("lines must not be empty")
        self.net.to(self.device).eval()
        token_hits = 0
        token_total = 0
        exact = 0
        for line in lines:
            ids = [BOS, *self.tok.encode(line).ids]
            if len(ids) < MIN_TOKENS:
                raise ValueError(f"line too short for recall: {line!r}")
            half = len(ids) // 2
            batch = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                guess = self.net(batch[:, :-1]).argmax(-1)[0]
            hits = guess[half - 1 :] == batch[0, half:]
            token_hits += int(hits.sum())
            token_total += hits.numel()
            exact += int(bool(hits.all()))
        return Recalled(
            token_hits / token_total, exact / len(lines), len(lines)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Hit:
    """Associative-recall accuracy at one distance."""

    distance: int
    accuracy: float
    trials: int


@dataclasses.dataclass(frozen=True, slots=True)
class Associative:
    """Key-value lookup probe: "k v ... filler ... k" must produce v.

    Attributes:
        net: Model.
        device: Torch device string.
    """

    net: model.TinyLM
    device: str

    def accuracy(
        self, distances: tuple[int, ...], trials: int, seed: int
    ) -> tuple[Hit, ...]:
        """Accuracy of recalling the value after `distance` filler tokens.

        Raises:
            ValueError: If trials or a distance is not positive, or a
                sequence would exceed the model's max_seq_len.
        """
        if trials <= 0:
            raise ValueError("trials must be positive")
        if any(d <= 0 for d in distances):
            raise ValueError("every distance must be positive")
        if any(d + EXTRA > self.net.config.max_seq_len for d in distances):
            raise ValueError("a distance exceeds max_seq_len")
        self.net.to(self.device).eval()
        vocab = self.net.config.vocab_size
        generator = torch.Generator()
        generator.manual_seed(seed)
        hits: list[Hit] = []
        for distance in distances:
            key = torch.randint(FIRST, vocab, (trials,), generator=generator)
            value = torch.randint(FIRST, vocab, (trials,), generator=generator)
            filler = torch.randint(
                FIRST, vocab, (trials, distance), generator=generator
            )
            collide = filler == key[:, None]
            filler[collide] = (filler[collide] - FIRST + 1) % (
                vocab - FIRST
            ) + FIRST
            batch = torch.cat(
                [
                    torch.full((trials, 1), BOS),
                    key[:, None],
                    value[:, None],
                    filler,
                    key[:, None],
                ],
                dim=1,
            ).to(self.device)
            with torch.no_grad():
                guess = self.net(batch)[:, -1].argmax(-1).cpu()
            hits.append(
                Hit(distance, float((guess == value).float().mean()), trials)
            )
        return tuple(hits)


@dataclasses.dataclass(frozen=True, slots=True)
class Found:
    """Result of fitting a sparse autoencoder.

    Attributes:
        r2: Fraction of variance reconstructed.
        alive: Dictionary features active on at least 0.5% of samples.
        per_dim: alive divided by the activation width.
    """

    r2: float
    alive: int
    per_dim: float


@dataclasses.dataclass(frozen=True, slots=True)
class Sae:
    """A small sparse autoencoder measuring features per dimension.

    Attributes:
        width: Activation width.
        dictionary: Number of dictionary features (at least width).
        steps: Full-batch Adam steps.
        l1: Sparsity penalty on the feature activations.
        seed: Seed for initialisation.
    """

    width: int
    dictionary: int
    steps: int
    l1: float
    seed: int

    def __post_init__(self) -> None:
        """Validates sizes."""
        if self.dictionary < self.width:
            raise ValueError("dictionary must be at least width")
        if self.width <= 0 or self.steps <= 0 or self.l1 < 0:
            raise ValueError("width and steps must be positive, l1 >= 0")

    def fit(self, activations: torch.Tensor) -> Found:
        """Trains on activations (samples x width) and reports what it found.

        Raises:
            ValueError: If activations does not have `width` columns.
        """
        if (
            activations.ndim != MATRIX_NDIM
            or activations.shape[1] != self.width
        ):
            raise ValueError(f"activations must have width {self.width}")
        seeding.set_seed(self.seed)
        encoder = nn.Linear(self.width, self.dictionary)
        decoder = nn.Linear(self.dictionary, self.width, bias=False)
        optimizer = torch.optim.Adam(
            [*encoder.parameters(), *decoder.parameters()], lr=LEARNING_RATE
        )
        for _ in range(self.steps):
            codes = functional.relu(encoder(activations))
            loss = functional.mse_loss(decoder(codes), activations)
            loss = loss + self.l1 * codes.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                decoder.weight.copy_(
                    functional.normalize(decoder.weight, dim=0)
                )
        with torch.no_grad():
            codes = functional.relu(encoder(activations))
            error = (decoder(codes) - activations).pow(2).sum()
            total = (activations - activations.mean(0)).pow(2).sum()
            alive = int(((codes > ACTIVE).float().mean(0) >= ALIVE).sum())
        return Found(float(1.0 - error / total), alive, alive / self.width)


@dataclasses.dataclass(frozen=True, slots=True)
class Superposition:
    """Measures how crowded a representation is.

    Attributes:
        net: Model whose hidden states are inspected.
    """

    net: model.TinyLM

    @staticmethod
    def rank(matrix: torch.Tensor) -> float:
        """Effective rank (participation ratio) of the singular spectrum.

        Raises:
            ValueError: If matrix is not 2-D.
        """
        if matrix.ndim != MATRIX_NDIM:
            raise ValueError("matrix must be 2-D")
        values: torch.Tensor = torch.linalg.svdvals(matrix.float())  # pyright: ignore[reportUnknownVariableType]
        power = typing.cast("torch.Tensor", values.pow(2))
        return float(power.sum().pow(2) / power.pow(2).sum())

    @staticmethod
    def interference(matrix: torch.Tensor) -> float:
        """Mean absolute cosine between distinct rows (sampled if large).

        Raises:
            ValueError: If matrix is not 2-D or has fewer than two rows.
        """
        if matrix.ndim != MATRIX_NDIM:
            raise ValueError("matrix must be 2-D")
        if matrix.shape[0] < MATRIX_NDIM:
            raise ValueError("need at least two rows")
        step = max(1, matrix.shape[0] // LIMIT)
        unit = functional.normalize(matrix[::step].float(), dim=1)
        cosine = unit @ unit.T
        count = cosine.shape[0]
        off = cosine.abs().sum() - cosine.diagonal().abs().sum()
        return float(off / (count * (count - 1)))

    def hidden(self, tokens: torch.Tensor, layer: int) -> torch.Tensor:
        """Residual states after block `layer`, one row per token.

        Raises:
            ValueError: If layer is outside the model's blocks.
        """
        blocks = Layers(self.net).blocks()
        if not 0 <= layer < len(blocks):
            raise ValueError(f"layer must be in [0, {len(blocks) - 1}]")
        captured: list[torch.Tensor] = []
        handle = blocks[layer].register_forward_hook(
            lambda _m, _a, out: captured.append(out.detach())
        )
        with torch.no_grad():
            self.net.eval()
            self.net(tokens)
        handle.remove()
        return captured[0].reshape(-1, captured[0].shape[-1])
