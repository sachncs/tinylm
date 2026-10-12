"""Training loop with a hard epoch budget, exact resume and evaluation."""

import collections.abc
import dataclasses
import math
import pathlib

import torch
from torch.nn import functional

from tinylm import ledger, model, objectives, optim, packing, tokenizer
from tinylm.ema import Average
from tinylm.tokens import Token

CHECKPOINT = "checkpoint.pt"
MASK = tokenizer.SPECIAL_TOKENS.index("<mask>")
STRIDE = 1_000_003
SCHEDULES = ("cosine", "wsd")


class Error(Exception):
    """Raised when training cannot continue, e.g. the loss is not finite."""


@dataclasses.dataclass(frozen=True, slots=True)
class Result:
    """Outcome of a training call.

    Attributes:
        losses: Training loss of every step taken in this call.
        steps: Total optimizer steps taken so far (including resumed ones).
        budget: The epoch ledger after the call.
    """

    losses: list[float]
    steps: int
    budget: ledger.Budget


@dataclasses.dataclass(frozen=True, slots=True)
class Checkpoint:
    """Everything needed to continue a run exactly.

    Attributes:
        model: Model state dict.
        optimizer: Optimizer state.
        budget: Ledger snapshot.
        ema: Averaged weights, or None when EMA is off.
        step: Optimizer steps already taken.
    """

    model: dict[str, torch.Tensor]
    optimizer: dict[str, object]
    budget: dict[str, int]
    ema: dict[str, torch.Tensor] | None
    step: int

    def write(self, path: pathlib.Path) -> None:
        """Writes the checkpoint atomically (temp file, then rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dataclasses.asdict(self), temporary)
        temporary.replace(path)

    @classmethod
    def read(cls, path: pathlib.Path, device: str) -> Checkpoint:
        """Reads a checkpoint.

        Raises:
            FileNotFoundError: If path does not exist.
            Error: If the file lacks fields of a checkpoint.
        """
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = torch.load(path, map_location=device)
        try:
            return cls(**raw)
        except TypeError as error:
            raise Error(f"{path} is not a valid checkpoint") from error


@dataclasses.dataclass(slots=True)
class Session:
    """The mutable state of one training run.

    Attributes:
        net: Model being trained.
        optimizer: Combined optimizer.
        average: Weight average, or None.
        budget: Epoch ledger.
        step: Optimizer steps taken so far.
    """

    net: model.TinyLM
    optimizer: optim.Optimizer
    average: Average | None
    budget: ledger.Budget
    step: int = 0

    @classmethod
    def open(
        cls, train: Train, net: model.TinyLM, budget: ledger.Budget
    ) -> Session:
        """Moves net to the device and builds the optimizer and average."""
        net.to(train.device).train()
        optimizer = optim.build(net, train.optimizer, train.lr, train.decay)
        average = Average(net, train.ema) if train.ema else None
        return cls(net, optimizer, average, budget)

    def checkpoint(self, path: pathlib.Path) -> None:
        """Writes the full state atomically."""
        Checkpoint(
            self.net.state_dict(),
            self.optimizer.state_dict(),
            self.budget.state(),
            self.average.state_dict() if self.average else None,
            self.step,
        ).write(path)

    def restore(self, path: pathlib.Path, device: str) -> None:
        """Loads a checkpoint written by checkpoint()."""
        saved = Checkpoint.read(path, device)
        self.net.load_state_dict(saved.model)
        self.optimizer.load_state_dict(saved.optimizer)
        self.budget = ledger.Budget.from_state(saved.budget)
        self.step = saved.step
        if self.average and saved.ema is not None:
            self.average.load_state_dict(saved.ema)

    def advance(self, train: Train, batch: torch.Tensor, rate: float) -> float:
        """Takes one optimizer step on batch and returns its loss.

        Raises:
            Error: If the loss is not finite.
            ledger.Error: If the batch would exceed the epoch cap.
        """
        self.budget.consume(batch.numel())
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        loss = train.loss(self.net, batch, self.step)
        if not torch.isfinite(loss):
            raise Error(f"non-finite loss at step {self.step}")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), train.clip)
        self.optimizer.step()
        if self.average:
            self.average.update(self.net)
        self.step += 1
        return float(loss.detach())

    def finish(self) -> None:
        """Makes the weight average (if any) the model's final weights."""
        if self.average:
            self.net.load_state_dict(self.average.shadow)


@dataclasses.dataclass(frozen=True, slots=True)
class Train:
    """Training hyperparameters and the loop that uses them.

    Attributes:
        seq_len: Input tokens per sequence.
        batch: Sequences per optimizer step.
        lr: Peak learning rate.
        floor: Final learning rate as a fraction of lr.
        warmup: Linear warmup length in steps.
        decay: Weight decay.
        clip: Maximum gradient norm.
        epochs: Passes over the data, 1 to 10.
        seed: Seed for data order and corruption.
        device: Torch device string.
        every: Steps between checkpoints.
        masked: Share of each batch trained with the masked objective.
        corrupt: Fraction of positions selected in masked rows.
        optimizer: "adamw" or "muon" (Muon on hidden matrices).
        schedule: "cosine" or "wsd" (warmup-stable-decay).
        tail: Share of training spent decaying under "wsd".
        shift: Shift chunk boundaries every epoch.
        ema: Decay of the weight average (0 disables it); the average
            becomes the model's final weights.
    """

    seq_len: int
    batch: int
    lr: float
    floor: float
    warmup: int
    decay: float
    clip: float
    epochs: int
    seed: int
    device: str
    every: int
    masked: float = 0.0
    corrupt: float = 0.15
    optimizer: str = "adamw"
    schedule: str = "cosine"
    tail: float = 0.2
    shift: bool = True
    ema: float = 0.0

    def __post_init__(self) -> None:
        """Validates every numeric and choice field."""
        for name in ("seq_len", "batch", "lr", "clip", "every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.epochs <= ledger.HARD_EPOCH_CAP:
            raise ValueError(f"epochs must be in [1, {ledger.HARD_EPOCH_CAP}]")
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("floor must be in [0, 1]")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.decay < 0:
            raise ValueError("decay must be non-negative")
        if not 0.0 <= self.masked <= 1.0:
            raise ValueError("masked must be in [0, 1]")
        if not 0.0 < self.corrupt < 1.0:
            raise ValueError("corrupt must be in (0, 1)")
        if self.optimizer not in optim.NAMES:
            raise ValueError(f"optimizer must be one of {optim.NAMES}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        if not 0.0 < self.tail < 1.0:
            raise ValueError("tail must be in (0, 1)")
        if not 0.0 <= self.ema < 1.0:
            raise ValueError("ema must be in [0, 1)")

    def rate(self, step: int, total: int, warmup: int) -> float:
        """Learning rate at a step.

        Linear warmup to lr, then cosine decay to lr * floor ("cosine") or a
        flat stable phase followed by a linear decay over the last `tail`
        share of training ("wsd").

        Raises:
            ValueError: If step < 0, total <= 0, or warmup is outside
                [0, total).
        """
        if step < 0 or total <= 0 or not 0 <= warmup < total:
            raise ValueError("require step >= 0 and 0 <= warmup < total")
        if step <= warmup:
            return self.lr * (step + 1) / (warmup + 1)
        low = self.lr * self.floor
        if self.schedule == "wsd":
            start = total * (1.0 - self.tail)
            if step <= start:
                return self.lr
            progress = min(1.0, (step - start) / (total - start))
            return self.lr + (low - self.lr) * progress
        progress = min(1.0, (step - warmup) / (total - warmup))
        return low + (self.lr - low) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    def chunk(self) -> int:
        """Tokens per training chunk: inputs plus the shifted target."""
        return self.seq_len + 1

    def batches(self, length: int, epoch: int) -> int:
        """Full batches an epoch yields from a stream of the given length."""
        first = packing.offset(epoch if self.shift else 0, self.chunk())
        return max(0, (length - first) // self.chunk()) // self.batch

    def loss(
        self, net: model.TinyLM, batch: torch.Tensor, step: int
    ) -> torch.Tensor:
        """Loss of one batch; corruption is seeded by step for exact resume."""
        if self.masked == 0.0:
            logits = net(batch[:, :-1])
            return functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
            )
        generator = torch.Generator()
        generator.manual_seed(self.seed * STRIDE + step)
        hybrid = objectives.Hybrid(self.masked, self.corrupt, MASK)
        return objectives.loss(net, batch, hybrid, generator)

    def run(  # noqa: PLR0913
        self,
        net: model.TinyLM,
        tokens: Token,
        out: pathlib.Path,
        *,
        budget: ledger.Budget | None = None,
        resume: bool = False,
        stop: int | None = None,
        after: collections.abc.Callable[[int], None] | None = None,
    ) -> Result:
        """Trains net on tokens, enforcing the epoch budget.

        Args:
            net: Model, trained in place.
            tokens: Training stream.
            out: Folder receiving checkpoint.pt.
            budget: Epoch ledger; defaults to self.epochs over the stream.
            resume: Continue from out/checkpoint.pt.
            stop: Checkpoint and return after this many total steps.
            after: Called with the epoch index after each finished epoch.

        Raises:
            ValueError: If seq_len exceeds the model or the stream is too
                short for one batch.
            FileNotFoundError: If resume is set without a checkpoint.
            Error: If a loss is not finite.
            ledger.Error: If the epoch cap would be exceeded.
        """
        if self.seq_len > net.config.max_seq_len:
            raise ValueError("seq_len exceeds model max_seq_len")
        length = len(tokens)
        if self.batches(length, 0) == 0:
            raise ValueError("stream too short for one batch")
        per_epoch = [self.batches(length, e) for e in range(self.epochs)]
        total = sum(per_epoch)
        session = Session.open(
            self, net, budget or ledger.Budget(length, self.epochs)
        )
        path = out / CHECKPOINT
        if resume:
            session.restore(path, self.device)
        losses: list[float] = []
        done = 0
        for epoch in range(self.epochs):
            if session.step >= done + per_epoch[epoch]:
                done += per_epoch[epoch]
                continue
            starts = packing.starts(
                length, self.chunk(), epoch, self.seed, shift=self.shift
            )
            for index in range(session.step - done, per_epoch[epoch]):
                if stop is not None and session.step >= stop:
                    session.checkpoint(path)
                    return Result(losses, session.step, session.budget)
                chosen = starts[index * self.batch : (index + 1) * self.batch]
                batch = tokens.rows(chosen, self.chunk()).to(self.device)
                rate = self.rate(
                    session.step, total, min(self.warmup, total - 1)
                )
                losses.append(session.advance(self, batch, rate))
                if session.step % self.every == 0:
                    session.checkpoint(path)
            done += per_epoch[epoch]
            if after:
                after(epoch)
        session.checkpoint(path)
        session.finish()
        return Result(losses, session.step, session.budget)

    def evaluate(self, net: model.TinyLM, tokens: Token) -> float:
        """Mean next-token cross-entropy over non-overlapping sequences.

        Raises:
            ValueError: If the stream is shorter than one sequence.
        """
        count = len(tokens) // self.chunk()
        if count == 0:
            raise ValueError("stream shorter than one sequence")
        net.to(self.device).eval()
        total = 0.0
        seen = 0
        starts = self.chunk() * torch.arange(count)
        with torch.no_grad():
            for first in range(0, count, self.batch):
                rows = tokens.rows(
                    starts[first : first + self.batch], self.chunk()
                )
                batch = rows.to(self.device)
                logits = net(batch[:, :-1])
                total += float(
                    functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        batch[:, 1:].reshape(-1),
                        reduction="sum",
                    )
                )
                seen += batch[:, 1:].numel()
        return total / seen
