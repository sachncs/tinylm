"""Enforces the rule that no corpus token is seen more than ten times."""

import dataclasses

HARD_EPOCH_CAP = 10


class Error(Exception):
    """Raised when training would exceed its allowed number of epochs."""


@dataclasses.dataclass
class Budget:
    """Counts every training token against max_epochs x unique_tokens.

    Every view of the data counts: re-masked, re-segmented, augmented and
    distillation passes all consume budget.

    Attributes:
        unique_tokens: Size of the cleaned corpus in tokens.
        max_epochs: Allowed passes, between 1 and HARD_EPOCH_CAP.
        consumed: Tokens consumed so far.
    """

    unique_tokens: int
    max_epochs: int = HARD_EPOCH_CAP
    consumed: int = 0

    def __post_init__(self) -> None:
        """Validates the corpus size and epoch cap."""
        if self.unique_tokens <= 0:
            raise ValueError("unique_tokens must be positive")
        if not 1 <= self.max_epochs <= HARD_EPOCH_CAP:
            raise ValueError(f"max_epochs must be in [1, {HARD_EPOCH_CAP}]")
        if not 0 <= self.consumed <= self.limit:
            raise ValueError("consumed must be within [0, limit]")

    @property
    def limit(self) -> int:
        """Total tokens that may be consumed."""
        return self.unique_tokens * self.max_epochs

    @property
    def remaining(self) -> int:
        """Tokens still available."""
        return self.limit - self.consumed

    @property
    def epochs_used(self) -> float:
        """Epochs consumed so far, as a fraction of the corpus."""
        return self.consumed / self.unique_tokens

    def consume(self, count: int) -> None:
        """Records count tokens as consumed.

        Raises:
            ValueError: If count is not positive.
            Error: If this would exceed the epoch cap; nothing
                is recorded in that case.
        """
        if count <= 0:
            raise ValueError("count must be positive")
        if count > self.remaining:
            raise Error(
                f"{self.consumed + count} tokens exceed {self.max_epochs} "
                f"epochs ({self.limit} tokens)"
            )
        self.consumed += count

    def state(self) -> dict[str, int]:
        """Returns a JSON-serialisable snapshot for checkpoints."""
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state: dict[str, int]) -> Budget:
        """Restores a budget from state().

        Raises:
            ValueError: If keys are missing or the state is overdrawn.
        """
        try:
            return cls(
                unique_tokens=state["unique_tokens"],
                max_epochs=state["max_epochs"],
                consumed=state["consumed"],
            )
        except KeyError as error:
            raise ValueError(f"incomplete state, missing {error}") from error
