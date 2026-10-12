"""Bootstrap confidence intervals for benchmark accuracy."""

import dataclasses

import torch

MIN_RESAMPLES = 100


@dataclasses.dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate with its confidence interval.

    Attributes:
        estimate: Observed value.
        low: Lower end of the interval.
        high: Upper end of the interval.
        level: Coverage, e.g. 0.95.
    """

    estimate: float
    low: float
    high: float
    level: float

    def __post_init__(self) -> None:
        """Validates ordering and level."""
        if not self.low <= self.high:
            raise ValueError("interval bounds are out of order")
        if not 0.0 < self.level < 1.0:
            raise ValueError("level must be in (0, 1)")


@dataclasses.dataclass(frozen=True, slots=True)
class Bootstrap:
    """Resamples benchmark items to put uncertainty on macro accuracy.

    Items are resampled with replacement inside each paradigm and the
    paradigm accuracies are averaged with equal weight, mirroring how the
    headline score is computed.

    Attributes:
        resamples: Number of bootstrap resamples (at least 100).
        level: Interval coverage in (0, 1).
        seed: Seed of the resampling.
    """

    resamples: int = 1000
    level: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        """Validates settings."""
        if self.resamples < MIN_RESAMPLES:
            raise ValueError(f"resamples must be at least {MIN_RESAMPLES}")
        if not 0.0 < self.level < 1.0:
            raise ValueError("level must be in (0, 1)")

    def macro(self, groups: dict[str, torch.Tensor]) -> Interval:
        """Interval for the equal-weight mean of per-group item means."""
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        draws = torch.zeros(self.resamples)
        observed = 0.0
        for values in groups.values():
            count = values.numel()
            index = torch.randint(
                0, count, (self.resamples, count), generator=generator
            )
            draws += values[index].mean(dim=1)
            observed += float(values.mean())
        draws /= len(groups)
        tail = (1.0 - self.level) / 2.0
        low, high = torch.quantile(draws, torch.tensor([tail, 1.0 - tail]))
        return Interval(
            observed / len(groups), float(low), float(high), self.level
        )

    def mean(self, outcomes: dict[str, list[bool]]) -> Interval:
        """Interval for macro accuracy given correct/incorrect per item.

        Raises:
            ValueError: If there are no groups or a group is empty.
        """
        if not outcomes:
            raise ValueError("outcomes needs at least one of groups")
        if any(not values for values in outcomes.values()):
            raise ValueError("a group is empty")
        return self.macro(
            {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in outcomes.items()
            }
        )

    def paired(
        self, first: dict[str, list[bool]], second: dict[str, list[bool]]
    ) -> Interval:
        """Interval for macro accuracy of `first` minus `second`.

        Both models must have been scored on the same items in the same
        order, so each item is resampled once and compared directly.

        Raises:
            ValueError: If the groups or their lengths differ.
        """
        if first.keys() != second.keys():
            raise ValueError("both models need the same groups")
        if any(len(first[k]) != len(second[k]) for k in first):
            raise ValueError("groups must have the same length for both models")
        if not first or any(not values for values in first.values()):
            raise ValueError("groups must be non-empty")
        return self.macro(
            {
                k: torch.tensor(first[k], dtype=torch.float32)
                - torch.tensor(second[k], dtype=torch.float32)
                for k in first
            }
        )
