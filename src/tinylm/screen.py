"""Factor screening: two-level designs, effect analysis and Pareto fronts."""

import dataclasses
import itertools
import math

GENERATOR_24 = "+++++-+-++--++--+-+----"
SIGNIFICANT = 2.353  # two-sided 10% Student t, 3 degrees of freedom
MIN_DUMMIES = 2
TINY = 1e-12


@dataclasses.dataclass(frozen=True, slots=True)
class Design:
    """A two-level screening design.

    Attributes:
        rows: One tuple of +1/-1 levels per run, one entry per column.
    """

    rows: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """Validates that rows are rectangular and use only +1 and -1."""
        if not self.rows:
            raise ValueError("rows must not be empty")
        width = len(self.rows[0])
        if any(len(row) != width for row in self.rows):
            raise ValueError("rows must all have the same length")
        if any(level not in (-1, 1) for row in self.rows for level in row):
            raise ValueError("levels must be +1 or -1")

    @classmethod
    def pb24(cls) -> Design:
        """The 24-run Plackett-Burman design for up to 23 factors."""
        base = tuple(1 if c == "+" else -1 for c in GENERATOR_24)
        shifts = [base[-i:] + base[:-i] for i in range(len(base))]
        return cls((*shifts, (-1,) * len(base)))


@dataclasses.dataclass(frozen=True, slots=True)
class Factor:
    """One thing to vary, with the overrides for its low and high level.

    Attributes:
        name: Unique factor name.
        low: Override strings applied at the low level.
        high: Override strings applied at the high level.
    """

    name: str
    low: tuple[str, ...]
    high: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """One run of a design: its name and the overrides to apply."""

    name: str
    sets: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Effect:
    """Main effect of one factor on the response.

    Attributes:
        name: Factor name.
        effect: Mean response at the high level minus at the low level.
        ratio: |effect| divided by the noise estimated from dummy columns.
        significant: Whether ratio reaches the critical value.
    """

    name: str
    effect: float
    ratio: float
    significant: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Analysis:
    """Main effects together with the noise measured by the dummy columns."""

    effects: tuple[Effect, ...]
    dummy_effects: tuple[float, ...]
    noise: float


@dataclasses.dataclass(frozen=True, slots=True)
class Screen:
    """A design whose first columns are factors and the next are dummies.

    Dummy columns change nothing in the experiment, so their apparent
    effects measure noise within the same set of runs.

    Attributes:
        design: Two-level design.
        factors: Factors assigned to the first columns.
        dummies: Number of columns after the factors left as dummies.
    """

    design: Design
    factors: tuple[Factor, ...]
    dummies: int

    def __post_init__(self) -> None:
        """Validates counts and name uniqueness."""
        if self.dummies < MIN_DUMMIES:
            raise ValueError(f"dummies must be at least {MIN_DUMMIES}")
        if len(self.factors) + self.dummies > len(self.design.rows[0]):
            raise ValueError("factors plus dummies exceed the design columns")
        names = [f.name for f in self.factors]
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique")

    def runs(self) -> tuple[Run, ...]:
        """Expands the design into named runs with their overrides."""
        out: list[Run] = []
        for index, row in enumerate(self.design.rows):
            sets = itertools.chain.from_iterable(
                factor.high if row[column] == 1 else factor.low
                for column, factor in enumerate(self.factors)
            )
            out.append(Run(f"x{index:02d}", tuple(sets)))
        return tuple(out)

    def analyse(self, response: list[float]) -> Analysis:
        """Estimates main effects and which exceed the dummy-column noise.

        Raises:
            ValueError: If the response length differs from the number of
                runs or contains a non-finite value.
        """
        runs = len(self.design.rows)
        if len(response) != runs:
            raise ValueError(f"expected {runs} responses, got {len(response)}")
        if not all(math.isfinite(y) for y in response):
            raise ValueError("responses must be finite")

        def effect(column: int) -> float:
            high = [
                y
                for y, r in zip(response, self.design.rows, strict=True)
                if r[column] == 1
            ]
            low = [
                y
                for y, r in zip(response, self.design.rows, strict=True)
                if r[column] == -1
            ]
            return sum(high) / len(high) - sum(low) / len(low)

        first = len(self.factors)
        dummy = tuple(effect(first + i) for i in range(self.dummies))
        noise = math.sqrt(sum(e * e for e in dummy) / len(dummy))
        effects: list[Effect] = []
        for column, factor in enumerate(self.factors):
            value = effect(column)
            if noise > 0:
                ratio = abs(value) / noise
            else:
                ratio = math.inf if abs(value) > TINY else 0.0
            effects.append(
                Effect(factor.name, value, ratio, ratio >= SIGNIFICANT)
            )
        return Analysis(tuple(effects), dummy, noise)


@dataclasses.dataclass(frozen=True, slots=True)
class Point:
    """A candidate in the accuracy / precision / size trade-off.

    Attributes:
        name: Candidate name.
        accuracy: Benchmark accuracy in [0, 1] (higher is better).
        precision: Confident-correct precision in [0, 1], or None if the
            candidate had no confident answers (counted as the worst).
        size: Parameters in millions (lower is better).
    """

    name: str
    accuracy: float
    precision: float | None
    size: float

    def __post_init__(self) -> None:
        """Validates ranges."""
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0.0 <= self.accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        if self.precision is not None and not 0.0 <= self.precision <= 1.0:
            raise ValueError("precision must be in [0, 1]")

    def worst(self) -> float:
        """Precision with None mapped below every real value."""
        return -1.0 if self.precision is None else self.precision


@dataclasses.dataclass(frozen=True, slots=True)
class Front:
    """Non-dominated points and the points they dominate."""

    front: tuple[Point, ...]
    dominated: tuple[Point, ...]


def beats(p: Point, q: Point) -> bool:
    """True if p dominates q (no worse everywhere, better somewhere).

    Exact ties are broken by name so exactly one of two twins survives.
    """
    no_worse = (
        p.accuracy >= q.accuracy and p.worst() >= q.worst() and p.size <= q.size
    )
    same = (
        p.accuracy == q.accuracy and p.worst() == q.worst() and p.size == q.size
    )
    return no_worse and (p.name < q.name if same else True)


def position(point: Point) -> tuple[float, str]:
    """Sort key: smallest first, ties by name."""
    return (point.size, point.name)


def front(points: tuple[Point, ...]) -> Front:
    """Splits points into the Pareto front and the dominated rest.

    Raises:
        ValueError: If points is empty.
    """
    if not points:
        raise ValueError("points must not be empty")
    kept = [
        p for p in points if not any(beats(q, p) for q in points if q is not p)
    ]
    gone = [p for p in points if p not in kept]
    return Front(
        tuple(sorted(kept, key=position)), tuple(sorted(gone, key=position))
    )
