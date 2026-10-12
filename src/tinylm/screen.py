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

    def foldover(self) -> Design:
        """The design plus its mirror image.

        Adding every run with all levels reversed removes the aliasing of
        main effects with two-factor interactions.
        """
        mirrored = tuple(tuple(-level for level in row) for row in self.rows)
        return Design((*self.rows, *mirrored))


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


MIN_FACTORIAL = 2
MAX_FACTORIAL = 8
LENTH_T = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}  # two-sided 5% Student t by degrees of freedom (floor used: conservative)


@dataclasses.dataclass(frozen=True, slots=True)
class Term:
    """A main effect or two-factor interaction in a full factorial.

    Attributes:
        name: Factor name, or "a*b" for an interaction.
        effect: Mean response at +1 minus at -1 of the term's contrast.
        significant: Whether |effect| exceeds Lenth's margin of error.
    """

    name: str
    effect: float
    significant: bool


@dataclasses.dataclass(frozen=True, slots=True)
class Lenth:
    """Main effects and interactions with Lenth's noise estimate.

    Attributes:
        terms: Every main effect and two-factor interaction.
        pse: Lenth's pseudo standard error of an effect.
        margin: Margin of error; effects beyond it are significant.
    """

    terms: tuple[Term, ...]
    pse: float
    margin: float


@dataclasses.dataclass(frozen=True, slots=True)
class Factorial:
    """A full two-level factorial that can see interactions.

    All 2^k combinations of the factors are run once. Without replicates the
    noise is estimated with Lenth's method, which assumes most effects are
    null. Use it on the few factors a screen flags, to find combinations that
    only work together.

    Attributes:
        factors: Between 2 and 8 factors.
    """

    factors: tuple[Factor, ...]

    def __post_init__(self) -> None:
        """Validates the number and names of the factors."""
        if not MIN_FACTORIAL <= len(self.factors) <= MAX_FACTORIAL:
            raise ValueError(
                f"factors must number {MIN_FACTORIAL} to {MAX_FACTORIAL}"
            )
        names = [f.name for f in self.factors]
        if len(set(names)) != len(names):
            raise ValueError("factor names must be unique")

    @property
    def rows(self) -> tuple[tuple[int, ...], ...]:
        """All combinations of +1/-1 levels, first factor varying fastest."""
        count = len(self.factors)
        return tuple(
            tuple(1 if (index >> column) & 1 else -1 for column in range(count))
            for index in range(2**count)
        )

    def runs(self) -> tuple[Run, ...]:
        """Expands the design into named runs with their overrides."""
        out: list[Run] = []
        for index, row in enumerate(self.rows):
            sets = itertools.chain.from_iterable(
                factor.high if level == 1 else factor.low
                for level, factor in zip(row, self.factors, strict=True)
            )
            out.append(Run(f"y{index:03d}", tuple(sets)))
        return tuple(out)

    def analyse(self, response: list[float]) -> Lenth:
        """Estimates all main effects and two-factor interactions.

        Raises:
            ValueError: If the response length differs from 2^k or contains
                a non-finite value.
        """
        rows = self.rows
        if len(response) != len(rows):
            raise ValueError(
                f"expected {len(rows)} responses, got {len(response)}"
            )
        if not all(math.isfinite(y) for y in response):
            raise ValueError("responses must be finite")
        names = [f.name for f in self.factors]
        contrasts: list[tuple[str, tuple[int, ...]]] = [
            (names[i], tuple(row[i] for row in rows)) for i in range(len(names))
        ]
        for i, j in itertools.combinations(range(len(names)), 2):
            contrasts.append(
                (
                    f"{names[i]}*{names[j]}",
                    tuple(row[i] * row[j] for row in rows),
                )
            )
        effects = [
            (
                name,
                sum(y * c for y, c in zip(response, column, strict=True))
                / (len(rows) / 2),
            )
            for name, column in contrasts
        ]
        absolute = sorted(abs(e) for _, e in effects)
        median = (
            absolute[len(absolute) // 2]
            if len(absolute) % 2
            else (
                absolute[len(absolute) // 2 - 1] + absolute[len(absolute) // 2]
            )
            / 2
        )
        trimmed = sorted(a for a in absolute if a < 2.5 * 1.5 * median)
        if trimmed:
            mid = len(trimmed) // 2
            robust = (
                trimmed[mid]
                if len(trimmed) % 2
                else (trimmed[mid - 1] + trimmed[mid]) / 2
            )
        else:
            robust = 0.0
        pse = 1.5 * robust
        degrees = max(1, len(effects) // 3)
        t = LENTH_T[max(k for k in LENTH_T if k <= degrees)]
        margin = t * pse
        return Lenth(
            tuple(
                Term(n, e, abs(e) > margin and abs(e) > TINY)
                for n, e in effects
            ),
            pse,
            margin,
        )
