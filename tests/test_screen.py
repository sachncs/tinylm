"""Tests for the factor-screening design, effect analysis and Pareto front."""

import math

import pytest
import torch

from tinylm.screen import Design, Factor, Factorial, Point, Screen, front


class TestDesign:
    def test_pb24_is_balanced_orthogonal_with_24_runs_and_23_columns(self):
        design = Design.pb24()
        matrix = torch.tensor(design.rows, dtype=torch.float32)
        assert matrix.shape == (24, 23)
        assert bool((matrix.abs() == 1).all())
        assert torch.equal(matrix.sum(0), torch.zeros(23))
        assert torch.equal(matrix.T @ matrix, 24 * torch.eye(23))

    def test_rejects_ragged_or_non_binary_rows(self):
        with pytest.raises(ValueError, match="rows"):
            Design(((1, -1), (1,)))
        with pytest.raises(ValueError, match="levels"):
            Design(((1, 0), (-1, 1)))
        with pytest.raises(ValueError, match="rows"):
            Design(())


def factors(n: int) -> tuple[Factor, ...]:
    return tuple(
        Factor(f"f{i}", (f"train.lr={i}e-3",), (f"train.lr={i}e-2",))
        for i in range(n)
    )


class TestScreen:
    def test_runs_apply_the_high_or_low_level_of_each_factor(self):
        screen = Screen(Design.pb24(), factors(3), dummies=2)
        runs = screen.runs()
        assert len(runs) == 24 and runs[0].name == "x00"
        first = Design.pb24().rows[0]
        for index in range(3):
            expected = (
                factors(3)[index].high
                if first[index] == 1
                else factors(3)[index].low
            )
            assert set(expected) <= set(runs[0].sets)

    def test_requires_enough_columns_and_unique_names(self):
        with pytest.raises(ValueError, match="columns"):
            Screen(Design.pb24(), factors(22), dummies=3)
        with pytest.raises(ValueError, match="unique"):
            Screen(Design.pb24(), (factors(1)[0], factors(1)[0]), dummies=2)
        with pytest.raises(ValueError, match="dummies"):
            Screen(Design.pb24(), factors(3), dummies=1)

    def test_recovers_planted_effects_and_flags_dummies_as_null(self):
        screen = Screen(Design.pb24(), factors(5), dummies=3)
        generator = torch.Generator()
        generator.manual_seed(0)
        rows = torch.tensor(Design.pb24().rows, dtype=torch.float32)
        response = (
            5.0
            + 2.0 * rows[:, 1]
            - 1.5 * rows[:, 3]
            + 0.05 * torch.randn(24, generator=generator)
        )
        found = screen.analyse([float(y) for y in response])
        by_name = {e.name: e for e in found.effects}
        assert by_name["f1"].effect == pytest.approx(4.0, abs=0.2)
        assert by_name["f3"].effect == pytest.approx(-3.0, abs=0.2)
        assert by_name["f1"].significant and by_name["f3"].significant
        assert not any(by_name[n].significant for n in ("f0", "f2", "f4"))
        assert found.noise < 0.1 and len(found.dummy_effects) == 3

    def test_noise_free_planted_effect_is_significant_with_infinite_ratio(self):
        screen = Screen(Design.pb24(), factors(2), dummies=3)
        rows = torch.tensor(Design.pb24().rows, dtype=torch.float32)
        found = screen.analyse([float(y) for y in 1.0 + rows[:, 0]])
        assert found.noise == 0.0
        first = found.effects[0]
        assert first.significant and math.isinf(first.ratio)
        assert not found.effects[1].significant

    def test_analyse_rejects_wrong_length_and_non_finite(self):
        screen = Screen(Design.pb24(), factors(2), dummies=3)
        with pytest.raises(ValueError, match="24"):
            screen.analyse([1.0] * 5)
        with pytest.raises(ValueError, match="finite"):
            screen.analyse([1.0] * 23 + [float("nan")])


class TestFront:
    def test_dominated_points_are_removed(self):
        points = (
            Point("small-good", 0.70, 0.80, 4.0),
            Point("big-same", 0.70, 0.80, 9.0),
            Point("big-better", 0.75, 0.85, 9.0),
            Point("worse", 0.60, 0.70, 5.0),
        )
        kept = front(points)
        assert {p.name for p in kept.front} == {"small-good", "big-better"}
        assert {p.name for p in kept.dominated} == {"big-same", "worse"}

    def test_exact_ties_keep_one_and_it_is_deterministic(self):
        a, b = Point("a", 0.7, 0.8, 4.0), Point("b", 0.7, 0.8, 4.0)
        assert [p.name for p in front((b, a)).front] == ["a"]
        assert [p.name for p in front((a, b)).dominated] == ["b"]

    def test_trade_offs_all_survive(self):
        points = (
            Point("tiny", 0.60, 0.7, 2.0),
            Point("mid", 0.70, 0.8, 4.0),
            Point("large", 0.80, 0.9, 8.0),
        )
        assert len(front(points).front) == 3

    def test_precision_none_counts_as_worst_and_empty_is_rejected(self):
        with pytest.raises(ValueError, match="points"):
            front(())
        kept = front((Point("a", 0.7, None, 4.0), Point("b", 0.7, 0.5, 4.0)))
        assert [p.name for p in kept.front] == ["b"]

    def test_rejects_bad_values(self):
        with pytest.raises(ValueError, match="size"):
            Point("x", 0.5, 0.5, 0.0)
        with pytest.raises(ValueError, match="accuracy"):
            Point("x", 1.5, 0.5, 1.0)


class TestFactorial:
    def test_full_design_has_every_combination_once(self):
        design = Factorial(factors(3))
        runs = design.runs()
        assert len(runs) == 8
        assert len({run.sets for run in runs}) == 8

    def test_main_effects_and_an_interaction_are_recovered(self):
        design = Factorial(factors(4))
        generator = torch.Generator()
        generator.manual_seed(0)
        rows = torch.tensor(design.rows, dtype=torch.float32)
        a, b = rows[:, 0], rows[:, 1]
        response = (
            10.0
            + 3.0 * a
            - 2.0 * b
            + 4.0 * a * b
            + 0.05 * torch.randn(16, generator=generator)
        )
        found = design.analyse([float(y) for y in response])
        by_name = {t.name: t for t in found.terms}
        assert by_name["f0"].effect == pytest.approx(6.0, abs=0.3)
        assert by_name["f1"].effect == pytest.approx(-4.0, abs=0.3)
        assert by_name["f0*f1"].effect == pytest.approx(8.0, abs=0.3)
        flagged = {t.name for t in found.terms if t.significant}
        assert flagged == {"f0", "f1", "f0*f1"}
        assert len(found.terms) == 4 + 6

    def test_a_pure_interaction_is_visible_where_main_effects_are_zero(self):
        design = Factorial(factors(3))
        rows = torch.tensor(design.rows, dtype=torch.float32)
        response = [float(y) for y in 5.0 + 2.0 * rows[:, 0] * rows[:, 1]]
        found = design.analyse(response)
        by_name = {t.name: t for t in found.terms}
        assert by_name["f0"].effect == pytest.approx(0.0, abs=1e-9)
        assert by_name["f1"].effect == pytest.approx(0.0, abs=1e-9)
        assert by_name["f0*f1"].effect == pytest.approx(4.0)
        assert by_name["f0*f1"].significant

    def test_pure_noise_flags_nothing_large(self):
        design = Factorial(factors(4))
        generator = torch.Generator()
        generator.manual_seed(5)
        noise = [float(y) for y in torch.randn(16, generator=generator)]
        assert sum(t.significant for t in design.analyse(noise).terms) <= 1

    def test_validation(self):
        with pytest.raises(ValueError, match="factors"):
            Factorial(factors(1))
        with pytest.raises(ValueError, match="factors"):
            Factorial(factors(9))
        with pytest.raises(ValueError, match="unique"):
            Factorial((factors(1)[0], factors(1)[0]))
        with pytest.raises(ValueError, match="16"):
            Factorial(factors(4)).analyse([1.0] * 5)
        with pytest.raises(ValueError, match="finite"):
            Factorial(factors(4)).analyse([float("inf")] + [1.0] * 15)


def test_foldover_doubles_the_runs_and_keeps_the_design_orthogonal():
    folded = Design.pb24().foldover()
    matrix = torch.tensor(folded.rows, dtype=torch.float32)
    assert matrix.shape == (48, 23)
    assert torch.equal(matrix[24:], -matrix[:24])
    assert torch.equal(matrix.T @ matrix, 48 * torch.eye(23))
