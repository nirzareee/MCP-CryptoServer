"""Tests for price-series analysis.

Correlation and volatility have known closed-form answers on constructed
input, so most of these assert against values computed by hand rather than
against whatever the implementation happens to produce. A test that only
locks in current behaviour cannot catch a wrong formula.
"""

import statistics
from datetime import date

import pytest

from timeseries import (
    Drawdown,
    align,
    annualised_volatility,
    daily_returns,
    describe_correlation,
    max_drawdown,
    pearson_correlation,
    summarise_series,
    to_daily_closes,
)

DAY_MS = 86_400_000


def series(*prices: float, start_day: int = 0) -> dict[date, float]:
    """Build a daily close series over consecutive days from 2024-01-01."""
    from datetime import timedelta

    base = date(2024, 1, 1) + timedelta(days=start_day)
    return {base + timedelta(days=i): p for i, p in enumerate(prices)}


class TestDailyBucketing:
    def test_last_observation_of_the_day_wins(self) -> None:
        """Hourly data must collapse to one close, not several."""
        points = [
            [1_704_067_200_000, 100.0],  # 2024-01-01 00:00 UTC
            [1_704_070_800_000, 105.0],  # 2024-01-01 01:00 UTC
            [1_704_074_400_000, 110.0],  # 2024-01-01 02:00 UTC
        ]
        closes = to_daily_closes(points)

        assert len(closes) == 1
        assert closes[date(2024, 1, 1)] == 110.0

    def test_separates_distinct_days(self) -> None:
        points = [
            [1_704_067_200_000, 100.0],
            [1_704_067_200_000 + DAY_MS, 200.0],
        ]
        closes = to_daily_closes(points)

        assert len(closes) == 2
        assert closes[date(2024, 1, 2)] == 200.0

    def test_skips_null_prices(self) -> None:
        points = [[1_704_067_200_000, None], [1_704_067_200_000 + DAY_MS, 200.0]]
        assert len(to_daily_closes(points)) == 1  # type: ignore[arg-type]

    def test_skips_malformed_points(self) -> None:
        assert to_daily_closes([[1_704_067_200_000]]) == {}

    def test_empty_input(self) -> None:
        assert to_daily_closes([]) == {}


class TestAlignment:
    def test_keeps_only_shared_days(self) -> None:
        """A coin listed later must not have its data paired with the wrong days."""
        a = series(1.0, 2.0, 3.0, 4.0)
        b = series(10.0, 20.0, start_day=2)

        days, xs, ys = align(a, b)

        assert days == [date(2024, 1, 3), date(2024, 1, 4)]
        assert xs == [3.0, 4.0]
        assert ys == [10.0, 20.0]

    def test_no_overlap_gives_empty(self) -> None:
        days, xs, ys = align(series(1.0, 2.0), series(1.0, 2.0, start_day=10))
        assert days == [] and xs == [] and ys == []

    def test_output_is_date_ordered(self) -> None:
        a = {date(2024, 1, 3): 3.0, date(2024, 1, 1): 1.0, date(2024, 1, 2): 2.0}
        b = {d: v * 10 for d, v in a.items()}

        _, xs, _ = align(a, b)
        assert xs == [1.0, 2.0, 3.0]


class TestReturns:
    def test_simple_percentage_change(self) -> None:
        assert daily_returns([100.0, 110.0]) == [pytest.approx(0.10)]

    def test_one_fewer_return_than_prices(self) -> None:
        assert len(daily_returns([1.0, 2.0, 3.0, 4.0])) == 3

    def test_negative_moves(self) -> None:
        assert daily_returns([100.0, 90.0]) == [pytest.approx(-0.10)]

    def test_single_price_gives_nothing(self) -> None:
        assert daily_returns([100.0]) == []

    def test_zero_price_is_skipped(self) -> None:
        """Division by a zero previous price is undefined, not infinite."""
        assert daily_returns([0.0, 100.0]) == []


class TestCorrelation:
    def test_perfect_positive(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0, 4.0, 6.0, 8.0]
        assert pearson_correlation(xs, ys) == pytest.approx(1.0)

    def test_perfect_negative(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [4.0, 3.0, 2.0, 1.0]
        assert pearson_correlation(xs, ys) == pytest.approx(-1.0)

    def test_known_value(self) -> None:
        """Checked against a hand-computed coefficient, not against the code."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 1.0, 4.0, 3.0, 5.0]
        assert pearson_correlation(xs, ys) == pytest.approx(0.8, abs=0.01)

    def test_stays_within_bounds(self) -> None:
        """Float error must not produce a coefficient like 1.0000000002."""
        xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        result = pearson_correlation(xs, [x * 3 for x in xs])
        assert result is not None
        assert -1.0 <= result <= 1.0

    def test_none_for_constant_series(self) -> None:
        """A stablecoin has no variance, so correlation is undefined.

        Returning 0.0 here would assert independence, which is a different
        and unsupported claim.
        """
        assert pearson_correlation([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None

    def test_none_for_too_few_points(self) -> None:
        assert pearson_correlation([1.0, 2.0], [1.0, 2.0]) is None

    def test_none_for_mismatched_lengths(self) -> None:
        assert pearson_correlation([1.0, 2.0, 3.0], [1.0, 2.0]) is None

    def test_none_for_empty(self) -> None:
        assert pearson_correlation([], []) is None

    def test_matches_stdlib_implementation(self) -> None:
        """Cross-checked against statistics.correlation, an independent source."""
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 1.0, 4.0, 3.0, 5.0]

        assert pearson_correlation(xs, ys) == pytest.approx(
            statistics.correlation(xs, ys)
        )


class TestCorrelationDescription:
    @pytest.mark.parametrize(
        "coefficient,expected",
        [
            (0.95, "very strong"),
            (0.65, "strong"),
            (0.45, "moderate"),
            (0.25, "weak"),
        ],
    )
    def test_strength_labels(self, coefficient: float, expected: str) -> None:
        assert expected in describe_correlation(coefficient)

    def test_sign_determines_direction(self) -> None:
        assert "together" in describe_correlation(0.9)
        assert "opposite" in describe_correlation(-0.9)

    def test_near_zero_avoids_a_strength_label(self) -> None:
        assert "independent" in describe_correlation(0.05)


class TestDrawdown:
    def test_simple_decline(self) -> None:
        result = max_drawdown([100.0, 50.0])
        assert result.magnitude_pct == pytest.approx(50.0)
        assert result.peak_value == 100.0
        assert result.trough_value == 50.0

    def test_measured_from_running_peak_not_global_max(self) -> None:
        """A crash before a later all-time high still counts.

        Here the series falls 50% early, then rallies past the old peak. An
        implementation anchored on the global maximum would report the wrong
        decline.
        """
        result = max_drawdown([100.0, 50.0, 200.0, 180.0])
        assert result.magnitude_pct == pytest.approx(50.0)

    def test_finds_the_largest_of_several(self) -> None:
        result = max_drawdown([100.0, 90.0, 100.0, 40.0, 100.0, 80.0])
        assert result.magnitude_pct == pytest.approx(60.0)

    def test_monotonic_rise_has_no_drawdown(self) -> None:
        result = max_drawdown([1.0, 2.0, 3.0, 4.0])
        assert result.magnitude_pct == 0.0
        assert not result.is_meaningful

    def test_recovery_flag(self) -> None:
        assert max_drawdown([100.0, 50.0, 100.0]).recovered
        assert not max_drawdown([100.0, 50.0, 75.0]).recovered

    def test_attaches_dates_when_given(self) -> None:
        prices = [100.0, 50.0, 60.0]
        dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]

        result = max_drawdown(prices, dates)

        assert result.peak_date == date(2024, 1, 1)
        assert result.trough_date == date(2024, 1, 2)

    def test_empty_series(self) -> None:
        result = max_drawdown([])
        assert result == Drawdown(0.0, 0.0, 0.0, None, None, recovered=False)


class TestVolatility:
    def test_flat_series_has_no_volatility(self) -> None:
        assert annualised_volatility([0.0, 0.0, 0.0]) == 0.0

    def test_scales_by_square_root_of_year(self) -> None:
        """A 1% daily standard deviation annualises to 1% * sqrt(365) ~= 19.1%."""
        returns = [0.01, -0.01, 0.01, -0.01]
        # sample sd of this series is exactly 0.01154...
        result = annualised_volatility(returns)
        assert result == pytest.approx(22.05, abs=0.1)

    def test_larger_swings_give_larger_figure(self) -> None:
        calm = annualised_volatility([0.001, -0.001, 0.001, -0.001])
        wild = annualised_volatility([0.10, -0.10, 0.10, -0.10])
        assert wild > calm

    def test_too_few_points(self) -> None:
        assert annualised_volatility([0.05]) == 0.0
        assert annualised_volatility([]) == 0.0


class TestSummary:
    def test_computes_period_return(self) -> None:
        stats = summarise_series("bitcoin", series(100.0, 110.0, 120.0))
        assert stats is not None
        assert stats.total_return_pct == pytest.approx(20.0)
        assert stats.observations == 3

    def test_none_for_series_too_short(self) -> None:
        assert summarise_series("bitcoin", series(100.0)) is None
        assert summarise_series("bitcoin", {}) is None

    def test_includes_drawdown(self) -> None:
        stats = summarise_series("bitcoin", series(100.0, 40.0, 80.0))
        assert stats is not None
        assert stats.drawdown.magnitude_pct == pytest.approx(60.0)
