"""Tests for portfolio valuation.

These functions take dicts and return dataclasses, so every case here is set
up by writing a literal. Nothing is patched, nothing is stubbed, and the whole
file runs in milliseconds.
"""

import pytest

from analytics import Portfolio, Position, format_portfolio, value_portfolio


def prices(**coins: float) -> dict[str, dict[str, float]]:
    """Build a /simple/price response body for the given coins."""
    return {coin: {"usd": price} for coin, price in coins.items()}


class TestValuation:
    def test_single_holding(self) -> None:
        result = value_portfolio({"bitcoin": 2.0}, prices(bitcoin=50_000.0))

        assert result.total == 100_000.0
        assert len(result.positions) == 1
        assert result.positions[0].coin_id == "bitcoin"
        assert result.positions[0].value == 100_000.0

    def test_totals_across_holdings(self) -> None:
        result = value_portfolio(
            {"bitcoin": 1.0, "ethereum": 10.0},
            prices(bitcoin=50_000.0, ethereum=3_000.0),
        )
        assert result.total == 80_000.0

    def test_fractional_quantities(self) -> None:
        result = value_portfolio({"bitcoin": 0.00042}, prices(bitcoin=50_000.0))
        assert result.total == pytest.approx(21.0)

    def test_empty_holdings(self) -> None:
        result = value_portfolio({}, {})
        assert result.total == 0.0
        assert result.positions == []
        assert result.unpriced == []

    def test_zero_quantity_is_a_real_position(self) -> None:
        """A holding of zero is still something the user asked about.

        Dropping it would make the output silently disagree with the input.
        """
        result = value_portfolio({"bitcoin": 0.0}, prices(bitcoin=50_000.0))
        assert len(result.positions) == 1
        assert result.total == 0.0


class TestSorting:
    def test_largest_position_first(self) -> None:
        result = value_portfolio(
            {"small": 1.0, "large": 1.0, "medium": 1.0},
            prices(small=10.0, large=1_000.0, medium=100.0),
        )
        assert [p.coin_id for p in result.positions] == ["large", "medium", "small"]

    def test_sorts_by_value_not_price(self) -> None:
        """A tiny amount of an expensive coin can be the smaller position."""
        result = value_portfolio(
            {"bitcoin": 0.001, "dogecoin": 100_000.0},
            prices(bitcoin=50_000.0, dogecoin=0.20),
        )
        assert result.positions[0].coin_id == "dogecoin"


class TestUnpricedCoins:
    def test_missing_coin_is_reported_not_dropped(self) -> None:
        result = value_portfolio(
            {"bitcoin": 1.0, "notacoin": 5.0}, prices(bitcoin=50_000.0)
        )

        assert result.unpriced == ["notacoin"]
        assert result.total == 50_000.0
        assert len(result.positions) == 1

    def test_null_price_counts_as_unpriced(self) -> None:
        """CoinGecko can return a coin with no value in the requested currency."""
        result = value_portfolio({"obscure": 1.0}, {"obscure": {"usd": None}})
        assert result.unpriced == ["obscure"]
        assert result.positions == []

    def test_all_unpriced(self) -> None:
        result = value_portfolio({"a": 1.0, "b": 2.0}, {})
        assert result.total == 0.0
        assert set(result.unpriced) == {"a", "b"}


class TestWeights:
    def test_weights_sum_to_one_hundred(self) -> None:
        result = value_portfolio(
            {"a": 1.0, "b": 1.0, "c": 1.0}, prices(a=100.0, b=300.0, c=600.0)
        )
        total_weight = sum(p.weight(result.total) for p in result.positions)
        assert total_weight == pytest.approx(100.0)

    def test_weight_of_zero_total_does_not_divide_by_zero(self) -> None:
        position = Position(
            coin_id="x", quantity=0.0, price=100.0, value=0.0, change_24h=None
        )
        assert position.weight(0.0) == 0.0


class TestWeightedChange:
    def test_larger_position_dominates(self) -> None:
        """A 10% move on 90% of the portfolio should outweigh -10% on the rest."""
        portfolio = Portfolio(
            positions=[
                Position("big", 1.0, 900.0, 900.0, change_24h=10.0),
                Position("small", 1.0, 100.0, 100.0, change_24h=-10.0),
            ],
            unpriced=[],
            total=1000.0,
        )
        assert portfolio.change_24h == pytest.approx(8.0)

    def test_none_when_no_position_reports_change(self) -> None:
        portfolio = Portfolio(
            positions=[Position("a", 1.0, 100.0, 100.0, change_24h=None)],
            unpriced=[],
            total=100.0,
        )
        assert portfolio.change_24h is None

    def test_positions_without_change_are_excluded_from_both_sides(self) -> None:
        """Missing data must not be treated as a 0% move.

        Counting the unknown position as flat would drag the average toward
        zero and understate a real move.
        """
        portfolio = Portfolio(
            positions=[
                Position("known", 1.0, 500.0, 500.0, change_24h=10.0),
                Position("unknown", 1.0, 500.0, 500.0, change_24h=None),
            ],
            unpriced=[],
            total=1000.0,
        )
        assert portfolio.change_24h == pytest.approx(10.0)


class TestFormatting:
    def test_includes_total_and_each_coin(self) -> None:
        result = value_portfolio(
            {"bitcoin": 1.0, "ethereum": 2.0},
            prices(bitcoin=50_000.0, ethereum=3_000.0),
        )
        output = format_portfolio(result)

        assert "56,000.00 USD" in output
        assert "bitcoin" in output
        assert "ethereum" in output

    def test_names_unpriced_coins(self) -> None:
        result = value_portfolio({"bitcoin": 1.0, "bad": 1.0}, prices(bitcoin=1.0))
        assert "bad" in format_portfolio(result)

    def test_message_when_nothing_could_be_priced(self) -> None:
        result = value_portfolio({"bad": 1.0}, {})
        assert "bad" in format_portfolio(result)

    def test_empty_portfolio_message(self) -> None:
        assert format_portfolio(value_portfolio({}, {})) == "No holdings provided."

    def test_currency_label_follows_argument(self) -> None:
        result = value_portfolio({"bitcoin": 1.0}, {"bitcoin": {"eur": 45_000.0}}, "eur")
        assert "EUR" in format_portfolio(result, "eur")
