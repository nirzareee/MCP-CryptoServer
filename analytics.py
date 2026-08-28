"""Portfolio and price-series computations.

Every function here is pure: it takes data and returns data, with no network
access, no clock, and no global state. That is deliberate. These are the parts
most likely to be wrong, and pure functions can be tested exhaustively without
mocking anything.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    """One valued holding within a portfolio."""

    coin_id: str
    quantity: float
    price: float
    value: float
    change_24h: float | None

    def weight(self, total: float) -> float:
        """This position's share of the portfolio, as a percentage."""
        return (self.value / total * 100) if total else 0.0


@dataclass(frozen=True)
class Portfolio:
    """The result of valuing a set of holdings."""

    positions: list[Position]
    unpriced: list[str]
    total: float

    @property
    def change_24h(self) -> float | None:
        """Weighted 24h change across positions that reported one.

        Returns None when no position has change data. Positions missing a
        change are excluded from both the numerator and the denominator, so
        the figure describes only the part of the portfolio it can see.
        """
        priced = [
            (p.value, p.change_24h) for p in self.positions if p.change_24h is not None
        ]
        if not priced:
            return None

        covered = sum(value for value, _ in priced)
        if covered == 0:
            return None

        return sum(value * change for value, change in priced) / covered


def value_portfolio(
    holdings: dict[str, float],
    prices: dict[str, dict[str, float]],
    currency: str = "usd",
) -> Portfolio:
    """Value holdings against a price map, largest position first.

    Coins absent from `prices`, or priced as null, are collected into
    `unpriced` rather than silently dropped: a total that omits a position
    without saying so is worse than no total at all.

    Args:
        holdings: Coin id to quantity held.
        prices: CoinGecko /simple/price response shape.
        currency: Currency key to read out of the price map.
    """
    positions: list[Position] = []
    unpriced: list[str] = []
    total = 0.0

    for coin_id, quantity in holdings.items():
        entry = prices.get(coin_id)
        if entry is None or entry.get(currency) is None:
            unpriced.append(coin_id)
            continue

        price = entry[currency]
        value = price * quantity

        if price is None:
            unpriced.append(coin_id)
            continue

        value = price * quantity
        total += value
        positions.append(
            Position(
                coin_id=coin_id,
                quantity=quantity,
                price=price,
                value=value,
                change_24h=entry.get(f"{currency}_24h_change"),
            )
        )

    positions.sort(key=lambda p: p.value, reverse=True)
    return Portfolio(positions=positions, unpriced=unpriced, total=total)


def format_portfolio(portfolio: Portfolio, currency: str = "usd") -> str:
    """Render a Portfolio as text for the model to read.

    Formatting is separated from computation so the numbers can be tested
    without asserting on prose.
    """
    unit = currency.upper()

    if not portfolio.positions:
        if portfolio.unpriced:
            return f"None of these could be priced: {', '.join(portfolio.unpriced)}"
        return "No holdings provided."

    lines = [f"Portfolio value: {portfolio.total:,.2f} {unit}"]

    overall = portfolio.change_24h
    if overall is not None:
        lines.append(f"24h change: {overall:+.2f}%")

    lines.append("")

    for p in portfolio.positions:
        line = (
            f"{p.coin_id}: {p.quantity:g} @ {p.price:,.2f} "
            f"= {p.value:,.2f} {unit} ({p.weight(portfolio.total):.1f}%)"
        )
        if p.change_24h is not None:
            line += f"  {p.change_24h:+.2f}% 24h"
        lines.append(line)

    if portfolio.unpriced:
        lines.append("")
        lines.append(
            f"Excluded from the total, could not be priced: "
            f"{', '.join(portfolio.unpriced)}"
        )

    return "\n".join(lines)
