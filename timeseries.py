"""Price-series analysis: returns, correlation, drawdown, volatility.

Separate from `analytics` because the concerns differ. That module answers
"what is this portfolio worth right now" from a single price snapshot. This
one answers questions about how prices moved over time, which brings its own
problems: observations that do not line up, series too short to say anything
about, and statistics that are undefined on degenerate input.

Like `analytics`, everything here is pure. No network, no clock, no globals.
"""

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime

# Below this many paired observations, a correlation coefficient is noise.
# Three is the arithmetic minimum; the real threshold is a judgement call.
MIN_OBSERVATIONS_FOR_CORRELATION = 3

# Correlations computed on fewer than this many days are reported with a
# caveat rather than presented as a finding.
RELIABLE_OBSERVATION_COUNT = 30

# Crypto markets do not close, so a year is 365 periods of daily data.
# Equity volatility conventionally annualises with 252 trading days; using
# that here would understate crypto volatility by about 18%.
TRADING_DAYS_PER_YEAR = 365


@dataclass(frozen=True)
class Drawdown:
    """The largest peak-to-trough decline in a price series."""

    magnitude_pct: float
    peak_value: float
    trough_value: float
    peak_date: date | None
    trough_date: date | None
    recovered: bool

    @property
    def is_meaningful(self) -> bool:
        """Whether the decline is large enough to be worth reporting."""
        return self.magnitude_pct > 0.01


@dataclass(frozen=True)
class SeriesStats:
    """Summary statistics for one coin's price history."""

    coin_id: str
    observations: int
    start_price: float
    end_price: float
    total_return_pct: float
    annualised_volatility_pct: float
    drawdown: Drawdown


def to_daily_closes(points: list[list[float]]) -> dict[date, float]:
    """Collapse raw CoinGecko market_chart points into one price per UTC day.

    CoinGecko varies its resolution by range: roughly five-minute data for a
    single day, hourly out to ninety days, daily beyond that. Correlating an
    hourly series against a daily one would compare different things, so
    everything is bucketed to a day first and the last observation in each
    bucket wins.

    Args:
        points: `[[unix_millis, price], ...]` as returned by CoinGecko.

    Returns:
        Map of UTC date to that day's closing price.
    """
    closes: dict[date, float] = {}

    for point in points:
        if len(point) < 2:
            continue
        timestamp_ms, price = point[0], point[1]
        if price is None:
            continue
        day = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).date()
        closes[day] = float(price)

    return closes


def align(
    a: dict[date, float], b: dict[date, float]
) -> tuple[list[date], list[float], list[float]]:
    """Restrict two daily series to the days both cover.

    Two coins can have different histories: one listed later, one missing a
    day of data. Zipping the raw lists would silently pair Monday's bitcoin
    with Tuesday's ethereum and produce a correlation that means nothing.

    Returns:
        `(dates, a_values, b_values)`, sorted by date, all the same length.
    """
    shared = sorted(set(a) & set(b))
    return shared, [a[d] for d in shared], [b[d] for d in shared]


def daily_returns(prices: list[float]) -> list[float]:
    """Convert a price series to simple period-over-period returns.

    Correlation is computed on returns rather than prices. Two coins in a
    rising market both trend upward, so their *prices* correlate strongly
    almost regardless of behaviour — a spurious result driven by shared trend.
    Returns strip the trend out and compare day-to-day movement, which is the
    question being asked.

    Days where the previous price is zero are skipped; the return is undefined.
    """
    returns = []
    for previous, current in zip(prices, prices[1:], strict=False):
        if previous == 0:
            continue
        returns.append((current - previous) / previous)
    return returns


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient, or None where it is undefined.

    Returns None when the series are too short, differ in length, or when
    either has zero variance. That last case is not hypothetical: a stablecoin
    holds its value by design, so its returns are all near zero and the
    denominator collapses. Returning None is honest; returning 0.0 would claim
    "no relationship" when the truth is "unanswerable from this data".
    """
    if len(xs) != len(ys):
        return None
    if len(xs) < MIN_OBSERVATIONS_FOR_CORRELATION:
        return None

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)

    denominator = math.sqrt(variance_x * variance_y)
    if denominator == 0:
        return None

    coefficient = covariance / denominator

    # Floating-point error can push a perfect correlation a hair outside
    # [-1, 1], which looks like a bug to anyone reading the output.
    return max(-1.0, min(1.0, coefficient))


def describe_correlation(coefficient: float) -> str:
    """Plain-language reading of a correlation coefficient.

    Thresholds are conventional rather than principled — there is no natural
    boundary between "moderate" and "strong". They exist to stop a bare
    decimal being over-read.
    """
    strength = abs(coefficient)

    if strength >= 0.8:
        label = "very strong"
    elif strength >= 0.6:
        label = "strong"
    elif strength >= 0.4:
        label = "moderate"
    elif strength >= 0.2:
        label = "weak"
    else:
        return "essentially independent day-to-day movement"

    direction = "together" if coefficient > 0 else "in opposite directions"
    return f"{label} tendency to move {direction}"


def max_drawdown(prices: list[float], dates: list[date] | None = None) -> Drawdown:
    """Largest peak-to-trough decline, as a positive percentage.

    Walks the series once, tracking the running maximum. A drawdown is
    measured from the highest point *seen so far*, not from the global
    maximum: a fall that happens before the all-time high is still a fall
    investors lived through.
    """
    if not prices:
        return Drawdown(0.0, 0.0, 0.0, None, None, recovered=False)

    peak = prices[0]
    peak_index = 0
    worst_pct = 0.0
    worst_peak = prices[0]
    worst_trough = prices[0]
    worst_peak_index = 0
    worst_trough_index = 0

    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            peak_index = i
            continue

        if peak == 0:
            continue

        decline = (peak - price) / peak * 100
        if decline > worst_pct:
            worst_pct = decline
            worst_peak = peak
            worst_trough = price
            worst_peak_index = peak_index
            worst_trough_index = i

    def date_at(index: int) -> date | None:
        return dates[index] if dates and index < len(dates) else None

    return Drawdown(
        magnitude_pct=worst_pct,
        peak_value=worst_peak,
        trough_value=worst_trough,
        peak_date=date_at(worst_peak_index),
        trough_date=date_at(worst_trough_index),
        recovered=prices[-1] >= worst_peak,
    )


def annualised_volatility(returns: list[float]) -> float:
    """Standard deviation of daily returns, scaled to a yearly figure.

    Uses the sample standard deviation (n-1 denominator) because these
    returns are a sample of the coin's behaviour, not its whole population.

    Scaling by the square root of the period count is the standard
    convention. It assumes returns are independent day to day — which is not
    strictly true of any real market, and is worth knowing when quoting the
    number.
    """
    n = len(returns)
    if n < 2:
        return 0.0

    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    daily_sd = math.sqrt(variance)

    return daily_sd * math.sqrt(TRADING_DAYS_PER_YEAR) * 100


def summarise_series(coin_id: str, closes: dict[date, float]) -> SeriesStats | None:
    """Compute summary statistics for one coin. None if the series is too short."""
    if len(closes) < 2:
        return None

    days = sorted(closes)
    prices = [closes[d] for d in days]
    returns = daily_returns(prices)

    start, end = prices[0], prices[-1]
    total_return = ((end - start) / start * 100) if start else 0.0

    return SeriesStats(
        coin_id=coin_id,
        observations=len(prices),
        start_price=start,
        end_price=end,
        total_return_pct=total_return,
        annualised_volatility_pct=annualised_volatility(returns),
        drawdown=max_drawdown(prices, days),
    )


def format_comparison(
    a: SeriesStats,
    b: SeriesStats,
    coefficient: float | None,
    shared_days: int,
) -> str:
    """Render a two-coin comparison for the model to read."""
    lines = [f"{a.coin_id} vs {b.coin_id}, {shared_days} overlapping days", ""]

    if coefficient is None:
        lines.append(
            "Correlation could not be computed: too few overlapping days, or "
            "one series barely moved (a stablecoin, for example)."
        )
    else:
        lines.append(
            f"Correlation of daily returns: {coefficient:+.2f} "
            f"({describe_correlation(coefficient)})"
        )
        if shared_days < RELIABLE_OBSERVATION_COUNT:
            lines.append(
                f"Based on only {shared_days} days, so treat this as indicative."
            )

    for stats in (a, b):
        lines.append("")
        lines.append(f"{stats.coin_id}")
        lines.append(f"  Period return: {stats.total_return_pct:+.1f}%")
        lines.append(f"  Annualised volatility: {stats.annualised_volatility_pct:.1f}%")
        if stats.drawdown.is_meaningful:
            lines.append(f"  Worst decline: -{stats.drawdown.magnitude_pct:.1f}%")

    lines.append("")
    lines.append(
        "Past correlation does not predict future correlation, and "
        "correlation is not causation."
    )

    return "\n".join(lines)
