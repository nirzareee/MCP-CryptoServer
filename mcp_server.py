"""MCP server exposing cryptocurrency market data tools.

This module is the composition root: it creates the one CoinGecko client the
tools share, and holds the tool definitions themselves. Network access and
caching live in `coingecko`; computation lives in `analytics` and
`timeseries`. The tools here are thin — fetch, compute, format, and turn
failures into something a model can read.

The client is created here rather than passed into each tool because a tool's
parameters are its public interface: MCP turns the signature into the schema
the model sees. A `client` argument would appear as something the model is
expected to supply, which it cannot.

The watchlist is exposed as a resource rather than a tool. The distinction is
who initiates: tools are called by the model when it decides it needs them,
resources are read by the client and attached to context. "What am I
tracking" is context the model should already have, not a question it should
have to think to ask. Changing the watchlist stays a tool, because that is an
action taken on purpose.
"""

from mcp.server.fastmcp import FastMCP

import analytics
import timeseries
from coingecko import CoinGeckoClient, CoinGeckoError
from watchlist import Watchlist, WatchlistError

mcp = FastMCP("crypto_price_tracker")

client = CoinGeckoClient()
watchlist = Watchlist()


@mcp.tool()
async def get_crypto_price(
    crypto_id: str, currency: str = "usd", refresh: bool = False
) -> str:
    """Get the current price of a cryptocurrency.

    Args:
        crypto_id: CoinGecko coin id, e.g. "bitcoin", "ethereum".
        currency: Currency code to price in, e.g. "usd", "eur", "inr".
        refresh: Bypass the short-lived price cache and fetch live. Set this
            when the user asks to check again or wants the very latest figure.
    """
    try:
        fetched = await client.simple_price([crypto_id], currency, refresh=refresh)
    except CoinGeckoError as exc:
        return str(exc)

    entry = fetched.data.get(crypto_id)
    if not entry or entry.get(currency) is None:
        return (
            f"No price found for '{crypto_id}'. Use search_coin to find the correct id."
        )

    price = entry[currency]
    change = entry.get(f"{currency}_24h_change")

    result = f"{crypto_id} is {price:,.2f} {currency.upper()}"
    if change is not None:
        result += f" ({change:+.2f}% over 24h)"

    note = fetched.staleness_note()
    return f"{result}\n{note}" if note else result


@mcp.tool()
async def get_portfolio(
    holdings: dict[str, float], currency: str = "usd", refresh: bool = False
) -> str:
    """Value a portfolio of cryptocurrency holdings.

    All coins are priced in a single API request. Coins that cannot be priced
    are reported separately rather than being dropped from the total.

    Args:
        holdings: Coin id to quantity held, e.g. {"bitcoin": 0.5, "ethereum": 3}.
            Use search_coin first if you are unsure of an id.
        currency: Currency to value the portfolio in, e.g. "usd".
        refresh: Bypass the short-lived price cache and fetch live.
    """
    if not holdings:
        return "No holdings provided."

    try:
        fetched = await client.simple_price(list(holdings), currency, refresh=refresh)
    except CoinGeckoError as exc:
        return str(exc)

    portfolio = analytics.value_portfolio(holdings, fetched.data, currency)
    output = analytics.format_portfolio(portfolio, currency)

    note = fetched.staleness_note()
    return f"{output}\n\n{note}" if note else output


@mcp.tool()
async def compare_coins(
    coin_a: str,
    coin_b: str,
    days: int = 90,
    currency: str = "usd",
) -> str:
    """Compare how two cryptocurrencies have moved relative to each other.

    Reports the correlation of their daily returns over the period, plus each
    coin's return, annualised volatility, and worst peak-to-trough decline.
    Useful for questions about diversification: two coins that move together
    offer less of it than holding one.

    Args:
        coin_a: CoinGecko coin id, e.g. "bitcoin".
        coin_b: CoinGecko coin id, e.g. "ethereum".
        days: Length of history to analyse, 2 to 365. Defaults to 90.
        currency: Currency to price both coins in.
    """
    if coin_a == coin_b:
        return "Pick two different coins; a coin is perfectly correlated with itself."

    if not 2 <= days <= 365:
        return "The `days` argument must be between 2 and 365."

    try:
        chart_a = await client.market_chart(coin_a, days, currency)
        chart_b = await client.market_chart(coin_b, days, currency)
    except CoinGeckoError as exc:
        return str(exc)

    closes_a = timeseries.to_daily_closes(chart_a.data.get("prices", []))
    closes_b = timeseries.to_daily_closes(chart_b.data.get("prices", []))

    if not closes_a:
        return f"No price history returned for '{coin_a}'. Check the coin id."
    if not closes_b:
        return f"No price history returned for '{coin_b}'. Check the coin id."

    shared_days, prices_a, prices_b = timeseries.align(closes_a, closes_b)

    if len(shared_days) < 2:
        return (
            f"{coin_a} and {coin_b} have too little overlapping history "
            f"({len(shared_days)} shared days) to compare."
        )

    coefficient = timeseries.pearson_correlation(
        timeseries.daily_returns(prices_a),
        timeseries.daily_returns(prices_b),
    )

    stats_a = timeseries.summarise_series(coin_a, closes_a)
    stats_b = timeseries.summarise_series(coin_b, closes_b)

    if stats_a is None or stats_b is None:
        return "Not enough price history to summarise both coins."

    output = timeseries.format_comparison(
        stats_a, stats_b, coefficient, len(shared_days)
    )

    note = chart_a.staleness_note() or chart_b.staleness_note()
    return f"{output}\n\n{note}" if note else output


@mcp.tool()
async def search_coin(query: str) -> str:
    """Find a cryptocurrency's CoinGecko id by name or ticker symbol.

    Use this when unsure of the exact id another tool needs.

    Args:
        query: Partial name or ticker, e.g. "btc", "doge", "chainlink".
    """
    try:
        fetched = await client.search(query)
    except CoinGeckoError as exc:
        return str(exc)

    coins = fetched.data.get("coins", [])
    if not coins:
        return f"No coins matched '{query}'."

    lines = [f"{c['name']} ({c['symbol'].upper()}) -> id: {c['id']}" for c in coins[:8]]
    return "Matches:\n" + "\n".join(lines)


@mcp.resource("crypto://watchlist")
async def watchlist_resource() -> str:
    """The user's tracked coins, with current prices.

    Read by the client rather than called by the model, so the model starts a
    conversation already knowing what is being tracked.
    """
    coins = watchlist.load()

    if not coins:
        return (
            "The watchlist is empty. Coins can be added with the add_to_watchlist tool."
        )

    try:
        fetched = await client.simple_price(coins)
    except CoinGeckoError as exc:
        return f"Watching: {', '.join(coins)}\n(Prices unavailable: {exc})"

    lines = ["Watchlist"]
    for coin_id in coins:
        entry = fetched.data.get(coin_id)
        price = entry.get("usd") if entry else None

        if price is None:
            lines.append(f"  {coin_id}: no price available")
            continue

        line = f"  {coin_id}: {price:,.2f} USD"
        change = entry.get("usd_24h_change")
        if change is not None:
            line += f" ({change:+.2f}% 24h)"
        lines.append(line)

    note = fetched.staleness_note()
    if note:
        lines.append(note)

    return "\n".join(lines)


@mcp.resource("crypto://coin/{coin_id}")
async def coin_resource(coin_id: str) -> str:
    """Market data for a single coin, addressable by URI.

    A resource template: the client can read `crypto://coin/bitcoin` directly
    without the model having to decide to call a tool first.
    """
    try:
        fetched = await client.simple_price([coin_id])
    except CoinGeckoError as exc:
        return f"Could not fetch {coin_id}: {exc}"

    entry = fetched.data.get(coin_id)
    if not entry or entry.get("usd") is None:
        return f"No data for '{coin_id}'."

    line = f"{coin_id}: {entry['usd']:,.2f} USD"
    change = entry.get("usd_24h_change")
    if change is not None:
        line += f" ({change:+.2f}% over 24h)"
    return line


@mcp.tool()
async def add_to_watchlist(coin_id: str) -> str:
    """Add a coin to the tracked watchlist.

    Args:
        coin_id: CoinGecko coin id, e.g. "bitcoin". Use search_coin first if
            unsure; an id that does not exist will be stored but will show no
            price.
    """
    try:
        coins = watchlist.add(coin_id)
    except WatchlistError as exc:
        return str(exc)

    return f"Watching {coin_id}. Now tracking {len(coins)}: {', '.join(coins)}"


@mcp.tool()
async def remove_from_watchlist(coin_id: str) -> str:
    """Remove a coin from the tracked watchlist.

    Args:
        coin_id: CoinGecko coin id to stop tracking.
    """
    try:
        coins = watchlist.remove(coin_id)
    except WatchlistError as exc:
        return str(exc)

    if not coins:
        return f"Removed {coin_id}. The watchlist is now empty."
    return f"Removed {coin_id}. Still tracking: {', '.join(coins)}"


if __name__ == "__main__":
    mcp.run()
