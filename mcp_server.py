"""MCP server exposing cryptocurrency market data tools.

This module holds tool definitions only. Network access and caching live in
`coingecko`, computation lives in `analytics`, so the tools here are thin:
fetch, compute, format, and turn failures into something a model can read.
"""

from mcp.server.fastmcp import FastMCP

import analytics
import coingecko
from coingecko import CoinGeckoError

mcp = FastMCP("crypto_price_tracker")


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
        fetched = await coingecko.simple_price([crypto_id], currency, refresh=refresh)
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
        fetched = await coingecko.simple_price(
            list(holdings), currency, refresh=refresh
        )
    except CoinGeckoError as exc:
        return str(exc)

    portfolio = analytics.value_portfolio(holdings, fetched.data, currency)
    output = analytics.format_portfolio(portfolio, currency)

    note = fetched.staleness_note()
    return f"{output}\n\n{note}" if note else output


@mcp.tool()
async def search_coin(query: str) -> str:
    """Find a cryptocurrency's CoinGecko id by name or ticker symbol.

    Use this when unsure of the exact id another tool needs.

    Args:
        query: Partial name or ticker, e.g. "btc", "doge", "chainlink".
    """
    try:
        fetched = await coingecko.search(query)
    except CoinGeckoError as exc:
        return str(exc)

    coins = fetched.data.get("coins", [])
    if not coins:
        return f"No coins matched '{query}'."

    lines = [f"{c['name']} ({c['symbol'].upper()}) -> id: {c['id']}" for c in coins[:8]]
    return "Matches:\n" + "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
