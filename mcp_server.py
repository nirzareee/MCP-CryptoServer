"""MCP server exposing cryptocurrency market data tools.

This module holds tool definitions only. Network access lives in `coingecko`
and computation lives in `analytics`, so the tools here are thin: fetch,
compute, format, and turn failures into something a model can read.
"""

from mcp.server.fastmcp import FastMCP

import analytics
import coingecko
from coingecko import CoinGeckoError

mcp = FastMCP("crypto_price_tracker")


@mcp.tool()
async def get_crypto_price(crypto_id: str, currency: str = "usd") -> str:
    """Get the current price of a cryptocurrency.

    Args:
        crypto_id: CoinGecko coin id, e.g. "bitcoin", "ethereum".
        currency: Currency code to price in, e.g. "usd", "eur", "inr".
    """
    try:
        data = await coingecko.simple_price([crypto_id], currency)
    except CoinGeckoError as exc:
        return str(exc)

    entry = data.get(crypto_id)
    if not entry or entry.get(currency) is None:
        return (
            f"No price found for '{crypto_id}'. "
            f"Use search_coin to find the correct id."
        )

    price = entry[currency]
    change = entry.get(f"{currency}_24h_change")

    result = f"{crypto_id} is {price:,.2f} {currency.upper()}"
    if change is not None:
        result += f" ({change:+.2f}% over 24h)"
    return result


@mcp.tool()
async def get_portfolio(holdings: dict[str, float], currency: str = "usd") -> str:
    """Value a portfolio of cryptocurrency holdings.

    All coins are priced in a single API request. Coins that cannot be priced
    are reported separately rather than being dropped from the total.

    Args:
        holdings: Coin id to quantity held, e.g. {"bitcoin": 0.5, "ethereum": 3}.
            Use search_coin first if you are unsure of an id.
        currency: Currency to value the portfolio in, e.g. "usd".
    """
    if not holdings:
        return "No holdings provided."

    try:
        prices = await coingecko.simple_price(list(holdings), currency)
    except CoinGeckoError as exc:
        return str(exc)

    portfolio = analytics.value_portfolio(holdings, prices, currency)
    return analytics.format_portfolio(portfolio, currency)


@mcp.tool()
async def search_coin(query: str) -> str:
    """Find a cryptocurrency's CoinGecko id by name or ticker symbol.

    Use this when unsure of the exact id another tool needs.

    Args:
        query: Partial name or ticker, e.g. "btc", "doge", "chainlink".
    """
    try:
        data = await coingecko.search(query)
    except CoinGeckoError as exc:
        return str(exc)

    coins = data.get("coins", [])
    if not coins:
        return f"No coins matched '{query}'."

    lines = [
        f"{c['name']} ({c['symbol'].upper()}) -> id: {c['id']}" for c in coins[:8]
    ]
    return "Matches:\n" + "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
