"""Thin async client for the CoinGecko public API.

All network access for this project goes through this module. Keeping it in
one place means there is exactly one seam to wrap when adding caching, retries,
or rate limiting, and exactly one thing to mock in tests.
"""

from typing import Any

import httpx

BASE_URL = "https://api.coingecko.com/api/v3"
USER_AGENT = "mcp-crypto-server/0.1"
DEFAULT_TIMEOUT = 30.0


class CoinGeckoError(Exception):
    """Raised when a CoinGecko request fails.

    Tools catch this and turn it into a message for the model, rather than
    letting an httpx exception escape as an opaque tool error.
    """


async def get(endpoint: str, params: dict[str, Any] | None = None) -> Any:
    """GET a CoinGecko endpoint and return the decoded JSON body.

    Args:
        endpoint: Path below the API root, e.g. "/simple/price".
        params: Query parameters.

    Raises:
        CoinGeckoError: On any transport error or non-2xx response.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{BASE_URL}{endpoint}",
                params=params or {},
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise CoinGeckoError(
                    "CoinGecko rate limit reached. The free tier allows a limited "
                    "number of calls per minute; try again shortly."
                ) from exc
            raise CoinGeckoError(f"CoinGecko returned HTTP {status}.") from exc
        except httpx.TimeoutException as exc:
            raise CoinGeckoError("CoinGecko request timed out.") from exc
        except httpx.HTTPError as exc:
            raise CoinGeckoError(f"Could not reach CoinGecko: {exc}") from exc


async def simple_price(
    coin_ids: list[str],
    currency: str = "usd",
    include_24h_change: bool = True,
) -> dict[str, dict[str, float]]:
    """Fetch current prices for one or more coins in a single request.

    CoinGecko's /simple/price accepts a comma-separated id list, so pricing
    twenty coins costs one call rather than twenty.
    """
    return await get(
        "/simple/price",
        {
            "ids": ",".join(coin_ids),
            "vs_currencies": currency,
            "include_24hr_change": str(include_24h_change).lower(),
        },
    )


async def market_chart(coin_id: str, days: int, currency: str = "usd") -> dict[str, Any]:
    """Fetch historical market data for a coin over the last `days` days."""
    return await get(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": currency, "days": days},
    )


async def search(query: str) -> dict[str, Any]:
    """Search coins by name or symbol."""
    return await get("/search", {"query": query})
