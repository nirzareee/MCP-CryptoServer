"""Thin async client for the CoinGecko public API.

All network access for this project goes through this module, so there is one
seam to wrap for caching, retries, or rate limiting, and one thing to mock in
tests.

Responses are cached for a short TTL. CoinGecko's free tier allows roughly
10-30 calls per minute, and a single conversation can easily ask about the
same coins several times in a row: "what's my portfolio worth", then "what
about just bitcoin", then "check again". Without a cache those are three
identical requests within seconds, and a demo can hit a 429 in front of an
audience.
"""

from dataclasses import dataclass
from typing import Any

import httpx

from cache import TTLCache, make_key

BASE_URL = "https://api.coingecko.com/api/v3"
USER_AGENT = "mcp-crypto-server/0.1"
DEFAULT_TIMEOUT = 30.0
CACHE_TTL_SECONDS = 60.0

_cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS)


class CoinGeckoError(Exception):
    """Raised when a CoinGecko request fails and no cached data can stand in."""


@dataclass(frozen=True)
class Fetched:
    """A response body plus where it came from.

    `age_seconds` is 0.0 for a live fetch. Anything higher means the data came
    from cache, and a caller presenting it to a user should say so.
    """

    data: Any
    age_seconds: float = 0.0

    @property
    def is_stale(self) -> bool:
        """True when the data is older than the cache's freshness window."""
        return self.age_seconds > CACHE_TTL_SECONDS

    def staleness_note(self) -> str | None:
        """A sentence describing the data's age, or None if it is fresh."""
        if not self.is_stale:
            return None
        minutes = self.age_seconds / 60
        return (
            f"Note: live data was unavailable, so these figures are "
            f"{minutes:.1f} minutes old."
        )


async def get(
    endpoint: str,
    params: dict[str, Any] | None = None,
    refresh: bool = False,
) -> Fetched:
    """GET a CoinGecko endpoint, using the cache when possible.

    Args:
        endpoint: Path below the API root, e.g. "/simple/price".
        params: Query parameters.
        refresh: Skip the cache and force a live request.

    Raises:
        CoinGeckoError: When the request fails and nothing is cached.
    """
    params = params or {}
    key = make_key(endpoint, params)

    if not refresh:
        cached = _cache.get(key)
        if cached is not None:
            return Fetched(data=cached, age_seconds=_cache.age(key))

    try:
        data = await _request(endpoint, params)
    except CoinGeckoError:
        stale = _cache.get_stale(key)
        if stale is not None:
            value, age = stale
            return Fetched(data=value, age_seconds=age)
        raise

    _cache.set(key, data)
    return Fetched(data=data, age_seconds=0.0)


async def _request(endpoint: str, params: dict[str, Any]) -> Any:
    """Perform the HTTP request, translating transport errors."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{BASE_URL}{endpoint}",
                params=params,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429:
                raise CoinGeckoError(
                    "CoinGecko rate limit reached. The free tier allows a "
                    "limited number of calls per minute; try again shortly."
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
    refresh: bool = False,
) -> Fetched:
    """Fetch current prices for one or more coins in a single request.

    CoinGecko's /simple/price accepts a comma-separated id list, so pricing
    twenty coins costs one call rather than twenty. Ids are sorted so that
    the same set of coins in a different order hits the same cache entry.
    """
    return await get(
        "/simple/price",
        {
            "ids": ",".join(sorted(coin_ids)),
            "vs_currencies": currency,
            "include_24hr_change": str(include_24h_change).lower(),
        },
        refresh=refresh,
    )


async def market_chart(
    coin_id: str,
    days: int,
    currency: str = "usd",
    refresh: bool = False,
) -> Fetched:
    """Fetch historical market data for a coin over the last `days` days."""
    return await get(
        f"/coins/{coin_id}/market_chart",
        {"vs_currency": currency, "days": days},
        refresh=refresh,
    )


async def search(query: str, refresh: bool = False) -> Fetched:
    """Search coins by name or symbol."""
    return await get("/search", {"query": query}, refresh=refresh)


def cache_stats() -> dict[str, Any]:
    """Current cache counters, for diagnostics."""
    return {
        "entries": _cache.size,
        "hits": _cache.hits,
        "misses": _cache.misses,
        "hit_rate": round(_cache.hit_rate, 3),
        "ttl_seconds": CACHE_TTL_SECONDS,
    }
