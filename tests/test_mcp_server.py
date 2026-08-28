"""Tests for the MCP tool functions themselves.

The other suites cover computation. This one covers the layer between the
model and everything else: whether a rate-limit failure reaches the user as
readable text, whether stale data arrives labelled with its age, whether a
coin that could not be priced is named rather than dropped.

Those are the behaviours a model depends on to say something true. They are
also the ones that break silently — a tool that returns a plausible sentence
built from wrong data looks exactly like a tool that works.

FastMCP's `@mcp.tool()` decorator registers the function and returns it
unchanged, so the tools can be awaited directly here. Mocking stays at the
HTTP transport, so the real fetch, parse, compute, and format path runs, and
each test gets its own client so no global state needs resetting.
"""

from collections.abc import Generator
from typing import Any

import httpx
import pytest
import respx

import mcp_server
from cache import TTLCache
from coingecko import BASE_URL, CoinGeckoClient
from mcp_server import (
    compare_coins,
    get_crypto_price,
    get_portfolio,
    search_coin,
)

PRICE_URL = f"{BASE_URL}/simple/price"
SEARCH_URL = f"{BASE_URL}/search"

DAY_MS = 86_400_000
JAN_1_2024_MS = 1_704_067_200_000


class FakeClock:
    """A clock the test controls."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def fresh_client() -> Generator[FakeClock, None, None]:
    """Give each test its own client, with a clock it can advance.

    `mcp_server` creates one client at import time, which is right for the
    running server and wrong for tests. Swapping in a per-test instance keeps
    cases isolated without clearing global state.
    """
    clock = FakeClock()
    original = mcp_server.client
    mcp_server.client = CoinGeckoClient(cache=TTLCache(60.0, clock=clock))
    yield clock
    mcp_server.client = original


def chart_url(coin_id: str) -> str:
    return f"{BASE_URL}/coins/{coin_id}/market_chart"


def chart(*prices: float) -> dict[str, Any]:
    """Build a market_chart body with one daily observation per price."""
    return {"prices": [[JAN_1_2024_MS + i * DAY_MS, p] for i, p in enumerate(prices)]}


class TestGetCryptoPrice:
    @respx.mock
    async def test_reports_price_and_change(self) -> None:
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(
                200, json={"bitcoin": {"usd": 50_000.0, "usd_24h_change": 2.5}}
            )
        )

        result = await get_crypto_price("bitcoin")

        assert "50,000.00 USD" in result
        assert "+2.50%" in result

    @respx.mock
    async def test_omits_change_when_absent(self) -> None:
        """A missing 24h figure must not render as '+0.00%' or 'None'."""
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json={"bitcoin": {"usd": 50_000.0}})
        )

        result = await get_crypto_price("bitcoin")

        assert "50,000.00 USD" in result
        assert "None" not in result
        assert "%" not in result

    @respx.mock
    async def test_unknown_coin_points_at_search(self) -> None:
        """A dead end should tell the model what to try next."""
        respx.get(PRICE_URL).mock(return_value=httpx.Response(200, json={}))

        result = await get_crypto_price("notacoin")

        assert "notacoin" in result
        assert "search_coin" in result

    @respx.mock
    async def test_rate_limit_reaches_the_user_as_text(self) -> None:
        """The tool must return a message, not raise.

        An exception escaping an MCP tool becomes an opaque error the model
        cannot explain.
        """
        respx.get(PRICE_URL).mock(return_value=httpx.Response(429))

        result = await get_crypto_price("bitcoin")

        assert "rate limit" in result.lower()

    @respx.mock
    async def test_refresh_forces_a_second_request(self) -> None:
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json={"bitcoin": {"usd": 1.0}})
        )

        await get_crypto_price("bitcoin")
        await get_crypto_price("bitcoin", refresh=True)

        assert route.call_count == 2


class TestStalenessSurfacing:
    @respx.mock
    async def test_stale_fallback_is_labelled_with_its_age(
        self, fresh_client: FakeClock
    ) -> None:
        """Stale prices presented as current would be the worst failure here.

        Time is advanced past the TTL, the live fetch is made to fail, and the
        tool must both serve the old figure and say how old it is.
        """
        route = respx.get(PRICE_URL)
        route.mock(
            return_value=httpx.Response(200, json={"bitcoin": {"usd": 50_000.0}})
        )
        await get_crypto_price("bitcoin")

        fresh_client.advance(300.0)
        route.mock(return_value=httpx.Response(503))
        result = await get_crypto_price("bitcoin")

        assert "50,000.00" in result
        assert "5.0 minutes old" in result


class TestGetPortfolio:
    @respx.mock
    async def test_values_holdings(self) -> None:
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(
                200,
                json={"bitcoin": {"usd": 50_000.0}, "ethereum": {"usd": 3_000.0}},
            )
        )

        result = await get_portfolio({"bitcoin": 1.0, "ethereum": 2.0})

        assert "56,000.00 USD" in result

    @respx.mock
    async def test_prices_everything_in_one_request(self) -> None:
        """Three coins must cost one call, not three."""
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json={"bitcoin": {"usd": 1.0}})
        )

        await get_portfolio({"bitcoin": 1.0, "ethereum": 1.0, "solana": 1.0})

        assert route.call_count == 1

    @respx.mock
    async def test_unpriced_coin_is_named_in_the_output(self) -> None:
        """A total that quietly omits a holding is worse than no total."""
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json={"bitcoin": {"usd": 50_000.0}})
        )

        result = await get_portfolio({"bitcoin": 1.0, "notacoin": 5.0})

        assert "notacoin" in result
        assert "50,000.00 USD" in result

    async def test_empty_holdings_needs_no_request(self) -> None:
        """Guarded before the network, so no respx mock is needed."""
        assert await get_portfolio({}) == "No holdings provided."

    @respx.mock
    async def test_api_failure_returns_text(self) -> None:
        respx.get(PRICE_URL).mock(return_value=httpx.Response(503))

        result = await get_portfolio({"bitcoin": 1.0})

        assert "503" in result


class TestSearchCoin:
    @respx.mock
    async def test_lists_ids_for_the_model_to_reuse(self) -> None:
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "coins": [
                        {"name": "Bitcoin", "symbol": "btc", "id": "bitcoin"},
                        {"name": "Bitcoin Cash", "symbol": "bch", "id": "bitcoin-cash"},
                    ]
                },
            )
        )

        result = await search_coin("bitcoin")

        assert "id: bitcoin" in result
        assert "id: bitcoin-cash" in result

    @respx.mock
    async def test_caps_the_result_list(self) -> None:
        """Twenty matches would bury the useful ones."""
        coins = [
            {"name": f"Coin {i}", "symbol": f"c{i}", "id": f"coin-{i}"}
            for i in range(20)
        ]
        respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"coins": coins})
        )

        result = await search_coin("coin")

        assert result.count("id:") == 8

    @respx.mock
    async def test_no_matches(self) -> None:
        respx.get(SEARCH_URL).mock(return_value=httpx.Response(200, json={"coins": []}))

        assert "zzzz" in await search_coin("zzzz")


class TestCompareCoins:
    async def test_rejects_a_coin_against_itself(self) -> None:
        """Trivially 1.0, and almost certainly a mistake by the caller."""
        result = await compare_coins("bitcoin", "bitcoin")
        assert "different coins" in result

    @pytest.mark.parametrize("days", [0, 1, 366, -5])
    async def test_rejects_out_of_range_days(self, days: int) -> None:
        result = await compare_coins("bitcoin", "ethereum", days=days)
        assert "between 2 and 365" in result

    @respx.mock
    async def test_reports_correlation_for_coins_that_move_together(self) -> None:
        respx.get(chart_url("bitcoin")).mock(
            return_value=httpx.Response(
                200, json=chart(100.0, 110.0, 121.0, 133.1, 146.4)
            )
        )
        respx.get(chart_url("ethereum")).mock(
            return_value=httpx.Response(200, json=chart(10.0, 11.0, 12.1, 13.31, 14.64))
        )

        result = await compare_coins("bitcoin", "ethereum", days=5)

        assert "bitcoin vs ethereum" in result
        assert "5 overlapping days" in result
        assert "+1.00" in result  # identical percentage moves

    @respx.mock
    async def test_says_so_when_correlation_is_undefined(self) -> None:
        """A stablecoin has no variance, so no coefficient can be reported."""
        respx.get(chart_url("bitcoin")).mock(
            return_value=httpx.Response(200, json=chart(100.0, 120.0, 90.0, 130.0))
        )
        respx.get(chart_url("tether")).mock(
            return_value=httpx.Response(200, json=chart(1.0, 1.0, 1.0, 1.0))
        )

        result = await compare_coins("bitcoin", "tether", days=4)

        assert "could not be computed" in result

    @respx.mock
    async def test_flags_a_short_sample(self) -> None:
        """A coefficient from four days should not be presented as a finding."""
        respx.get(chart_url("bitcoin")).mock(
            return_value=httpx.Response(200, json=chart(100.0, 110.0, 105.0, 115.0))
        )
        respx.get(chart_url("ethereum")).mock(
            return_value=httpx.Response(200, json=chart(10.0, 9.0, 11.0, 10.5))
        )

        result = await compare_coins("bitcoin", "ethereum", days=4)

        assert "indicative" in result

    @respx.mock
    async def test_refuses_when_histories_barely_overlap(self) -> None:
        """Non-overlapping series must not be compared day-against-day."""
        respx.get(chart_url("bitcoin")).mock(
            return_value=httpx.Response(200, json=chart(100.0, 110.0, 120.0))
        )
        respx.get(chart_url("newcoin")).mock(
            return_value=httpx.Response(
                200,
                json={"prices": [[JAN_1_2024_MS + 10 * DAY_MS, 5.0]]},
            )
        )

        result = await compare_coins("bitcoin", "newcoin", days=30)

        assert "too little overlapping history" in result

    @respx.mock
    async def test_empty_history_names_the_coin(self) -> None:
        respx.get(chart_url("bitcoin")).mock(
            return_value=httpx.Response(200, json={"prices": []})
        )
        respx.get(chart_url("ethereum")).mock(
            return_value=httpx.Response(200, json=chart(1.0, 2.0))
        )

        result = await compare_coins("bitcoin", "ethereum")

        assert "bitcoin" in result
        assert "Check the coin id" in result

    @respx.mock
    async def test_always_carries_the_causation_caveat(self) -> None:
        """A correlation figure without this line invites over-reading."""
        respx.get(chart_url("bitcoin")).mock(
            return_value=httpx.Response(200, json=chart(100.0, 110.0, 120.0, 130.0))
        )
        respx.get(chart_url("ethereum")).mock(
            return_value=httpx.Response(200, json=chart(10.0, 11.0, 12.0, 13.0))
        )

        result = await compare_coins("bitcoin", "ethereum", days=4)

        assert "not causation" in result
