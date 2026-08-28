"""Tests for the watchlist resources and the tools that change them.

Separate from `test_mcp_server` because these need a temporary watchlist file
swapped in as well as a client, and because the resource layer is a different
concern from the price tools.

The resources are read for their content, not through a live MCP session:
`@mcp.resource()` registers the function and returns it unchanged, so calling
it directly exercises the same code the client would receive.
"""

from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
import respx

import mcp_server
from coingecko import BASE_URL, CoinGeckoClient
from mcp_server import (
    add_to_watchlist,
    coin_resource,
    remove_from_watchlist,
    watchlist_resource,
)
from watchlist import MAX_ENTRIES, Watchlist

PRICE_URL = f"{BASE_URL}/simple/price"


@pytest.fixture(autouse=True)
def isolated_server(tmp_path: Path) -> Generator[None, None, None]:
    """Point the server at a temporary watchlist and a fresh client.

    Without this the suite would read and write the real
    `~/.mcp-crypto-server/watchlist.json`.
    """
    original_watchlist = mcp_server.watchlist
    original_client = mcp_server.client

    mcp_server.watchlist = Watchlist(path=tmp_path / "watchlist.json")
    mcp_server.client = CoinGeckoClient()

    yield

    mcp_server.watchlist = original_watchlist
    mcp_server.client = original_client


class TestAddTool:
    async def test_confirms_and_lists_current_contents(self) -> None:
        result = await add_to_watchlist("bitcoin")

        assert "bitcoin" in result
        assert mcp_server.watchlist.load() == ["bitcoin"]

    async def test_reports_the_running_count(self) -> None:
        await add_to_watchlist("bitcoin")
        result = await add_to_watchlist("ethereum")

        assert "2" in result
        assert "ethereum" in result

    async def test_invalid_id_returns_text_rather_than_raising(self) -> None:
        """An exception escaping a tool is opaque to the model."""
        result = await add_to_watchlist("not a coin!")

        assert "not a valid" in result
        assert mcp_server.watchlist.load() == []

    async def test_duplicate_is_accepted_quietly(self) -> None:
        await add_to_watchlist("bitcoin")
        result = await add_to_watchlist("bitcoin")

        assert "bitcoin" in result
        assert mcp_server.watchlist.load() == ["bitcoin"]

    async def test_capacity_message_reaches_the_user(self) -> None:
        for i in range(MAX_ENTRIES):
            await add_to_watchlist(f"coin-{i}")

        result = await add_to_watchlist("one-too-many")

        assert str(MAX_ENTRIES) in result


class TestRemoveTool:
    async def test_removes_and_lists_what_remains(self) -> None:
        await add_to_watchlist("bitcoin")
        await add_to_watchlist("ethereum")

        result = await remove_from_watchlist("bitcoin")

        assert "ethereum" in result
        assert mcp_server.watchlist.load() == ["ethereum"]

    async def test_says_so_when_the_list_empties(self) -> None:
        await add_to_watchlist("bitcoin")

        assert "empty" in await remove_from_watchlist("bitcoin")

    async def test_absent_coin_returns_text(self) -> None:
        result = await remove_from_watchlist("bitcoin")

        assert "not on the watchlist" in result


class TestWatchlistResource:
    async def test_empty_list_explains_how_to_fill_it(self) -> None:
        """A resource is read before the model acts, so it should orient."""
        result = await watchlist_resource()

        assert "empty" in result
        assert "add_to_watchlist" in result

    @respx.mock
    async def test_prices_every_tracked_coin(self) -> None:
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "bitcoin": {"usd": 50_000.0, "usd_24h_change": 2.5},
                    "ethereum": {"usd": 3_000.0, "usd_24h_change": -1.2},
                },
            )
        )
        await add_to_watchlist("bitcoin")
        await add_to_watchlist("ethereum")

        result = await watchlist_resource()

        assert "50,000.00 USD" in result
        assert "+2.50%" in result
        assert "-1.20%" in result

    @respx.mock
    async def test_prices_the_whole_list_in_one_request(self) -> None:
        route = respx.get(PRICE_URL).mock(return_value=httpx.Response(200, json={}))
        for coin in ("bitcoin", "ethereum", "solana"):
            await add_to_watchlist(coin)

        await watchlist_resource()

        assert route.call_count == 1

    @respx.mock
    async def test_unpriced_coin_is_shown_not_hidden(self) -> None:
        """A tracked coin missing from the response still belongs in the list."""
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json={"bitcoin": {"usd": 50_000.0}})
        )
        await add_to_watchlist("bitcoin")
        await add_to_watchlist("notacoin")

        result = await watchlist_resource()

        assert "notacoin" in result
        assert "no price available" in result

    @respx.mock
    async def test_api_failure_still_names_the_tracked_coins(self) -> None:
        """Prices are the extra; the list itself is the point of the resource."""
        respx.get(PRICE_URL).mock(return_value=httpx.Response(503))
        await add_to_watchlist("bitcoin")

        result = await watchlist_resource()

        assert "bitcoin" in result
        assert "unavailable" in result.lower()


class TestCoinResource:
    @respx.mock
    async def test_returns_price_for_the_uri_parameter(self) -> None:
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(
                200, json={"bitcoin": {"usd": 50_000.0, "usd_24h_change": 2.5}}
            )
        )

        result = await coin_resource("bitcoin")

        assert "50,000.00 USD" in result
        assert "+2.50%" in result

    @respx.mock
    async def test_unknown_coin(self) -> None:
        respx.get(PRICE_URL).mock(return_value=httpx.Response(200, json={}))

        assert "No data" in await coin_resource("notacoin")

    @respx.mock
    async def test_api_failure_returns_text(self) -> None:
        respx.get(PRICE_URL).mock(return_value=httpx.Response(503))

        assert "503" in await coin_resource("bitcoin")
