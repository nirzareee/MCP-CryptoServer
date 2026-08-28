"""Tests for the CoinGecko client.

This is the one module that touches the network, so it is the one that needs
mocking. `respx` intercepts httpx at the transport layer, which means the real
request-building and response-parsing code runs — only the socket is faked.

The module holds a single shared cache, so every test clears it first. That
global is a real design tradeoff: it keeps the client's call sites simple, at
the cost of test isolation that has to be managed by hand.
"""

from collections.abc import Generator

import httpx
import pytest
import respx

import coingecko

PRICE_URL = f"{coingecko.BASE_URL}/simple/price"
SEARCH_URL = f"{coingecko.BASE_URL}/search"


@pytest.fixture(autouse=True)
def clear_cache() -> Generator[None, None, None]:
    """Reset the module-level cache around every test."""
    coingecko._cache.clear()
    yield
    coingecko._cache.clear()


BITCOIN_RESPONSE = {"bitcoin": {"usd": 50_000.0, "usd_24h_change": 2.5}}


class TestSuccessfulFetch:
    @respx.mock
    async def test_returns_response_body(self) -> None:
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        fetched = await coingecko.simple_price(["bitcoin"])

        assert fetched.data == BITCOIN_RESPONSE
        assert fetched.age_seconds == 0.0
        assert not fetched.is_stale

    @respx.mock
    async def test_batches_ids_into_one_request(self) -> None:
        """Three coins must cost one call, not three."""
        route = respx.get(PRICE_URL).mock(return_value=httpx.Response(200, json={}))

        await coingecko.simple_price(["bitcoin", "ethereum", "solana"])

        assert route.call_count == 1
        sent_ids = route.calls[0].request.url.params["ids"]
        assert sent_ids == "bitcoin,ethereum,solana"

    @respx.mock
    async def test_ids_are_sorted_for_cache_stability(self) -> None:
        """Same coins in a different order must hit the same cache entry."""
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        await coingecko.simple_price(["ethereum", "bitcoin"])
        await coingecko.simple_price(["bitcoin", "ethereum"])

        assert route.call_count == 1


class TestCaching:
    @respx.mock
    async def test_second_call_does_not_hit_network(self) -> None:
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        first = await coingecko.simple_price(["bitcoin"])
        second = await coingecko.simple_price(["bitcoin"])

        assert route.call_count == 1
        assert second.data == first.data

    @respx.mock
    async def test_refresh_bypasses_cache(self) -> None:
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        await coingecko.simple_price(["bitcoin"])
        await coingecko.simple_price(["bitcoin"], refresh=True)

        assert route.call_count == 2

    @respx.mock
    async def test_different_currencies_cached_separately(self) -> None:
        route = respx.get(PRICE_URL).mock(return_value=httpx.Response(200, json={}))

        await coingecko.simple_price(["bitcoin"], "usd")
        await coingecko.simple_price(["bitcoin"], "eur")

        assert route.call_count == 2


class TestErrorTranslation:
    @respx.mock
    async def test_rate_limit_message_is_specific(self) -> None:
        """429 is the failure users will actually hit, so it says what to do."""
        respx.get(PRICE_URL).mock(return_value=httpx.Response(429))

        with pytest.raises(coingecko.CoinGeckoError) as exc_info:
            await coingecko.simple_price(["bitcoin"])

        assert "rate limit" in str(exc_info.value).lower()

    @respx.mock
    async def test_server_error_reports_status(self) -> None:
        respx.get(PRICE_URL).mock(return_value=httpx.Response(503))

        with pytest.raises(coingecko.CoinGeckoError) as exc_info:
            await coingecko.simple_price(["bitcoin"])

        assert "503" in str(exc_info.value)

    @respx.mock
    async def test_timeout_becomes_domain_error(self) -> None:
        """Callers should never have to catch an httpx exception."""
        respx.get(PRICE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

        with pytest.raises(coingecko.CoinGeckoError) as exc_info:
            await coingecko.simple_price(["bitcoin"])

        assert "timed out" in str(exc_info.value).lower()

    @respx.mock
    async def test_connection_error_becomes_domain_error(self) -> None:
        respx.get(PRICE_URL).mock(side_effect=httpx.ConnectError("no route"))

        with pytest.raises(coingecko.CoinGeckoError):
            await coingecko.simple_price(["bitcoin"])


class TestStaleFallback:
    @respx.mock
    async def test_serves_stale_data_when_live_fetch_fails(self) -> None:
        route = respx.get(PRICE_URL)
        route.mock(return_value=httpx.Response(200, json=BITCOIN_RESPONSE))

        await coingecko.simple_price(["bitcoin"])

        route.mock(return_value=httpx.Response(503))
        fetched = await coingecko.simple_price(["bitcoin"], refresh=True)

        assert fetched.data == BITCOIN_RESPONSE

    @respx.mock
    async def test_raises_when_failing_with_nothing_cached(self) -> None:
        """No fallback available means a real error, not an empty result."""
        respx.get(PRICE_URL).mock(return_value=httpx.Response(503))

        with pytest.raises(coingecko.CoinGeckoError):
            await coingecko.simple_price(["bitcoin"])


class TestStalenessReporting:
    def test_fresh_data_has_no_note(self) -> None:
        assert coingecko.Fetched(data={}, age_seconds=0.0).staleness_note() is None

    def test_data_within_ttl_has_no_note(self) -> None:
        fetched = coingecko.Fetched(data={}, age_seconds=30.0)
        assert fetched.staleness_note() is None

    def test_old_data_says_how_old(self) -> None:
        """The age must reach the user; stale prices presented as live are a bug."""
        note = coingecko.Fetched(data={}, age_seconds=240.0).staleness_note()

        assert note is not None
        assert "4.0 minutes" in note


class TestSearch:
    @respx.mock
    async def test_passes_query_through(self) -> None:
        route = respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"coins": []})
        )

        await coingecko.search("btc")

        assert route.calls[0].request.url.params["query"] == "btc"
