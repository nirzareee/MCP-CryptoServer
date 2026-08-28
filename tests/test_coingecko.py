"""Tests for the CoinGecko client.

This is the one module that touches the network, so it is the one that needs
mocking. `respx` intercepts httpx at the transport layer, which means the real
request-building and response-parsing code runs — only the socket is faked.

Each test builds its own client, so there is no shared cache to reset between
cases and no private attribute to reach into. Tests that care about expiry
inject a cache with a controllable clock rather than sleeping.
"""

import httpx
import pytest
import respx

from cache import TTLCache
from coingecko import BASE_URL, CoinGeckoClient, CoinGeckoError, Fetched

PRICE_URL = f"{BASE_URL}/simple/price"
SEARCH_URL = f"{BASE_URL}/search"

BITCOIN_RESPONSE = {"bitcoin": {"usd": 50_000.0, "usd_24h_change": 2.5}}


class FakeClock:
    """A clock the test controls."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def client() -> CoinGeckoClient:
    return CoinGeckoClient()


class TestSuccessfulFetch:
    @respx.mock
    async def test_returns_response_body(self, client: CoinGeckoClient) -> None:
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        fetched = await client.simple_price(["bitcoin"])

        assert fetched.data == BITCOIN_RESPONSE
        assert fetched.age_seconds == 0.0
        assert not fetched.is_stale

    @respx.mock
    async def test_batches_ids_into_one_request(self, client: CoinGeckoClient) -> None:
        """Three coins must cost one call, not three."""
        route = respx.get(PRICE_URL).mock(return_value=httpx.Response(200, json={}))

        await client.simple_price(["bitcoin", "ethereum", "solana"])

        assert route.call_count == 1
        assert route.calls[0].request.url.params["ids"] == "bitcoin,ethereum,solana"

    @respx.mock
    async def test_ids_are_sorted_for_cache_stability(
        self, client: CoinGeckoClient
    ) -> None:
        """Same coins in a different order must hit the same cache entry."""
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        await client.simple_price(["ethereum", "bitcoin"])
        await client.simple_price(["bitcoin", "ethereum"])

        assert route.call_count == 1


class TestCaching:
    @respx.mock
    async def test_second_call_does_not_hit_network(
        self, client: CoinGeckoClient
    ) -> None:
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        first = await client.simple_price(["bitcoin"])
        second = await client.simple_price(["bitcoin"])

        assert route.call_count == 1
        assert second.data == first.data

    @respx.mock
    async def test_refresh_bypasses_cache(self, client: CoinGeckoClient) -> None:
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        await client.simple_price(["bitcoin"])
        await client.simple_price(["bitcoin"], refresh=True)

        assert route.call_count == 2

    @respx.mock
    async def test_different_currencies_cached_separately(
        self, client: CoinGeckoClient
    ) -> None:
        route = respx.get(PRICE_URL).mock(return_value=httpx.Response(200, json={}))

        await client.simple_price(["bitcoin"], "usd")
        await client.simple_price(["bitcoin"], "eur")

        assert route.call_count == 2

    @respx.mock
    async def test_expired_entry_triggers_a_fresh_request(self) -> None:
        """Injecting the clock lets expiry be tested without sleeping."""
        clock = FakeClock()
        client = CoinGeckoClient(cache=TTLCache(ttl_seconds=60.0, clock=clock))

        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        await client.simple_price(["bitcoin"])
        clock.advance(61.0)
        await client.simple_price(["bitcoin"])

        assert route.call_count == 2

    @respx.mock
    async def test_clients_do_not_share_a_cache(self) -> None:
        """Instance-owned state: one client's cache must not serve another's call.

        This is the property the module-level singleton could not offer, and
        the reason tests no longer reset global state between cases.
        """
        route = respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        await CoinGeckoClient().simple_price(["bitcoin"])
        await CoinGeckoClient().simple_price(["bitcoin"])

        assert route.call_count == 2


class TestErrorTranslation:
    @respx.mock
    async def test_rate_limit_message_is_specific(
        self, client: CoinGeckoClient
    ) -> None:
        """429 is the failure users will actually hit, so it says what to do."""
        respx.get(PRICE_URL).mock(return_value=httpx.Response(429))

        with pytest.raises(CoinGeckoError) as exc_info:
            await client.simple_price(["bitcoin"])

        assert "rate limit" in str(exc_info.value).lower()

    @respx.mock
    async def test_server_error_reports_status(self, client: CoinGeckoClient) -> None:
        respx.get(PRICE_URL).mock(return_value=httpx.Response(503))

        with pytest.raises(CoinGeckoError) as exc_info:
            await client.simple_price(["bitcoin"])

        assert "503" in str(exc_info.value)

    @respx.mock
    async def test_timeout_becomes_domain_error(self, client: CoinGeckoClient) -> None:
        """Callers should never have to catch an httpx exception."""
        respx.get(PRICE_URL).mock(side_effect=httpx.TimeoutException("timed out"))

        with pytest.raises(CoinGeckoError) as exc_info:
            await client.simple_price(["bitcoin"])

        assert "timed out" in str(exc_info.value).lower()

    @respx.mock
    async def test_connection_error_becomes_domain_error(
        self, client: CoinGeckoClient
    ) -> None:
        respx.get(PRICE_URL).mock(side_effect=httpx.ConnectError("no route"))

        with pytest.raises(CoinGeckoError):
            await client.simple_price(["bitcoin"])


class TestStaleFallback:
    @respx.mock
    async def test_serves_stale_data_when_live_fetch_fails(
        self, client: CoinGeckoClient
    ) -> None:
        route = respx.get(PRICE_URL)
        route.mock(return_value=httpx.Response(200, json=BITCOIN_RESPONSE))

        await client.simple_price(["bitcoin"])

        route.mock(return_value=httpx.Response(503))
        fetched = await client.simple_price(["bitcoin"], refresh=True)

        assert fetched.data == BITCOIN_RESPONSE

    @respx.mock
    async def test_stale_fallback_carries_its_age(self) -> None:
        """Age is measured, not assumed — the clock makes it exact."""
        clock = FakeClock()
        client = CoinGeckoClient(cache=TTLCache(ttl_seconds=60.0, clock=clock))

        route = respx.get(PRICE_URL)
        route.mock(return_value=httpx.Response(200, json=BITCOIN_RESPONSE))
        await client.simple_price(["bitcoin"])

        clock.advance(300.0)
        route.mock(return_value=httpx.Response(503))
        fetched = await client.simple_price(["bitcoin"])

        assert fetched.data == BITCOIN_RESPONSE
        assert fetched.age_seconds == pytest.approx(300.0)
        assert fetched.is_stale

        note = fetched.staleness_note()
        assert note is not None
        assert "5.0 minutes" in note

    @respx.mock
    async def test_raises_when_failing_with_nothing_cached(
        self, client: CoinGeckoClient
    ) -> None:
        """No fallback available means a real error, not an empty result."""
        respx.get(PRICE_URL).mock(return_value=httpx.Response(503))

        with pytest.raises(CoinGeckoError):
            await client.simple_price(["bitcoin"])


class TestStalenessReporting:
    def test_fresh_data_has_no_note(self) -> None:
        assert Fetched(data={}, age_seconds=0.0).staleness_note() is None

    def test_data_within_ttl_has_no_note(self) -> None:
        assert Fetched(data={}, age_seconds=30.0).staleness_note() is None

    def test_old_data_says_how_old(self) -> None:
        """The age must reach the user; stale prices presented as live are a bug."""
        note = Fetched(data={}, age_seconds=240.0).staleness_note()

        assert note is not None
        assert "4.0 minutes" in note

    def test_staleness_follows_the_cache_ttl(self) -> None:
        """A client with a longer TTL should not call the same data stale."""
        assert not Fetched(data={}, age_seconds=120.0, ttl_seconds=300.0).is_stale
        assert Fetched(data={}, age_seconds=120.0, ttl_seconds=60.0).is_stale


class TestSearch:
    @respx.mock
    async def test_passes_query_through(self, client: CoinGeckoClient) -> None:
        route = respx.get(SEARCH_URL).mock(
            return_value=httpx.Response(200, json={"coins": []})
        )

        await client.search("btc")

        assert route.calls[0].request.url.params["query"] == "btc"


class TestCacheStats:
    @respx.mock
    async def test_counts_hits_and_misses(self, client: CoinGeckoClient) -> None:
        respx.get(PRICE_URL).mock(
            return_value=httpx.Response(200, json=BITCOIN_RESPONSE)
        )

        await client.simple_price(["bitcoin"])
        await client.simple_price(["bitcoin"])

        stats = client.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entries"] == 1
