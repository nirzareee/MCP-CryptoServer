"""Tests for the TTL cache.

No mocking library appears here. The cache takes its clock as a constructor
argument, so time can be moved forward by assignment instead of by sleeping.
A suite that sleeps is a suite nobody runs.
"""

import pytest

from cache import TTLCache, make_key


class FakeClock:
    """A clock the test controls.

    Callable so it can be passed straight in as `clock=`.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache(clock: FakeClock) -> TTLCache:
    return TTLCache(ttl_seconds=60.0, clock=clock)


class TestBasicStorage:
    def test_returns_stored_value(self, cache: TTLCache) -> None:
        cache.set("k", {"price": 100})
        assert cache.get("k") == {"price": 100}

    def test_returns_none_for_unknown_key(self, cache: TTLCache) -> None:
        assert cache.get("never-stored") is None

    def test_overwrites_existing_key(self, cache: TTLCache) -> None:
        cache.set("k", 1)
        cache.set("k", 2)
        assert cache.get("k") == 2
        assert cache.size == 1

    def test_stores_falsy_values(self, cache: TTLCache) -> None:
        """A cached empty dict must not read as a cache miss.

        `get` returns None for "absent", so any implementation that checks
        truthiness instead of identity would refetch every time an endpoint
        legitimately returns {} or 0.
        """
        cache.set("empty", {})
        assert cache.get("empty") == {}


class TestExpiry:
    def test_value_survives_within_ttl(
        self, cache: TTLCache, clock: FakeClock
    ) -> None:
        cache.set("k", "v")
        clock.advance(59.0)
        assert cache.get("k") == "v"

    def test_value_expires_after_ttl(self, cache: TTLCache, clock: FakeClock) -> None:
        cache.set("k", "v")
        clock.advance(61.0)
        assert cache.get("k") is None

    def test_boundary_is_inclusive(self, cache: TTLCache, clock: FakeClock) -> None:
        """Exactly at the TTL the entry is still fresh; a hair past, it is not.

        Pinning the boundary matters because `>` and `>=` both look correct
        while reading the code.
        """
        cache.set("k", "v")
        clock.advance(60.0)
        assert cache.get("k") == "v"

        clock.advance(0.001)
        assert cache.get("k") is None

    def test_reset_on_overwrite(self, cache: TTLCache, clock: FakeClock) -> None:
        """Re-storing a key restarts its lifetime rather than keeping the old one."""
        cache.set("k", "first")
        clock.advance(59.0)
        cache.set("k", "second")
        clock.advance(59.0)
        assert cache.get("k") == "second"


class TestStaleFallback:
    def test_expired_entry_still_retrievable(
        self, cache: TTLCache, clock: FakeClock
    ) -> None:
        cache.set("k", "v")
        clock.advance(300.0)

        assert cache.get("k") is None  # not fresh

        stale = cache.get_stale("k")
        assert stale is not None
        value, age = stale
        assert value == "v"
        assert age == pytest.approx(300.0)

    def test_stale_returns_none_when_never_stored(self, cache: TTLCache) -> None:
        assert cache.get_stale("nothing") is None

    def test_age_is_infinite_for_missing_key(self, cache: TTLCache) -> None:
        assert cache.age("nothing") == float("inf")


class TestEviction:
    def test_evicts_oldest_when_full(self, clock: FakeClock) -> None:
        cache = TTLCache(ttl_seconds=60.0, max_entries=3, clock=clock)

        for i in range(3):
            cache.set(f"k{i}", i)
            clock.advance(1.0)

        assert cache.size == 3

        cache.set("k3", 3)

        assert cache.size == 3
        assert cache.get_stale("k0") is None  # oldest, gone
        assert cache.get("k3") == 3

    def test_overwrite_at_capacity_does_not_evict(self, clock: FakeClock) -> None:
        """Updating an existing key is not a new entry, so nothing is dropped."""
        cache = TTLCache(ttl_seconds=60.0, max_entries=2, clock=clock)
        cache.set("a", 1)
        clock.advance(1.0)
        cache.set("b", 2)

        cache.set("a", 99)

        assert cache.size == 2
        assert cache.get("a") == 99
        assert cache.get("b") == 2


class TestCounters:
    def test_hits_and_misses(self, cache: TTLCache) -> None:
        cache.set("k", "v")

        cache.get("k")
        cache.get("k")
        cache.get("absent")

        assert cache.hits == 2
        assert cache.misses == 1
        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_hit_rate_is_zero_before_any_lookup(self, cache: TTLCache) -> None:
        """Guards the division; an unused cache must not raise."""
        assert cache.hit_rate == 0.0

    def test_expired_lookup_counts_as_miss(
        self, cache: TTLCache, clock: FakeClock
    ) -> None:
        cache.set("k", "v")
        clock.advance(61.0)
        cache.get("k")
        assert cache.misses == 1

    def test_clear_resets_everything(self, cache: TTLCache) -> None:
        cache.set("k", "v")
        cache.get("k")
        cache.clear()

        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0


class TestConstructorValidation:
    @pytest.mark.parametrize("ttl", [0, -1, -0.5])
    def test_rejects_non_positive_ttl(self, ttl: float) -> None:
        with pytest.raises(ValueError):
            TTLCache(ttl_seconds=ttl)

    @pytest.mark.parametrize("size", [0, -1])
    def test_rejects_non_positive_max_entries(self, size: int) -> None:
        with pytest.raises(ValueError):
            TTLCache(max_entries=size)


class TestKeyBuilding:
    def test_param_order_does_not_matter(self) -> None:
        """The point of sorting: one request must not produce two keys."""
        a = make_key("/simple/price", {"ids": "bitcoin", "vs_currencies": "usd"})
        b = make_key("/simple/price", {"vs_currencies": "usd", "ids": "bitcoin"})
        assert a == b

    def test_different_params_differ(self) -> None:
        a = make_key("/simple/price", {"ids": "bitcoin"})
        b = make_key("/simple/price", {"ids": "ethereum"})
        assert a != b

    def test_different_endpoints_differ(self) -> None:
        assert make_key("/a", {"x": 1}) != make_key("/b", {"x": 1})

    def test_empty_params_gives_bare_endpoint(self) -> None:
        assert make_key("/search/trending", {}) == "/search/trending"
