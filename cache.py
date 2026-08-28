"""A small in-memory TTL cache.

Deliberately not Redis, not `diskcache`, not `functools.lru_cache`.

An MCP server over stdio is one process serving one client, so there is no
second process to share state with and nothing to survive a restart — the
cache is cold every launch either way. An external store would add a
dependency, a connection to fail, and a service to run, to solve a problem
this deployment does not have.

`lru_cache` is the closer call, but it has no expiry: cached prices would be
returned forever. Staleness is the whole point here, so time has to be part
of the design rather than something bolted on.

The clock is injected so tests can advance time without sleeping.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CacheEntry:
    """A stored value and the moment it was stored."""

    value: Any
    stored_at: float


class TTLCache:
    """Maps keys to values that expire after a fixed number of seconds.

    Expired entries are retained rather than deleted, so a caller whose live
    fetch fails can fall back to stale data — as long as it says so. See
    `get_stale`.
    """

    def __init__(
        self,
        ttl_seconds: float = 60.0,
        max_entries: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            ttl_seconds: How long an entry counts as fresh.
            max_entries: Cap on stored entries; the oldest is evicted first.
            clock: Returns a monotonically increasing time in seconds.
                Defaults to `time.monotonic`, which is immune to system
                clock adjustments — wall-clock time can jump backwards on
                an NTP correction and make entries look newer than they are.
        """
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._entries: dict[str, CacheEntry] = {}

        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        """Return the value if present and still fresh, else None."""
        entry = self._entries.get(key)

        if entry is None or self.age(key) > self.ttl_seconds:
            self.misses += 1
            return None

        self.hits += 1
        return entry.value

    def get_stale(self, key: str) -> tuple[Any, float] | None:
        """Return `(value, age_in_seconds)` for any entry, fresh or expired.

        For fallback when a live fetch fails. Callers must surface the age to
        the user: silently returning stale prices as current is worse than
        returning an error.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        return entry.value, self.age(key)

    def set(self, key: str, value: Any) -> None:
        """Store a value, evicting the oldest entry if at capacity."""
        if key not in self._entries and len(self._entries) >= self.max_entries:
            oldest = min(self._entries, key=lambda k: self._entries[k].stored_at)
            del self._entries[oldest]

        self._entries[key] = CacheEntry(value=value, stored_at=self._clock())

    def age(self, key: str) -> float:
        """Seconds since the entry was stored. Infinite if absent."""
        entry = self._entries.get(key)
        if entry is None:
            return float("inf")
        return self._clock() - entry.stored_at

    def clear(self) -> None:
        """Drop all entries and reset counters."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from fresh cache. Zero if never used."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def make_key(endpoint: str, params: dict[str, Any]) -> str:
    """Build a stable cache key from an endpoint and its query parameters.

    Parameters are sorted so that argument order does not produce two keys
    for one request.
    """
    if not params:
        return endpoint
    encoded = "&".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{endpoint}?{encoded}"
